# Design: Piece-Square Token Stream Encoding

- **Status:** Proposed
- **Date:** 2026-10-03
- **Scope:** New position encoding (`token_stream`), its encoder/decoder, and the
  token/embedding conventions downstream models should follow.
- **Out of scope (separate changes):** window emission in the ETL, the factorized
  embedding layer + model config changes, loader-side masking updates, dataset
  schema migration.

---

## 1. Context

The current `TokenSequenceEncoder` (`src/dataset/position_encoder.py:45`) encodes a
position as a fixed-length `(69,) int8` vector: 64 square slots (mostly `empty`)
plus 5 state tokens. This works for single positions but is a poor unit for the
project goal of encoding *positions, windows, and whole games as one token
stream*:

- **Padding tax.** ~49 of 69 tokens are `empty`. Attention cost is O(L²), so
  padding is real compute, not free.
- **No composition.** There is no separator concept; concatenating positions
  produces an ambiguous stream with no per-position boundary.
- **Length.** An 80-ply game at 69 tokens/position is ~5,520 tokens — too long
  for cheap training. A 10-position window is 690.
- **Fixed shape coupling.** `SEQUENCE_LENGTH = 69` and `VOCABULARY_SIZE = 33`
  are hardcoded in `src/modeling/model.py:185-186`.

Requirements for the replacement:

1. One flat token stream per sample; position / window / game are the same
   format at different granularities.
2. Sequences short enough that training is cheap (target: windows of 8–12
   positions fit a 512-token budget).
3. Everything in the stream must be a *property of the position* and
   reconstructable from context — no target leakage.
4. Deterministic round trip: `decode(encode(board))` reproduces the board.

## 2. Decision

### 2.1 Composite piece-square tokens

Encode **one token per occupied square**, where the token ID jointly identifies
the piece *and* its square, plus a small set of state tokens. Boards are
encoded as a piece list, not as 64 square slots.

Rationale:

- A position has ≤ 32 pieces; a piece list costs ~32 tokens in the opening and
  ~10–15 in the endgame, vs. a fixed 69. Roughly a 2× average reduction, much
  more late-game.
- Baking the square into the token makes the board coordinate part of the
  token *content*, which enables factorized embeddings (§2.5) and removes the
  need for a 1D sequence position embedding (the canonical sort order makes
  stream position information-free).
- Jointly masking a piece-square token forces the model to predict piece
  *and* square (768-way), i.e. masked prediction becomes a spatial reasoning
  task — a deliberate feature for an embedding model.

### 2.2 Vocabulary layout (806 tokens)

| Token IDs | Count | Meaning |
|-----------|-------|---------|
| 0–3       | 4     | Specials: `PAD=0`, `MASK=1`, `SEP=2`, `CLS=3` |
| 4–5       | 2     | `TURN_WHITE=4`, `TURN_BLACK=5` |
| 6–21      | 16    | `CASTLE_<bits>` — 4-bit bitmask, bit order W-K, W-Q, B-K, B-Q (`CASTLE_0000`=6 … `CASTLE_1111`=21) |
| 22–37     | 16    | `EP_<square>` — ep target square. Only rank 3/6 squares are reachable, so 16 tokens (`EP_a3`=22 … `EP_h3`=29, `EP_a6`=30 … `EP_h6`=37) |
| 38–805    | 768   | Piece-square tokens: `id = 38 + 64·piece + square`, piece ∈ {wP=0, wN=1, wB=2, wR=3, wQ=4, wK=5, bP=6, …, bK=11}, square ∈ 0–63 (a1=0 … h8=63, python-chess order) |

Helper: `SPECIAL`, `TURN_*`, `CASTLE_*`, `EP_*`, and `piece_square_id(piece,
square)` / `parse_piece_square(id)` constants/functions live next to the
encoder.

**dtype: `int16` (not `int8`).** IDs exceed 255, so the current `int8` is no
longer sufficient. 2 bytes × ~35 tokens is still cheaper than 1 byte × 69.

### 2.3 Segment grammar (what `encode(board)` emits)

`encode(board)` returns one **segment** — a variable-length `int16` vector:

```
segment := TURN CASTLE [EP] piece*
piece   := piece-square token, sorted by square ascending
```

- State tokens first (`TURN`, `CASTLE`, optional `EP`), then pieces sorted by
  square ascending. The order is canonical so that encoding is deterministic
  and decoding is unambiguous.
- `EP` is emitted **only when `board.has_legal_en_passant()`** — a pseudo-legal
  ep square that cannot actually be captured is not encoded (it has no effect
  on the position's legal move set). This keeps the ep token cost at ~0 on
  average (legal ep occurs in ~1–3% of positions).
- Castling is one 16-way token, not four binary tokens.
- Not encoded (deliberately): en-passant *capture legality beyond the target
  square*, halfmove clock, fullmove number, ply. These are either not needed
  for the embedding or are provided alongside (see §2.6).

### 2.4 Stream grammar (position / window / game)

Packing is a separate step from encoding:

```
stream  := [CLS] segment (SEP segment)* [SEP]
```

- `encode_window(boards)` / a pack helper inserts `SEP` between segments and
  optionally a leading `CLS`.
- A single position = one segment (no `CLS`/`SEP` required).
- A window = k consecutive segments. A game = sampled positions from a game,
  truncated/packed to the model's `MAX_SEQ_LEN` (512 recommended).
- Granularity is therefore a *packing decision*, not a new encoder — this is
  what makes "concatenate those vectors" a non-event.

**Sequence-length budget** (avg ~32 piece tokens + 2–3 state tokens, ~50%
fewer for endgames):

| Unit | Approx. tokens | Fits 512? |
|------|----------------|-----------|
| 1 position | ~35 | yes |
| Window of 8 | ~280 | yes |
| Window of 12 | ~420 | yes |
| 80-ply game, every 4th ply (20 positions) | ~700 | no (sample/truncate) |

### 2.5 Embedding conventions (for the model side)

Two coordinate systems apply to every token: **board space** (file, rank) and
**stream space** (slot in sequence, board index in window). The token stream
already carries board space in the token ID; stream space must be added at the
embedding layer:

```
E(token) = E_piece[piece] + E_square[square] + E_segment[board_index]
```

- `E_piece + E_square` is a **factorized** replacement for a learned 806×d
  table (12d + 64d params instead of 806d) and doubles as the 2D position
  encoding (ViT-style row+column factorization). State tokens get their own
  small embedding or map to a reserved piece index.
- `E_segment` is a BERT-style segment embedding over the board index within
  the window (`E_segment[0]` for single positions). This supplies the temporal
  axis without a 1D position embedding.
- **Drop the 1D `PositionEmbedding`** (`model.py:207`): because pieces are
  canonically sorted by square, stream position carries no information, and
  removing it gives length generalization for free.
- Optional later refinement: per-head relative attention bias on square delta
  `b(Δfile, Δrank)` (15×15 table per head) if local geometric relations prove
  hard to learn from absolute square embeddings alone.
- Pooling: `CLS` embedding or mean-pool over the target segment — to be decided
  in the model change, not here.

### 2.6 What is *not* in the stream, and where it lives

Rule: only positional facts belong in tokens; everything else is a dataset
column provided alongside.

| Info | In stream? | Rationale |
|------|-----------|-----------|
| Pieces + squares | yes | the content |
| Side to move | yes (`TURN`) | positional fact; alternation is free temporal signal |
| Castling rights | yes (`CASTLE`) | positional fact; affects legality |
| Legal en passant | yes, conditional | positional fact; near-zero average cost |
| Ply | no | reconstructable from turn parity within a window; dataset column |
| Result, white/black ELO, opening, event, date | no | not positional, not reconstructable (MLM leak/noise); dataset columns, usable later as conditioning on the pooled embedding |

### 2.7 Masking semantics (training)

- Masking selects **piece-square tokens** and replaces them with `MASK=1`;
  state tokens may optionally be masked as a separate config.
- Prediction target is the original token ID (768-way for piece-squares). The
  model must infer both identity and square from context.
- `PAD=0` fills ragged tails in collated batches and is excluded from the loss.

## 3. Consequences / trade-offs

- **+** ~50% average token reduction vs. `(69,)`, far more in endgames; cheap
  windows of 8–12 positions.
- **+** One format for position/window/game; no per-granularity encoders.
- **+** Factorized embeddings shrink the vocabulary table ~35×.
- **−** Variable-length sequences: ragged storage (HF `Sequence` feature) and
  collate-time padding are required; `output_shape` is no longer a static
  tuple.
- **−** `int16` doubles per-token bytes (net win given fewer tokens).
- **−** Masked prediction is 768-way joint piece+square — harder than the
  12-way square-content alternative; accepted deliberately (§2.1).
- **−** New vocabulary is *not* backward compatible with `token_sequence`
  datasets; both encoders coexist via the registry during migration.

## 4. Implementation plan

All changes are additive; `token_sequence` remains the default until the ETL
and loader migrate.

### 4.1 `src/dataset/types.py`

```python
TOKEN_STREAM: EncodingFormat = "token_stream"
```

`ENCODING_SHAPES` is *not* extended (the format is variable-length); code that
requires a static shape must special-case `TOKEN_STREAM` or read lengths.

### 4.2 `src/dataset/token_stream.py` (new module)

Keeps the vocabulary out of `position_encoder.py` to limit that file's growth.

```python
PAD, MASK, SEP, CLS = 0, 1, 2, 3
TURN_WHITE, TURN_BLACK = 4, 5
CASTLE_BASE = 6                      # + 4-bit mask -> 6..21
EP_BASE = 22                         # + ep square index -> 22..37
PIECE_SQUARE_BASE = 38               # + 64*piece + square -> 38..805
VOCAB_SIZE = 806

PIECE_INDEX = {chess.PAWN: 0, ..., chess.KING: 5}   # per color, +6 for black

def piece_square_id(piece: chess.Piece, square: int) -> int: ...
def parse_piece_square(token: int) -> tuple[chess.Piece, int]: ...
def castle_token(board: chess.Board) -> int: ...    # 4-bit mask -> 6..21
def ep_token(board: chess.Board) -> int | None: ... # None unless legal ep

@dataclass
class TokenStreamEncoder(PositionEncoder):
    encoding_format = TOKEN_STREAM
    output_shape = None               # variable length; document why

    def encode(self, board: chess.Board) -> np.ndarray:
        # 1. state tokens: TURN, CASTLE, EP (only if board.has_legal_en_passant())
        # 2. piece tokens for board.piece_map(), sorted by square ascending
        # 3. return np.array(tokens, dtype=np.int16)

    def decode(self, data: np.ndarray) -> chess.Board:
        # 1. first token must be TURN_* -> board.turn
        # 2. second must be CASTLE_* -> board.set_castling_fen from bitmask
        # 3. optional EP_* -> board.ep_square (strip trailing bits as python-chess does)
        # 4. remaining tokens must be piece-squares; reject duplicates/overlap
        # 5. build board via set_piece_map; raise ValueError on malformed input

    def encode_batch(self, boards) -> tuple[np.ndarray, np.ndarray]:
        # pad with PAD to max length; return (padded (N, L) int16, lengths (N,))

    def decode_batch(self, data: np.ndarray, lengths=None) -> list[chess.Board]:
        # decode rows, trimming PAD when lengths given
```

Additionally a pure function (not on the class, usable by Ray callables):

```python
def pack_stream(segments: list[np.ndarray], add_cls: bool = True) -> np.ndarray:
    # CLS + seg0 + SEP + seg1 + ... + SEP, dtype int16
```

### 4.3 Registry wiring

- `position_encoder.py`: import and `register_encoder(TOKEN_STREAM,
  TokenStreamEncoder)` — via the registry dict, no ABC change. (`PositionEncoder`
  ABC stays unchanged; `output_shape` returns `None` and is documented as
  variable-length for this format.)
- `__init__.py`: export `TokenStreamEncoder` and the special-token constants.

### 4.4 Tests (`test/`)

1. **Round trip (property):** for N random games (fixed seed), for every ply:
   `decode(encode(board)).fen() == board.fen()` — covers castling changes,
   promotion (all 4 promotion pieces), ep-capture positions, both colors.
2. **Ep condition:** a double push whose ep capture is legal yields the `EP`
   token; a position with only pseudo-legal ep does not.
3. **Canonical order:** encode output is identical across runs; piece tokens
   strictly increasing in square.
4. **Castling:** all 16 `CASTLE_*` tokens round trip.
5. **Segment/pack:** `pack_stream` output parses as `[CLS]? seg (SEP seg)* SEP`;
   `decode` of a single segment ignores `CLS`/`SEP` (decode should tolerate or
   reject them explicitly — pick one and test).
6. **Batch padding:** `encode_batch` pads to batch max; `decode_batch` with
   lengths reproduces boards.
7. **dtype/values:** `int16`, all IDs in `[0, 806)`.

### 4.5 Sequencing

1. `token_stream.py` + tests (pure python-chess/numpy, no Ray/HF involvement).
2. Registry + `__init__.py` export.
3. Land; migrate ETL window emission, model embeddings, and loader masking in
   separate changes that consume this one.

### 4.6 Acceptance criteria

- `uv run pytest` green, including the new property tests.
- `uv run ruff format && uv run ruff check` clean.
- Round-trip holds on ≥ 1,000 plies of random games including promotions,
  ep captures, and castling transitions.
- Encoder obtainable via `get_encoder("token_stream")`; existing formats
  untouched and their tests still green.
