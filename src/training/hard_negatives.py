from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from src.dataset.token_stream import (
    CASTLE_BASE,
    EP_BASE,
    N_SLOTS,
    SEGMENT_LEN,
    TURN_BLACK,
    TokenStreamEncoder,
)
from src.training.contrastive import build_contrastive_masks
from src.training.objectives import (
    extract_alone_views_multi,
    restore_visible_tokens,
    select_target_segments_multi,
)

if TYPE_CHECKING:
    from src.modeling.transformer import MaskedStreamTransformer
    from src.training.objectives import AuxiliaryLossConfig

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _material(board: chess.Board) -> int:
    return sum(
        _PIECE_VALUES[piece.piece_type] * (1 if piece.color else -1)
        for piece in board.piece_map().values()
    )


def _piece_distance(left: chess.Board, right: chess.Board) -> int:
    a, b = left.piece_map(), right.piece_map()
    return sum(1 for square in set(a) | set(b) if a.get(square) != b.get(square))


def collect_hard_negatives(
    model: MaskedStreamTransformer,
    batch: dict[str, torch.Tensor],
    config: AuxiliaryLossConfig,
    *,
    step: int,
    top_k: int = 100,
    generator: torch.Generator | None = None,
) -> list[dict]:
    """Return the hardest SupCon negatives in one training batch.

    The returned rows are the top-``top_k`` anchors by per-anchor SupCon loss,
    each paired with its hardest non-ignored negative. Opening views are
    excluded from the negative set when ``config.ignore_opening_negatives`` is
    enabled.
    """
    was_training = model.training
    model.eval()
    try:
        tokens = batch["tokens"]
        segment_ids = batch["segment_ids"]
        padding = batch["padding_mask"]
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=tokens.device.type,
                dtype=torch.bfloat16,
                enabled=tokens.device.type == "cuda",
            ),
        ):
            _, segment_mask = model.embed(tokens, segment_ids, padding)
            targets_per_row = select_target_segments_multi(
                segment_mask,
                count=config.alone_targets_per_window,
                generator=generator,
            )
            visible = restore_visible_tokens(tokens, batch["targets"])
            (
                alone_tokens,
                alone_segments,
                alone_padding,
                view_rows,
                view_targets,
            ) = extract_alone_views_multi(
                visible, segment_ids, padding, targets_per_row
            )
            content, _, _ = model.encode_views(
                alone_tokens, alone_segments, alone_padding
            )
            embeddings = F.normalize(content[:, 0].float(), dim=-1)
    finally:
        model.train(was_training)

    rows_tensor = view_rows.detach().cpu()
    targets_tensor = view_targets.detach().cpu()
    rows = rows_tensor.tolist()
    targets = targets_tensor.tolist()
    if "game_id" in batch and "start_index" in batch:
        view_games = batch["game_id"].detach().cpu()[rows_tensor].tolist()
        view_plies = (
            batch["start_index"].detach().cpu()[rows_tensor] + targets_tensor
        ).tolist()
    else:
        view_games = rows
        view_plies = targets

    opening = None
    if config.ignore_opening_negatives:
        opening = [bool(ply < config.opening_ply_threshold) for ply in view_plies]
    positive, ignore = build_contrastive_masks(
        view_games,
        view_plies,
        temporal_radius=config.temporal_radius,
        opening=opening,
    )
    positive = positive.to(embeddings.device)
    ignore = ignore.to(embeddings.device)
    identity = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    ignored = ignore | identity
    logits = embeddings @ embeddings.T / max(config.nce_temperature, 1e-8)
    logits = logits.masked_fill(ignored, float("-inf"))
    positives = positive & ~ignored

    records = []
    encoder = TokenStreamEncoder()
    for index in torch.nonzero(positives.any(dim=1)).flatten().tolist():
        candidates = ~positives[index] & torch.isfinite(logits[index])
        if not bool(candidates.any()):
            continue
        pos_logits = logits[index][positives[index]]
        loss = float(-(pos_logits.mean() - torch.logsumexp(logits[index], dim=0)))
        negative_scores = logits[index].masked_fill(~candidates, float("-inf"))
        neighbor = int(torch.argmax(negative_scores))
        raw_anchor = alone_tokens[index, 1 : 1 + SEGMENT_LEN]
        raw_negative = alone_tokens[neighbor, 1 : 1 + SEGMENT_LEN]
        anchor_tokens = raw_anchor.detach().cpu().numpy().astype(np.int16)
        negative_tokens = raw_negative.detach().cpu().numpy().astype(np.int16)
        anchor = encoder.decode(anchor_tokens)
        negative = encoder.decode(negative_tokens)
        anchor_opening = bool(opening[index]) if opening is not None else False
        negative_opening = bool(opening[neighbor]) if opening is not None else False
        records.append(
            {
                "step": int(step),
                "loss": loss,
                "similarity": float(embeddings[index] @ embeddings[neighbor]),
                "positive_count": int(positives[index].sum()),
                "piece_distance": _piece_distance(anchor, negative),
                "anchor_fen": anchor.fen(),
                "negative_fen": negative.fen(),
                "anchor_tokens": anchor_tokens.tolist(),
                "negative_tokens": negative_tokens.tolist(),
                "anchor_material": _material(anchor),
                "negative_material": _material(negative),
                "anchor_piece_count": len(anchor.piece_map()),
                "negative_piece_count": len(negative.piece_map()),
                "anchor_ply": int(view_plies[index]),
                "negative_ply": int(view_plies[neighbor]),
                "anchor_game": int(view_games[index]),
                "negative_game": int(view_games[neighbor]),
                "anchor_turn": int(anchor_tokens[N_SLOTS] == TURN_BLACK),
                "negative_turn": int(negative_tokens[N_SLOTS] == TURN_BLACK),
                "anchor_castle": int(anchor_tokens[N_SLOTS + 1] - CASTLE_BASE),
                "negative_castle": int(negative_tokens[N_SLOTS + 1] - CASTLE_BASE),
                "anchor_ep": int(anchor_tokens[N_SLOTS + 2] - EP_BASE),
                "negative_ep": int(negative_tokens[N_SLOTS + 2] - EP_BASE),
                "anchor_opening": anchor_opening,
                "negative_opening": negative_opening,
            }
        )

    records.sort(key=lambda record: record["loss"], reverse=True)
    if top_k > 0:
        records = records[:top_k]
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    return records


def write_hard_negatives(
    records: list[dict],
    out_dir: str | Path,
    step: int,
) -> Path | None:
    """Write one Parquet shard of hard negatives under ``out_dir``."""
    if not records:
        return None
    directory = Path(out_dir) / "hard_negatives"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{step:06d}.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    return path
