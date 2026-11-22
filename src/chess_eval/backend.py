from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import chess
import numpy as np
import torch
from torch import nn

from src.dataset.game_outcomes import state_features
from src.dataset.puzzle_variants import pack_segments
from src.dataset.token_stream import N_SLOTS, SEGMENT_LEN, TokenStreamEncoder
from src.run.classify_position import load_classifier
from src.run.eval_score_probe import load_score_probe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOTS = (
    REPO_ROOT / "data/models/downstream/score",
    REPO_ROOT / "data/models/downstream/sacrifice",
    REPO_ROOT / "data/models/score_probe",
    REPO_ROOT / "data/models/puzzle_classifier",
)

_MODEL_CACHE: dict[tuple[str, str], ModelHandle] = {}
_TOKEN_ENCODER = TokenStreamEncoder()

_TASK_ORDER = {"sacrifice": 0, "score": 1}
_MODEL_NAMES = {
    "run_20m_lazy": "20M",
    "run_6m_isolated": "6M",
}
_RECIPE_NAMES = {
    "native": "frozen native",
    "full": "frozen full",
    "native_attn": "frozen native attn",
    "frozen_mean": "frozen mean",
    "frozen_attn": "frozen attn",
    "ftlast": "last block mean",
    "ftlast_mean": "last block mean",
    "ftlast_attn": "last block attn",
    "unfreeze_all": "full FT mean",
    "unfreeze_all_mean": "full FT mean",
    "unfreeze_all_attn": "full FT attn",
}


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    label: str
    task: str
    path: Path
    recipe: str
    model: str
    seed: int | None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "task": self.task,
            "recipe": self.recipe,
            "model": self.model,
            "seed": self.seed,
        }


@dataclass
class ModelHandle:
    info: ModelInfo
    model: nn.Module
    metrics: dict[str, float]
    metadata: dict


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        resolved = device
    elif device:
        resolved = torch.device(device)
    else:
        name = os.environ.get(
            "CHESSPOS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        resolved = torch.device(name)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()
    return


def discover_models(
    roots: Sequence[str | Path] | None = None,
) -> list[ModelInfo]:
    """Find the preferred checkpoint in each run directory under ``roots``."""
    model_roots = [Path(root) for root in (roots or DEFAULT_MODEL_ROOTS)]
    found: dict[Path, ModelInfo] = {}
    for root in model_roots:
        if not root.exists():
            continue
        by_parent: dict[Path, list[Path]] = {}
        for path in root.rglob("*.pt"):
            by_parent.setdefault(path.parent, []).append(path)
        for parent, files in by_parent.items():
            best = [path for path in files if path.name == "best.pt"]
            last = [path for path in files if path.name == "last.pt"]
            chosen = best or last or sorted(files)
            for path in chosen:
                task = _infer_task(root, path)
                if task is None:
                    continue
                recipe, model, seed = _path_parts(root, path)
                info = ModelInfo(
                    model_id=_model_id(path),
                    label=_pretty_label(recipe, model, seed),
                    task=task,
                    path=path,
                    recipe=recipe,
                    model=model,
                    seed=seed,
                )
                found[path.resolve()] = info
    return sorted(
        found.values(),
        key=lambda info: (
            _TASK_ORDER.get(info.task, 99),
            info.model,
            info.recipe,
            info.seed or 0,
            info.label,
        ),
    )


def catalog_by_id(models: Sequence[ModelInfo]) -> dict[str, ModelInfo]:
    return {info.model_id: info for info in models}


def load_model(
    info: ModelInfo,
    device: str | torch.device | None = None,
) -> ModelHandle:
    resolved = resolve_device(device)
    key = (info.model_id, str(resolved))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    handle = load_checkpoint_handle(info.path, task=info.task, device=resolved)
    handle.info = info
    _MODEL_CACHE[key] = handle
    return handle


def load_checkpoint_handle(
    path: str | Path,
    task: str | None = None,
    device: str | torch.device | None = None,
) -> ModelHandle:
    """Load a checkpoint without scanning a model directory.

    ``task`` is inferred from the checkpoint when omitted: score probes carry a
    ``state_features`` entry in their head config, the sacrifice classifier
    does not.
    """
    path = Path(path)
    resolved = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    resolved_task = task or infer_task(checkpoint)
    info = ModelInfo(
        model_id=str(path),
        label=path.stem,
        task=resolved_task,
        path=path,
        recipe="",
        model="",
        seed=None,
    )
    metrics = {
        name: float(value)
        for name, value in (checkpoint.get("metrics") or {}).items()
        if isinstance(value, (int, float))
    }
    if resolved_task == "score":
        model = load_score_probe(path, resolved)
        metadata = {
            "feature": checkpoint.get("feature", "native"),
            "window": int(checkpoint.get("window", 1) or 1),
            "include_state_features": bool(
                checkpoint.get("include_state_features", False)
            ),
            "token_pooling": checkpoint.get("token_pooling", "mean"),
        }
    elif resolved_task == "sacrifice":
        model = load_classifier(str(path), resolved)
        metadata = {
            "feature": checkpoint.get("feature", "native"),
            "window": 1,
        }
    else:
        raise ValueError(f"unknown task {resolved_task!r}")
    return ModelHandle(info=info, model=model, metrics=metrics, metadata=metadata)


def infer_task(checkpoint: Mapping) -> str:
    head_config = checkpoint.get("head_config") or {}
    if "state_features" in head_config:
        return "score"
    return "sacrifice"


def evaluate_models(
    models: Sequence[ModelInfo],
    boards: Sequence[chess.Board],
    device: str | torch.device | None = None,
) -> list[dict]:
    resolved = resolve_device(device)
    results: list[dict] = []
    for info in models:
        result = {
            "model_id": info.model_id,
            "label": info.label,
            "task": info.task,
            "recipe": info.recipe,
            "model": info.model,
            "seed": info.seed,
            "value": None,
            "error": None,
        }
        try:
            handle = load_model(info, resolved)
            value = evaluate_handle(handle, boards, resolved)
            result.update(
                {
                    "value": value,
                    "metrics": handle.metrics,
                    "metadata": handle.metadata,
                }
            )
            if info.task == "score":
                result["expected_points"] = (value + 1.0) / 2.0
            else:
                threshold = handle.metrics.get("threshold")
                result["percent"] = value * 100.0
                result["threshold"] = threshold
                result["above_threshold"] = bool(
                    threshold is not None and value >= threshold
                )
        except Exception as exc:  # pragma: no cover - exercised by bad checkpoints
            result["error"] = str(exc)
        results.append(result)
    return results


@dataclass(frozen=True)
class ModelBatch:
    """One packed model input plus the token spans of its square slots.

    ``tokens`` is the un-batched row; ``square_positions[i, s]`` is the token
    index of square ``s`` (``chess.A1`` .. ``chess.H8``) in segment ``i``.
    """

    tokens: np.ndarray
    segment_ids: np.ndarray
    padding_mask: np.ndarray
    features: np.ndarray | None
    boards: tuple[chess.Board, ...]
    square_positions: np.ndarray


def model_batch(
    handle: ModelHandle,
    boards: Sequence[chess.Board],
) -> ModelBatch:
    """Pack the boards a handle consumes into its token-stream input.

    Score probes may use a window of preceding positions and optional explicit
    state features; the sacrifice classifier always reads one position.
    """
    if not boards:
        raise ValueError("at least one board is required")
    if handle.info.task == "score":
        window = max(1, int(handle.metadata.get("window", 1) or 1))
        selected = list(boards[-window:])
    elif handle.info.task == "sacrifice":
        selected = [boards[-1]]
    else:
        raise ValueError(f"unknown task {handle.info.task!r}")

    segments = [_TOKEN_ENCODER.encode(board) for board in selected]
    tokens, segment_ids = pack_segments(segments)
    features = None
    if handle.info.task == "score" and handle.metadata.get("include_state_features"):
        features = state_features(segments[-1])
    starts = 1 + np.arange(len(segments), dtype=np.int64) * (SEGMENT_LEN + 1)
    square_positions = starts[:, None] + np.arange(N_SLOTS, dtype=np.int64)[None, :]
    return ModelBatch(
        tokens=tokens,
        segment_ids=segment_ids,
        padding_mask=np.ones(tokens.shape, dtype=bool),
        features=features,
        boards=tuple(board.copy() for board in selected),
        square_positions=square_positions,
    )


def forward_tokens(
    handle: ModelHandle,
    batch: ModelBatch,
    token_rows: np.ndarray,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Evaluate one or more token rows against a handle's packed batch."""
    resolved = resolve_device(device)
    rows = np.asarray(token_rows, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows[None]
    count = rows.shape[0]
    tokens = torch.from_numpy(rows).to(resolved)
    segment_ids = (
        torch.from_numpy(batch.segment_ids).unsqueeze(0).repeat(count, 1).to(resolved)
    )
    padding_mask = (
        torch.from_numpy(batch.padding_mask).unsqueeze(0).repeat(count, 1).to(resolved)
    )
    features = None
    if batch.features is not None:
        features = (
            torch.from_numpy(np.asarray(batch.features, dtype=np.float32))
            .unsqueeze(0)
            .repeat(count, 1)
            .to(resolved)
        )
    with torch.no_grad():
        if handle.info.task == "score":
            output = handle.model(tokens, segment_ids, padding_mask, features)
        else:
            output = torch.sigmoid(handle.model(tokens, segment_ids, padding_mask))
    return output.detach().cpu().numpy().astype(np.float64).reshape(-1)


def evaluate_handle(
    handle: ModelHandle,
    boards: Sequence[chess.Board],
    device: str | torch.device | None = None,
) -> float:
    batch = model_batch(handle, boards)
    return float(forward_tokens(handle, batch, batch.tokens, device)[0])


def replay(
    start_fen: str,
    moves: Sequence[str],
) -> tuple[list[chess.Board], list[str]]:
    """Replay UCI moves and return every board plus the SAN move list."""
    try:
        board = chess.Board(start_fen)
    except ValueError as exc:
        raise ValueError(f"invalid FEN: {exc}") from exc
    boards = [board.copy()]
    san_moves: list[str] = []
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise ValueError(f"invalid move {uci!r}") from exc
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {uci!r}")
        san_moves.append(board.san(move))
        board.push(move)
        boards.append(board.copy())
    return boards, san_moves


def apply_move(
    start_fen: str,
    moves: Sequence[str],
    source: str,
    target: str,
    promotion: str | None = None,
) -> tuple[list[chess.Board], list[str]]:
    boards, _ = replay(start_fen, moves)
    board = boards[-1]
    uci = f"{source}{target}{promotion or ''}"
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as exc:
        raise ValueError(f"invalid move {uci!r}") from exc
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {uci!r}")
    return replay(start_fen, [*moves, uci])


def position_payload(
    start_fen: str,
    moves: Sequence[str],
    boards: Sequence[chess.Board],
    san_moves: Sequence[str],
) -> dict:
    board = boards[-1]
    game_over = board.is_game_over(claim_draw=True)
    material = _material_balance(board)
    return {
        "start_fen": start_fen,
        "fen": board.fen(),
        "moves": list(moves),
        "san_moves": list(san_moves),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "legal_moves": sorted(move.uci() for move in board.legal_moves),
        "material_balance": material,
        "material_text": f"{material:+d}",
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_game_over": game_over,
        "status": _status(board),
        "result": board.result(claim_draw=True) if game_over else None,
        "last_move": moves[-1] if moves else None,
        "move_number": board.fullmove_number,
    }


def state_response(
    start_fen: str,
    moves: Sequence[str],
    model_ids: Sequence[str],
    catalog: Mapping[str, ModelInfo],
    device: str | torch.device | None = None,
) -> dict:
    boards, san_moves = replay(start_fen, moves)
    evaluations: list[dict] = []
    for model_id in model_ids:
        info = catalog.get(model_id)
        if info is None:
            evaluations.append(
                {
                    "model_id": model_id,
                    "label": model_id,
                    "task": None,
                    "value": None,
                    "error": "unknown model id",
                }
            )
            continue
        evaluations.extend(evaluate_models([info], boards, device))
    return {
        "position": position_payload(start_fen, moves, boards, san_moves),
        "evaluations": evaluations,
        "device": str(resolve_device(device)),
    }


def _infer_task(root: Path, path: Path) -> str | None:
    text = f"{root.name}/{path}".lower()
    if "sacrifice" in text or "puzzle" in text:
        return "sacrifice"
    if "score" in text or "probe" in text:
        return "score"
    return None


def _path_parts(root: Path, path: Path) -> tuple[str, str, int | None]:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    if len(parts) >= 4:
        seed_part = parts[-2]
        seed = (
            int(seed_part.removeprefix("seed"))
            if seed_part.startswith("seed") and seed_part[4:].isdigit()
            else None
        )
        return parts[-4], parts[-3], seed
    if len(parts) >= 3:
        return parts[-3], parts[-2], None
    return root.name, path.stem, None


def _pretty_label(
    recipe: str,
    model: str,
    seed: int | None,
) -> str:
    model_name = _MODEL_NAMES.get(model, model)
    recipe_name = _RECIPE_NAMES.get(recipe, recipe.replace("_", " "))
    suffix = f" · seed {seed}" if seed is not None else ""
    return f"{model_name} · {recipe_name}{suffix}"


def _model_id(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _material_balance(board: chess.Board) -> int:
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    balance = 0
    for piece in board.piece_map().values():
        value = values.get(piece.piece_type, 0)
        balance += value if piece.color == chess.WHITE else -value
    return balance


def _status(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_seventyfive_moves():
        return "seventy-five-move rule"
    if board.is_fivefold_repetition():
        return "fivefold repetition"
    if board.is_check():
        return "check"
    return "playing"
