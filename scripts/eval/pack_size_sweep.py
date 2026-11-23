"""Clean pack-size capacity sweep for an attention-autoencoder checkpoint.

Reconstructs gold windows at increasing numbers of positions per latent, using
the same protocol as the 6M post-mortem (192 windows, no MASK tokens).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.dataset.token_stream import (
    CLS,
    EMPTY,
    PAD,
    PIECE_BASE,
    SEP,
    TURN_WHITE,
)
from src.modeling.factory import build_model, model_config_from_dict
from src.training.window_dataset import load_game_pool

SIZES = (1, 2, 3, 5, 8, 12, 16, 19, 24, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--windows", type=int, default=192)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-len", type=int, default=2048)
    return parser.parse_args()


def build_pack(segments, max_len: int):
    row = np.full(max_len, PAD, dtype=np.int64)
    cursor = 0
    row[cursor] = CLS
    cursor += 1
    for segment in segments:
        row[cursor : cursor + segment.size] = segment
        cursor += segment.size
        row[cursor] = SEP
        cursor += 1
    return row, np.arange(max_len) < cursor


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(model_config_from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    print(f"checkpoint step={checkpoint.get('step')}", flush=True)

    games = load_game_pool(args.data_dir, args.games, prefetch_shards=2)
    windows = []
    for game in games:
        segments = [segment for segment in game if segment.size == 67]
        if len(segments) >= 30:
            windows.append(segments[:30])
        if len(windows) >= args.windows:
            break
    print(f"games={len(games)} full windows={len(windows)}", flush=True)

    results = {}
    for size in SIZES:
        rows, masks = [], []
        for window in windows:
            row, mask = build_pack(window[:size], args.max_len)
            rows.append(row)
            masks.append(mask)
        tokens = torch.from_numpy(np.stack(rows)).to(device)
        padding = torch.from_numpy(np.stack(masks)).to(device)
        segment_ids = torch.cumsum(tokens == SEP, dim=1)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            latent = model.encode_latent(tokens, segment_ids, padding)
            hidden = model.decode(latent, tokens, segment_ids, padding)
            logits = model.head(hidden).float()
        predictions = logits.argmax(-1)
        keep = padding & (tokens >= EMPTY)
        gold = tokens[keep].cpu().numpy()
        pred = predictions[keep].cpu().numpy()
        empty = gold == EMPTY
        piece = (gold >= PIECE_BASE) & (gold < TURN_WHITE)
        state = gold >= TURN_WHITE
        results[size] = {
            "tokens": int(gold.size),
            "accuracy": float((pred == gold).mean()),
            "empty_recall": float((pred[empty] == EMPTY).mean()),
            "piece_accuracy": float((pred[piece] == gold[piece]).mean()),
            "state_accuracy": float((pred[state] == gold[state]).mean())
            if state.any()
            else None,
            "always_empty": float((pred == EMPTY).mean()),
        }
        print(
            f"{size:2d} positions: acc={results[size]['accuracy']:.3f} "
            f"empty={results[size]['empty_recall']:.3f} "
            f"piece={results[size]['piece_accuracy']:.3f} "
            f"state={results[size]['state_accuracy'] or float('nan'):.3f} "
            f"always-empty={results[size]['always_empty']:.3f}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {out}", flush=True)
    return


if __name__ == "__main__":
    main()
