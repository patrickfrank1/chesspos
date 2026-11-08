from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.evaluation.observability.embeddings import (
    embed_samples,
    save_embeddings,
    save_samples,
)
from src.evaluation.observability.loading import load_checkpoint
from src.evaluation.observability.ranking import (
    surrogate_ranking_examples,
    surrogate_ranking_metrics,
    temporal_ranking_examples,
    temporal_ranking_metrics,
    theme_ranking_examples,
    theme_ranking_metrics,
)
from src.evaluation.observability.render import render_report
from src.evaluation.observability.samples import (
    PositionSample,
    load_puzzle_samples,
    load_test_samples,
)
from src.evaluation.observability.surrogates import SURROGATE_TASKS

DEFAULT_THEMES = (
    "fork",
    "pin",
    "skewer",
    "sacrifice",
    "discoveredAttack",
    "doubleCheck",
    "backRankMate",
    "promotion",
)


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint: str
    out_dir: str
    test_dir: str | None = None
    puzzles: str | None = None
    themes: tuple[str, ...] = DEFAULT_THEMES
    n_test: int = 2_000
    n_puzzles: int = 2_000
    puzzle_position: str = "first"
    per_game_cap: int = 1
    max_ply_distance: int = 5
    rank_k: int = 5
    rank_queries: int = 200
    rank_examples: int = 3
    rank_example_modes: tuple[str, ...] = ("matched",)
    device: str = "cpu"
    batch_size: int = 256
    bf16: bool = False
    seed: int = 0


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _theme_names(samples: Sequence[PositionSample], requested: Sequence[str]):
    present = sorted({theme for sample in samples for theme in sample.themes})
    if requested:
        selected = [theme for theme in requested if theme in present]
        return selected or present
    return present


def _mode_list(selection: Sequence[str]) -> tuple[str, ...]:
    modes: list[str] = []
    for value in selection:
        if value == "both":
            modes.extend(["random", "matched"])
        elif value in ("random", "matched"):
            modes.append(value)
        else:
            raise ValueError(f"Unknown ranking example mode {value!r}")
    return tuple(dict.fromkeys(modes))


def _add_ranking(metrics: dict, base: str, row: dict, *, suffix: str = "") -> None:
    for key in (
        "top1",
        "mrr",
        "mean_rank",
        "n_queries",
        "chance_top1",
        "chance_mrr",
        "chance_mean_rank",
    ):
        metrics[f"{base}{key}{suffix}"] = row[key]
    return


def _flatten_temporal(metrics: dict, result: dict, k: int) -> None:
    base = f"test_temporal_rank{k}_"
    _add_ranking(metrics, base, result["random"])
    _add_ranking(metrics, base, result["matched"], suffix="_matched")
    return


def _flatten_themes(metrics: dict, result: dict, themes: Sequence[str], k: int) -> None:
    base = f"motif_rank{k}_"
    _add_ranking(metrics, base, result["mean_random"], suffix="_mean")
    _add_ranking(metrics, base, result["mean_matched"], suffix="_matched_mean")
    for theme in themes:
        _add_ranking(
            metrics, base, result["per_theme"][theme]["random"], suffix=f"_{theme}"
        )
        _add_ranking(
            metrics,
            base,
            result["per_theme"][theme]["matched"],
            suffix=f"_matched_{theme}",
        )
    return


def _flatten_surrogates(metrics: dict, result: dict, k: int) -> None:
    base = f"surrogate_rank{k}_"
    _add_ranking(metrics, base, result["mean_random"], suffix="_mean")
    _add_ranking(metrics, base, result["mean_matched"], suffix="_matched_mean")
    for task_name, task in result["per_task"].items():
        for class_label, row in task["per_class"].items():
            _add_ranking(
                metrics,
                base,
                row["random"],
                suffix=f"_{task_name}_{class_label}",
            )
            _add_ranking(
                metrics,
                base,
                row["matched"],
                suffix=f"_matched_{task_name}_{class_label}",
            )
    return


def _collect_theme_examples(
    embeddings: np.ndarray,
    samples: Sequence[PositionSample],
    themes: Sequence[str],
    config: EvaluationConfig,
    modes: Sequence[str],
) -> dict[str, dict[str, list[dict]]]:
    examples: dict[str, dict[str, list[dict]]] = {}
    for mode in modes:
        rows = theme_ranking_examples(
            embeddings,
            samples,
            themes,
            k=config.rank_k,
            max_queries=config.rank_queries,
            seed=config.seed,
            matched=mode == "matched",
            max_per_bucket=config.rank_examples,
        )
        examples[mode] = {
            theme: {
                bucket: [
                    row
                    for row in rows
                    if row.get("group") == theme and row.get("bucket") == bucket
                ]
                for bucket in ("best", "random", "worst")
            }
            for theme in themes
        }
    return examples


def run_evaluation(config: EvaluationConfig) -> dict:
    device = torch.device(config.device)
    model, info = load_checkpoint(config.checkpoint, device)
    samples: list[PositionSample] = []
    if config.test_dir:
        samples.extend(
            load_test_samples(
                config.test_dir,
                config.n_test,
                per_game_cap=config.per_game_cap,
                seed=config.seed,
            )
        )
    if config.puzzles:
        samples.extend(
            load_puzzle_samples(
                config.puzzles,
                config.n_puzzles,
                themes=config.themes,
                position=config.puzzle_position,
                seed=config.seed,
            )
        )
    if not samples:
        raise ValueError("No samples loaded; provide --test-dir and/or --puzzles")
    samples = [replace(sample, sample_id=index) for index, sample in enumerate(samples)]
    embeddings = embed_samples(
        model,
        samples,
        device,
        batch_size=config.batch_size,
        bf16=config.bf16,
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_samples(out_dir / "samples.parquet", samples)
    save_embeddings(out_dir / "embeddings.npy", embeddings)

    modes = _mode_list(config.rank_example_modes)
    metrics: dict = {
        "n_samples": len(samples),
        "checkpoint": info.get("path"),
        "step": info.get("step"),
        "val_loss": info.get("val_loss"),
    }

    test_indices = [
        index for index, sample in enumerate(samples) if sample.source == "test"
    ]
    test_metrics = None
    test_examples: dict[str, list[dict]] = {}
    if test_indices:
        test_samples = [samples[index] for index in test_indices]
        test_metrics = temporal_ranking_metrics(
            embeddings[test_indices],
            test_samples,
            k=config.rank_k,
            max_queries=config.rank_queries,
            seed=config.seed,
            max_ply_distance=config.max_ply_distance,
        )
        _flatten_temporal(metrics, test_metrics, config.rank_k)
        for mode in modes:
            rows = temporal_ranking_examples(
                embeddings[test_indices],
                test_samples,
                k=config.rank_k,
                max_queries=config.rank_queries,
                seed=config.seed,
                matched=mode == "matched",
                max_per_bucket=config.rank_examples,
                max_ply_distance=config.max_ply_distance,
            )
            test_examples[mode] = {
                bucket: [row for row in rows if row.get("bucket") == bucket]
                for bucket in ("best", "random", "worst")
            }

    puzzle_indices = [
        index for index, sample in enumerate(samples) if sample.source == "puzzle"
    ]
    motif_metrics = None
    motif_examples: dict[str, dict[str, list[dict]]] = {}
    themes: list[str] = []
    if puzzle_indices:
        puzzle_samples = [samples[index] for index in puzzle_indices]
        themes = _theme_names(puzzle_samples, config.themes)
        if themes:
            motif_metrics = theme_ranking_metrics(
                embeddings[puzzle_indices],
                puzzle_samples,
                themes,
                k=config.rank_k,
                max_queries=config.rank_queries,
                seed=config.seed,
            )
            _flatten_themes(metrics, motif_metrics, themes, config.rank_k)
            motif_examples = _collect_theme_examples(
                embeddings[puzzle_indices],
                puzzle_samples,
                themes,
                config,
                modes,
            )

    surrogate_indices = test_indices or list(range(len(samples)))
    surrogate_samples = [samples[index] for index in surrogate_indices]
    surrogate_metrics = surrogate_ranking_metrics(
        embeddings[surrogate_indices],
        surrogate_samples,
        SURROGATE_TASKS,
        k=config.rank_k,
        max_queries_per_class=config.rank_queries,
        seed=config.seed,
    )
    _flatten_surrogates(metrics, surrogate_metrics, config.rank_k)
    surrogate_examples = surrogate_ranking_examples(
        embeddings[surrogate_indices],
        surrogate_samples,
        SURROGATE_TASKS,
        k=config.rank_k,
        max_queries_per_class=config.rank_queries,
        seed=config.seed,
        modes=modes,
        max_per_bucket=config.rank_examples,
    )

    examples = {
        "test": test_examples,
        "motifs": motif_examples,
        "surrogates": surrogate_examples,
    }
    (out_dir / "ranking_examples.json").write_text(
        json.dumps(_jsonable(examples), indent=2)
    )
    metrics = _jsonable(metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (out_dir / "config.json").write_text(
        json.dumps(_jsonable(asdict(config)), indent=2, sort_keys=True)
    )

    report_data = {
        "info": info,
        "config": asdict(config),
        "n_samples": len(samples),
        "test": {
            "n": len(test_indices),
            "metrics": test_metrics,
            "examples": test_examples,
        },
        "motifs": {
            "n": len(puzzle_indices),
            "themes": themes,
            "metrics": motif_metrics,
            "examples": motif_examples,
        },
        "surrogates": {
            "n": len(surrogate_indices),
            "source": "test" if test_indices else "all",
            "tasks": [asdict(task) for task in SURROGATE_TASKS],
            "metrics": surrogate_metrics,
            "examples": surrogate_examples,
        },
    }
    render_report(
        out_dir / "report.html",
        report=_jsonable(report_data),
        modes=modes,
    )
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description="Observe position similarity with focused 5-way ranking tasks"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--puzzles", default=None)
    parser.add_argument("--themes", default=",".join(DEFAULT_THEMES))
    parser.add_argument("--n-test", type=int, default=2_000)
    parser.add_argument("--n-puzzles", type=int, default=2_000)
    parser.add_argument(
        "--puzzle-position", choices=("first", "last", "random"), default="first"
    )
    parser.add_argument("--per-game-cap", type=int, default=1)
    parser.add_argument(
        "--max-ply-distance",
        type=int,
        default=5,
        help="Temporal positive is sampled within this many plies",
    )
    parser.add_argument("--rank-k", type=int, default=5)
    parser.add_argument(
        "--rank-queries",
        type=int,
        default=200,
        help="Queries per theme/class for ranking metrics",
    )
    parser.add_argument(
        "--rank-examples",
        type=int,
        default=3,
        help="Examples per best/random/worst bucket per condition",
    )
    parser.add_argument(
        "--rank-example-mode",
        choices=("random", "matched", "both"),
        default="matched",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return EvaluationConfig(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        test_dir=args.test_dir,
        puzzles=args.puzzles,
        themes=tuple(
            theme.strip() for theme in args.themes.split(",") if theme.strip()
        ),
        n_test=args.n_test,
        n_puzzles=args.n_puzzles,
        puzzle_position=args.puzzle_position,
        per_game_cap=args.per_game_cap,
        max_ply_distance=args.max_ply_distance,
        rank_k=args.rank_k,
        rank_queries=args.rank_queries,
        rank_examples=args.rank_examples,
        rank_example_modes=(args.rank_example_mode,),
        device=device,
        batch_size=args.batch_size,
        bf16=args.bf16,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    metrics = run_evaluation(config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"report written to {Path(config.out_dir) / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
