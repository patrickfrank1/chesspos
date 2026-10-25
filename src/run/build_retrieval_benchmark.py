from __future__ import annotations

import argparse

from src.evaluation.retrieval import (
    BENCHMARK_SUBDIR,
    DEFAULT_CLUSTER_THEMES,
    DEFAULT_STRATEGIC_NEGATIVES,
    DEFAULT_STRATEGIC_POSITIVES,
    BenchmarkConfig,
    build_benchmark,
    upload_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a frozen retrieval benchmark from validation puzzles"
    )
    parser.add_argument("--data-dir", default="data/processed/puzzles")
    parser.add_argument("--output-dir", default="data/benchmarks/retrieval")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-positions", type=int, default=10_000)
    parser.add_argument("--max-queries", type=int, default=1_000)
    parser.add_argument("--near-hamming", type=int, default=2)
    parser.add_argument(
        "--strategic-positives",
        type=int,
        default=DEFAULT_STRATEGIC_POSITIVES,
        help="Same-theme positions sampled per strategic query",
    )
    parser.add_argument(
        "--strategic-negatives",
        type=int,
        default=DEFAULT_STRATEGIC_NEGATIVES,
        help="Non-theme positions sampled per strategic query",
    )
    parser.add_argument("--cluster-themes", default=",".join(DEFAULT_CLUSTER_THEMES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HuggingFace dataset repo id to upload the benchmark to",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--commit-message", default="Add retrieval benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cluster_themes = tuple(
        theme.strip() for theme in args.cluster_themes.split(",") if theme.strip()
    )
    config = BenchmarkConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        max_positions=args.max_positions,
        max_queries=args.max_queries,
        near_hamming=args.near_hamming,
        strategic_positives=args.strategic_positives,
        strategic_negatives=args.strategic_negatives,
        cluster_themes=cluster_themes,
        seed=args.seed,
    )
    path = build_benchmark(config)
    print(f"wrote retrieval benchmark to {path}")
    if args.repo_id is not None:
        upload_benchmark(
            path,
            args.repo_id,
            revision=args.revision,
            commit_message=args.commit_message,
        )
        print(f"uploaded retrieval benchmark to {args.repo_id}:{BENCHMARK_SUBDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
