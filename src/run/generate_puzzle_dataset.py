import argparse

from src.dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Lichess puzzle classification dataset as parquet"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/lichess_db_puzzle.csv",
        help="Path to the Lichess puzzle CSV dump",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/puzzles",
        help="Directory for the output parquet shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Target number of puzzles per parquet shard",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of Ray workers",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=4096,
        help="Memory limit per Ray worker in MB",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Ray in local, single-threaded debug mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PuzzleDatasetConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        worker_count=args.workers,
        memory_limit_mb=args.memory,
        debug=args.debug,
    )

    print(f"Reading puzzles from: {config.input_path}")
    print(f"Writing parquet shards to: {config.output_dir}")
    print(
        f"Shard size: {config.shard_size}, workers: {config.worker_count}, "
        f"memory: {config.memory_limit_mb}MB"
    )

    output_dir = PuzzleDataset(config).generate()
    print(f"Done. Dataset written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
