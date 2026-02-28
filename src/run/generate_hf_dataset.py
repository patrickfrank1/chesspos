import argparse
import sys

from src.dataset.config import (
    DatasetConfig,
    EncoderConfig,
    PreprocessingConfig,
    SamplingFilters,
)
from src.dataset.dataset import ChessPositionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chess position datasets and push to HuggingFace Hub"
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace Hub repository name (e.g., username/chesspos-positions)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=3,
        help="Number of batches to generate (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Number of positions per batch (default: 100000)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Train/test split ratio (default: 0.95)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/raw",
        help="Path to directory containing PGN files (default: ./data/raw)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["token_sequence", "tensor", "bitboard"],
        default="token_sequence",
        help="Position encoding format (default: token_sequence)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Temporal window size for training (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate locally without pushing to HuggingFace Hub",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last batch number on HuggingFace Hub",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of Ray workers for parallel processing (default: 4)",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=4096,
        help="Memory limit per worker in MB (default: 4096)",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=2000,
        help="Minimum ELO for position sampling (default: 2000)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.33,
        help="Position subsampling rate (default: 0.33)",
    )
    parser.add_argument(
        "--create-card",
        action="store_true",
        help="Create and push a dataset card to HuggingFace Hub",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sampling_filters = SamplingFilters(
        min_elo=args.min_elo,
        subsample_rate=args.subsample,
    )

    dataset_config = DatasetConfig(
        repo_name=args.repo,
        batch_size=args.batch_size,
        encoding=args.encoding,
        train_ratio=args.train_ratio,
        data_path=args.data_path,
    )

    preprocessing_config = PreprocessingConfig(
        worker_count=args.workers,
        memory_limit_mb=args.memory,
        sampling_filters=sampling_filters,
    )

    encoder_config = EncoderConfig(
        encoding_format=args.encoding,
        window_size=args.window_size,
    )

    dataset = ChessPositionDataset(
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        encoder_config=encoder_config,
    )

    print(f"Generating {args.batches} batches for {args.repo}")
    print(f"Batch size: {args.batch_size}, Encoding: {args.encoding}")
    print(f"Workers: {args.workers}, Memory: {args.memory}MB")

    if args.dry_run:
        print("DRY RUN: Data will not be pushed to HuggingFace Hub")

    batches_completed = 0
    total_positions = 0

    for train_data, test_data in dataset.generate(
        num_batches=args.batches,
        resume=args.resume,
        dry_run=args.dry_run,
    ):
        batches_completed += 1
        total_positions += len(train_data) + len(test_data)
        print(
            f"Batch {batches_completed}/{args.batches}: "
            f"train={len(train_data)}, test={len(test_data)}"
        )

    if args.create_card and not args.dry_run:
        card = dataset.create_dataset_card()
        dataset.hf_client.push_dataset_card(card)
        print("Dataset card pushed to HuggingFace Hub")

    print(
        f"\nComplete! Generated {total_positions} positions in {batches_completed} batches"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
