import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from src.dataset.config import (
    DatasetConfig,
    EncoderConfig,
    PreprocessingConfig,
    SamplingFilters,
)
from src.dataset.etl import ChessPositionDataset


def load_yaml_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_nested(data: dict[str, Any], key: str) -> Any:
    flat_fields = {"batch_size", "encoding", "train_ratio", "repo_name", "data_path"}
    encoder_fields = {"window_size"}
    runtime_fields = {"num_batches", "dry_run", "resume", "create_card"}
    preprocessing_fields = {"worker_count", "memory_limit_mb", "debug"}
    sampling_fields = {"min_elo", "min_ply", "max_ply", "subsample_rate"}

    if key in flat_fields:
        return data.get(key)
    if key in runtime_fields:
        return data.get(key)
    if key in encoder_fields:
        encoder = data.get("encoder", {})
        if isinstance(encoder, dict):
            return encoder.get(key)
        return None
    if key in preprocessing_fields:
        pre = data.get("preprocessing", {})
        if isinstance(pre, dict):
            return pre.get(key)
        return None
    if key in sampling_fields:
        sampling = data.get("sampling", {})
        if isinstance(sampling, dict):
            return sampling.get(key)
        return None

    return data.get(key)


def _yaml_val(yaml_cfg: dict[str, Any], key: str) -> Any:
    """Retrieve a value from a YAML config dict using the nested key mapping."""
    yaml_key = {"batches": "num_batches"}.get(key, key)
    return _get_nested(yaml_cfg, yaml_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chess position datasets and push to HuggingFace Hub"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (CLI args override file values)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="HuggingFace Hub repository name (e.g., username/chesspos-positions)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=None,
        help="Number of batches to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of positions per batch",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Train/test split ratio",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to directory containing PGN files",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["token_sequence", "tensor", "bitboard"],
        default=None,
        help="Position encoding format",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Temporal window size for training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Generate locally without pushing to HuggingFace Hub",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Continue from last batch number on HuggingFace Hub",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of Ray workers for parallel processing",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=None,
        help="Memory limit per worker in MB",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=None,
        help="Minimum ELO for position sampling",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=None,
        help="Minimum ply to start sampling",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=None,
        help="Maximum ply for sampling",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Position subsampling rate",
    )
    parser.add_argument(
        "--create-card",
        action="store_true",
        default=None,
        help="Create and push a dataset card to HuggingFace Hub",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Run in debug mode (local, single-threaded, breakpoints work)",
    )

    return parser.parse_args()


def _resolve(
    cli_val: Any,
    yaml_val: Any,
    default: Any,
) -> Any:
    if cli_val is not None:
        return cli_val
    if yaml_val is not None:
        return yaml_val
    return default


def main() -> int:
    args = parse_args()

    yaml_cfg: dict[str, Any] = {}
    if args.config:
        yaml_path = Path(args.config)
        if not yaml_path.exists():
            print(f"Config file not found: {args.config}", file=sys.stderr)
            return 1
        yaml_cfg = load_yaml_config(args.config)

    repo_name: str = _resolve(args.repo, _yaml_val(yaml_cfg, "repo_name"), None)
    if not repo_name:
        print("Error: --repo is required (or set repo_name in YAML config)", file=sys.stderr)
        return 1

    data_path: str = _resolve(args.data_path, _yaml_val(yaml_cfg, "data_path"), "./data/raw")
    batch_size: int = _resolve(args.batch_size, _yaml_val(yaml_cfg, "batch_size"), 100_000)
    encoding: str = _resolve(args.encoding, _yaml_val(yaml_cfg, "encoding"), "token_sequence")
    train_ratio: float = _resolve(args.train_ratio, _yaml_val(yaml_cfg, "train_ratio"), 0.95)
    window_size: int = _resolve(args.window_size, _yaml_val(yaml_cfg, "window_size"), 10)
    num_batches: int = _resolve(args.batches, _yaml_val(yaml_cfg, "batches"), 3)
    worker_count: int = _resolve(args.workers, _yaml_val(yaml_cfg, "worker_count"), 4)
    memory_limit_mb: int = _resolve(args.memory, _yaml_val(yaml_cfg, "memory_limit_mb"), 4096)
    min_elo: int = _resolve(args.min_elo, _yaml_val(yaml_cfg, "min_elo"), 2000)
    min_ply: int = _resolve(args.min_ply, _yaml_val(yaml_cfg, "min_ply"), 0)
    max_ply: int | None = _resolve(args.max_ply, _yaml_val(yaml_cfg, "max_ply"), None)
    subsample_rate: float = _resolve(args.subsample, _yaml_val(yaml_cfg, "subsample_rate"), 0.33)
    dry_run: bool = _resolve(args.dry_run, _yaml_val(yaml_cfg, "dry_run"), False)
    resume: bool = _resolve(args.resume, _yaml_val(yaml_cfg, "resume"), False)
    create_card: bool = _resolve(args.create_card, _yaml_val(yaml_cfg, "create_card"), False)
    debug: bool = _resolve(args.debug, _yaml_val(yaml_cfg, "debug"), False)

    sampling_filters = SamplingFilters(
        min_elo=min_elo,
        min_ply=min_ply,
        max_ply=max_ply,
        subsample_rate=subsample_rate,
    )

    dataset_config = DatasetConfig(
        repo_name=repo_name,
        batch_size=batch_size,
        encoding=encoding,
        train_ratio=train_ratio,
        data_path=data_path,
    )

    preprocessing_config = PreprocessingConfig(
        worker_count=worker_count,
        memory_limit_mb=memory_limit_mb,
        sampling_filters=sampling_filters,
        debug=debug,
    )

    encoder_config = EncoderConfig(
        encoding_format=encoding,
        window_size=window_size,
    )

    dataset = ChessPositionDataset(
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        encoder_config=encoder_config,
    )

    print(f"Generating {num_batches} batches for {repo_name}")
    print(f"Data path: {data_path}")
    print(f"Batch size: {batch_size}, Encoding: {encoding}")
    print(f"Workers: {worker_count}, Memory: {memory_limit_mb}MB")
    print(f"ELO filter: >= {min_elo}, Subsample: {subsample_rate}")

    if dry_run:
        print("DRY RUN: Data will not be pushed to HuggingFace Hub")

    batches_completed = 0
    total_positions = 0

    for train_data, test_data in dataset.generate(
        num_batches=num_batches,
        resume=resume,
        dry_run=dry_run,
    ):
        batches_completed += 1
        total_positions += len(train_data) + len(test_data)
        print(
            f"Batch {batches_completed}/{num_batches}: "
            f"train={len(train_data)}, test={len(test_data)}"
        )

    if create_card and not dry_run:
        card = dataset.create_dataset_card()
        dataset.hf_client.push_dataset_card(card)
        print("Dataset card pushed to HuggingFace Hub")

    print(
        f"\nComplete! Generated {total_positions} positions in {batches_completed} batches"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
