"""CLI: build a LanceDB position vector index from packed game shards."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Sequence

from src.search.schema import IndexManifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LanceDB index of position embeddings"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", default=None)
    source.add_argument("--repo", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default="data/hf-shards")
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-games", type=int, default=32)
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument(
        "--index-type",
        choices=("ivf_pq", "ivf_hnsw_sq", "ivf_flat"),
        default="ivf_pq",
    )
    parser.add_argument("--num-partitions", type=int, default=None)
    parser.add_argument("--num-sub-vectors", type=int, default=None)
    parser.add_argument("--no-index", action="store_true")
    build_mode = parser.add_mutually_exclusive_group()
    build_mode.add_argument("--overwrite", action="store_true")
    build_mode.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing index instead of creating it",
    )
    parser.add_argument(
        "--partition-by",
        choices=("none", "year", "source"),
        default="none",
    )
    parser.add_argument("--source-year", type=int, default=None)
    parser.add_argument("--default-split", default="train")
    parser.add_argument("--table", default="positions")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from src.search.embedder import CheckpointEmbedder, file_sha256
    from src.search.index import build_position_index as build_from_dir
    from src.search.lancedb_store import LancePositionStore
    from src.search.partitioned_store import PartitionedPositionStore

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    embedder = CheckpointEmbedder(args.checkpoint, device=args.device)
    partition_by = None if args.partition_by == "none" else args.partition_by
    if args.repo:
        from src.training.hf_data import download_split

        data_dir = download_split(
            args.repo,
            args.split,
            cache_dir=args.cache_dir,
            max_bytes=args.max_bytes,
            revision=args.revision,
            log=print,
        )
    else:
        data_dir = Path(args.data_dir)
    store_class = PartitionedPositionStore if partition_by else LancePositionStore
    store = store_class(
        output,
        dim=embedder.dim,
        create=not args.append,
        overwrite=args.overwrite,
        table_name=args.table,
    )
    stats = build_from_dir(
        store=store,
        embedder=embedder,
        data_dir=data_dir,
        batch_games=args.batch_games,
        limit_games=args.limit_games,
        source_year=args.source_year,
        default_split=args.split if args.repo else args.default_split,
        partition_by=partition_by,
        progress=print,
    )
    if stats.positions and not args.no_index:
        store.create_index(
            index_type=args.index_type,
            num_partitions=args.num_partitions,
            num_sub_vectors=args.num_sub_vectors,
        )
        store.create_scalar_indexes()
        store.optimize()

    import lancedb

    previous = store.read_manifest() if args.append else None
    total_rows = store.count()
    previous_rows = max(0, total_rows - stats.positions)
    previous_dated = (
        previous.date_coverage * previous_rows if previous is not None else 0.0
    )
    total_games = (
        store.games_count()
        if hasattr(store, "games_count")
        else (previous.games if previous is not None else 0) + stats.games
    )
    partitions = (
        {name: int(info.get("rows", 0)) for name, info in store.partitions.items()}
        if hasattr(store, "partitions")
        else None
    )
    manifest = IndexManifest(
        index_version=1,
        checkpoint=args.checkpoint,
        checkpoint_sha256=file_sha256(args.checkpoint),
        feature="native_positions",
        dim=embedder.dim,
        dtype="float16",
        schema_version=1,
        rows=total_rows,
        games=total_games,
        date_coverage=(
            (previous_dated + stats.dated_positions) / total_rows if total_rows else 0.0
        ),
        lancedb_version=lancedb.__version__,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        partitions=partitions or None,
    )
    manifest_path = store.write_manifest(manifest)
    store.close()
    print(
        json.dumps(
            {
                "manifest": manifest.to_dict(),
                "manifest_path": str(manifest_path),
                "stats": stats.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
