from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bdd import given, then, when
from src.dataset.puzzle_variants import split_for_puzzle
from src.dataset.token_stream import TokenStreamEncoder, pack_stream
from src.evaluation.retrieval import (
    BENCHMARK_SUBDIR,
    DEFAULT_CLUSTER_THEMES,
    DEFAULT_STRATEGIC_NEGATIVES,
    DEFAULT_STRATEGIC_POSITIVES,
    BenchmarkConfig,
    build_benchmark,
    load_benchmark,
    resolve_benchmark,
    upload_benchmark,
)


def val_puzzle_ids(count: int, val_ratio: float = 0.5) -> list[str]:
    ids: list[str] = []
    index = 0
    while len(ids) < count:
        puzzle_id = f"puzzle-{index}"
        if split_for_puzzle(puzzle_id, val_ratio) == "val":
            ids.append(puzzle_id)
        index += 1
    return ids


def train_puzzle_id(val_ratio: float = 0.5) -> str:
    index = 0
    while True:
        puzzle_id = f"puzzle-{index}"
        if split_for_puzzle(puzzle_id, val_ratio) == "train":
            return puzzle_id
        index += 1


def base_board() -> chess.Board:
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(uci)
    return board


def changed_board(board: chess.Board, changes: list[tuple[int, chess.Piece]]):
    changed = board.copy()
    for square, piece in changes:
        changed.set_piece_at(square, piece)
    return changed


ONE_CHANGE = [(chess.F3, chess.Piece(chess.BISHOP, chess.WHITE))]
TWO_CHANGES = ONE_CHANGE + [(chess.A3, chess.Piece(chess.PAWN, chess.WHITE))]
THREE_CHANGES = TWO_CHANGES + [(chess.C6, chess.Piece(chess.BISHOP, chess.BLACK))]


def write_puzzle_shard(
    path: Path,
    entries: list[dict],
    theme_names: tuple[str, ...] = (),
) -> None:
    encoder = TokenStreamEncoder()
    columns: dict[str, list] = {
        "puzzle_id": [],
        "packed": [],
        "n_positions": [],
        "rating": [],
    }
    for theme in theme_names:
        columns[theme] = []
    for entry in entries:
        segments = [encoder.encode(board) for board in entry["boards"]]
        columns["puzzle_id"].append(entry["puzzle_id"])
        columns["packed"].append(pack_stream(segments).tolist())
        columns["n_positions"].append(len(segments))
        columns["rating"].append(1500)
        tagged = set(entry.get("themes", ()))
        for theme in theme_names:
            columns[theme].append(1 if theme in tagged else 0)

    table = pa.table(
        {
            "puzzle_id": pa.array(columns["puzzle_id"], type=pa.string()),
            "packed": pa.array(columns["packed"], type=pa.list_(pa.int16())),
            "n_positions": pa.array(columns["n_positions"], type=pa.int32()),
            "rating": pa.array(columns["rating"], type=pa.int32()),
            **{
                theme: pa.array(columns[theme], type=pa.int8()) for theme in theme_names
            },
        }
    )
    pq.write_table(table, path)
    return


def make_config(
    data_dir: Path,
    output_dir: Path,
    *,
    cluster_themes: tuple[str, ...] = ("rookEndgame", "pawnEndgame"),
    max_positions: int = 100,
    max_queries: int = 20,
    near_hamming: int = 2,
    strategic_positives: int = DEFAULT_STRATEGIC_POSITIVES,
    strategic_negatives: int = DEFAULT_STRATEGIC_NEGATIVES,
    seed: int = 7,
    val_ratio: float = 0.5,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        val_ratio=val_ratio,
        max_positions=max_positions,
        max_queries=max_queries,
        near_hamming=near_hamming,
        strategic_positives=strategic_positives,
        strategic_negatives=strategic_negatives,
        cluster_themes=cluster_themes,
        seed=seed,
    )


def index_of(benchmark, puzzle_id: str, ply: int = 0) -> int:
    for position in benchmark.positions:
        if position.puzzle_id == puzzle_id and position.ply == ply:
            return position.position_id
    raise AssertionError(f"position {puzzle_id}@{ply} not in benchmark")


def near_grade(benchmark, first: int, second: int) -> int:
    if first in benchmark.near_relevance and second in benchmark.near_relevance[first]:
        return benchmark.near_relevance[first][second]
    if second in benchmark.near_relevance and first in benchmark.near_relevance[second]:
        return benchmark.near_relevance[second][first]
    raise AssertionError(f"positions {first} and {second} are not a near pair")


class TestBenchmarkPositions:
    def test_only_validation_puzzles_are_included(self, tmp_path):
        with given("a shard with validation and training puzzles"):
            val_ids = val_puzzle_ids(2)
            train_id = train_puzzle_id()
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": pid, "boards": [base_board()]}
                    for pid in [*val_ids, train_id]
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("only validation puzzles and their positions are kept"):
            assert {position.puzzle_id for position in benchmark.positions} == set(
                val_ids
            )
            assert all(len(position.tokens) > 0 for position in benchmark.positions)
        return

    def test_only_configured_theme_columns_are_recorded(self, tmp_path):
        with given("a puzzle tagged with a configured and an ignored theme"):
            puzzle_id = val_puzzle_ids(1)[0]
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {
                        "puzzle_id": puzzle_id,
                        "boards": [base_board()],
                        "themes": ("rookEndgame", "sacrifice"),
                    }
                ],
                theme_names=("rookEndgame", "sacrifice"),
            )

        with when("the benchmark records cluster themes"):
            benchmark = load_benchmark(
                build_benchmark(
                    make_config(
                        tmp_path,
                        tmp_path / "bench",
                        cluster_themes=("rookEndgame",),
                    )
                )
            )

        with then("only the configured theme is stored"):
            assert benchmark.positions[0].themes == ("rookEndgame",)
        return


class TestClusterThemes:
    def test_endgame_is_not_a_default_cluster_theme(self):
        with given("the default strategic cluster themes"):
            themes = DEFAULT_CLUSTER_THEMES

        with when("checking for broad phase and specific endgame labels"):
            has_endgame = "endgame" in themes
            has_specific_endgames = {
                "rookEndgame",
                "pawnEndgame",
                "queenRookEndgame",
                "bishopEndgame",
                "knightEndgame",
            }.issubset(themes)

        with then("the broad endgame phase is excluded but types remain"):
            assert not has_endgame
            assert has_specific_endgames
            assert "promotion" in themes
            assert "zugzwang" in themes
        return


class TestReciprocalNearPairs:
    def test_reciprocal_near_pair_is_recorded_once(self, tmp_path):
        with given("two cross-puzzle identical positions"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("the pair is stored and queried in one direction only"):
            source = index_of(benchmark, first)
            target = index_of(benchmark, second)
            assert len(benchmark.near_queries) == 1
            assert sum(len(rows) for rows in benchmark.near_relevance.values()) == 1
            assert near_grade(benchmark, source, target) == 0
        return

    def test_disjoint_pairs_stay_independent(self, tmp_path):
        with given("two unrelated pairs of cross-puzzle duplicates"):
            first, second, third, fourth = val_puzzle_ids(4)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                    {
                        "puzzle_id": third,
                        "boards": [changed_board(base_board(), THREE_CHANGES)],
                    },
                    {
                        "puzzle_id": fourth,
                        "boards": [changed_board(base_board(), THREE_CHANGES)],
                    },
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("each pair becomes exactly one query"):
            assert len(benchmark.near_queries) == 2
            assert sum(len(rows) for rows in benchmark.near_relevance.values()) == 2
            assert (
                near_grade(
                    benchmark, index_of(benchmark, first), index_of(benchmark, second)
                )
                == 0
            )
            assert (
                near_grade(
                    benchmark, index_of(benchmark, third), index_of(benchmark, fourth)
                )
                == 0
            )
        return


class TestNearDuplicateRelevance:
    def test_exact_cross_puzzle_duplicate_gets_grade_zero(self, tmp_path):
        with given("the same board in two different validation puzzles"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("the cross-puzzle pair is an exact positive"):
            source = index_of(benchmark, first)
            target = index_of(benchmark, second)
            assert near_grade(benchmark, source, target) == 0
        return

    def test_same_puzzle_positions_are_never_related(self, tmp_path):
        with given("a duplicated board inside one puzzle and once in another"):
            first, second = val_puzzle_ids(2)
            board = base_board()
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [board, board]},
                    {"puzzle_id": second, "boards": [board]},
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("no relevance pair links two positions of the same puzzle"):
            for query, rows in benchmark.near_relevance.items():
                for candidate in rows:
                    assert (
                        benchmark.positions[query].puzzle_id
                        != benchmark.positions[candidate].puzzle_id
                    )
            first_zero = index_of(benchmark, first, ply=0)
            second_zero = index_of(benchmark, second)
            assert near_grade(benchmark, first_zero, second_zero) == 0
        return

    def test_one_square_difference_is_graded_one(self, tmp_path):
        with given("two cross-puzzle boards differing on one square"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {
                        "puzzle_id": second,
                        "boards": [changed_board(base_board(), ONE_CHANGE)],
                    },
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("the pair is graded as a one-square near duplicate"):
            source = index_of(benchmark, first)
            target = index_of(benchmark, second)
            assert near_grade(benchmark, source, target) == 1
        return

    def test_two_square_difference_is_graded_two(self, tmp_path):
        with given("two cross-puzzle boards differing on two squares"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {
                        "puzzle_id": second,
                        "boards": [changed_board(base_board(), TWO_CHANGES)],
                    },
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("the pair is graded as a two-square near duplicate"):
            source = index_of(benchmark, first)
            target = index_of(benchmark, second)
            assert near_grade(benchmark, source, target) == 2
        return

    def test_three_square_difference_is_not_relevant(self, tmp_path):
        with given("two cross-puzzle boards differing on three squares"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {
                        "puzzle_id": second,
                        "boards": [changed_board(base_board(), THREE_CHANGES)],
                    },
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("no relevance pair is recorded and no query is selected"):
            assert len(benchmark.near_queries) == 0
        return

    def test_same_placement_with_other_turn_is_graded_one(self, tmp_path):
        with given("two cross-puzzle boards with identical placement and other turn"):
            first, second = val_puzzle_ids(2)
            flipped = base_board()
            flipped.turn = not flipped.turn
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [flipped]},
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("the pair is near but not an exact match"):
            source = index_of(benchmark, first)
            target = index_of(benchmark, second)
            assert near_grade(benchmark, source, target) == 1
        return

    def test_unique_positions_do_not_become_queries(self, tmp_path):
        with given("two cross-puzzle boards that are not related"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {
                        "puzzle_id": second,
                        "boards": [changed_board(base_board(), THREE_CHANGES)],
                    },
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("the query set is empty"):
            assert len(benchmark.near_queries) == 0
            assert benchmark.near_relevance == {}
        return


class TestBenchmarkSampling:
    def test_same_seed_reproduces_the_same_benchmark(self, tmp_path):
        with given("eight identical cross-puzzle boards"):
            puzzle_ids = val_puzzle_ids(8)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": puzzle_id, "boards": [base_board()]}
                    for puzzle_id in puzzle_ids
                ],
            )

        with when("the benchmark is built twice with the same seed"):
            first = load_benchmark(
                build_benchmark(
                    make_config(
                        tmp_path,
                        tmp_path / "bench-a",
                        max_positions=4,
                        max_queries=3,
                        seed=11,
                    )
                )
            )
            second = load_benchmark(
                build_benchmark(
                    make_config(
                        tmp_path,
                        tmp_path / "bench-b",
                        max_positions=4,
                        max_queries=3,
                        seed=11,
                    )
                )
            )

        with then("positions, queries and relevance agree"):
            assert [p.puzzle_id for p in first.positions] == [
                p.puzzle_id for p in second.positions
            ]
            np.testing.assert_array_equal(first.near_queries, second.near_queries)
            assert first.near_relevance == second.near_relevance
        return

    def test_respects_position_and_query_limits(self, tmp_path):
        with given("eight identical cross-puzzle boards"):
            puzzle_ids = val_puzzle_ids(8)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": puzzle_id, "boards": [base_board()]}
                    for puzzle_id in puzzle_ids
                ],
            )

        with when("the benchmark is limited to four positions and two queries"):
            benchmark = load_benchmark(
                build_benchmark(
                    make_config(
                        tmp_path,
                        tmp_path / "bench",
                        max_positions=4,
                        max_queries=2,
                    )
                )
            )

        with then("the limits are respected"):
            assert len(benchmark.positions) == 4
            assert len(benchmark.near_queries) <= 2
        return

    def test_manifest_records_the_configuration_and_counts(self, tmp_path):
        with given("two exact cross-puzzle duplicates"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                ],
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(
                    make_config(
                        tmp_path,
                        tmp_path / "bench",
                        near_hamming=2,
                        seed=5,
                    )
                )
            )

        with then("the manifest describes how it was built"):
            manifest = benchmark.manifest
            assert manifest["seed"] == 5
            assert manifest["near_hamming"] == 2
            assert manifest["strategic_positives"] == DEFAULT_STRATEGIC_POSITIVES
            assert manifest["strategic_negatives"] == DEFAULT_STRATEGIC_NEGATIVES
            assert manifest["n_positions"] == len(benchmark.positions)
            assert manifest["n_near_queries"] == len(benchmark.near_queries)
            assert (tmp_path / "bench" / "positions.parquet").exists()
        return


class TestBenchmarkArtifacts:
    def test_roundtrip_preserves_positions_and_relevance(self, tmp_path):
        with given("a built benchmark"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                ],
            )
            built = build_benchmark(make_config(tmp_path, tmp_path / "bench"))

        with when("the benchmark is loaded again"):
            benchmark = load_benchmark(built)

        with then("the stored positions and relevance survive the roundtrip"):
            assert len(benchmark.positions) == 2
            assert all(len(position.tokens) == 67 for position in benchmark.positions)
            assert benchmark.near_relevance
        return

    def test_strategic_queries_require_a_cross_puzzle_same_theme_match(self, tmp_path):
        with given("two rook endgames in different puzzles and a pawn endgame"):
            first, second, third = val_puzzle_ids(3)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {
                        "puzzle_id": first,
                        "boards": [base_board()],
                        "themes": ("rookEndgame",),
                    },
                    {
                        "puzzle_id": second,
                        "boards": [base_board()],
                        "themes": ("rookEndgame",),
                    },
                    {
                        "puzzle_id": third,
                        "boards": [base_board()],
                        "themes": ("pawnEndgame",),
                    },
                ],
                theme_names=("rookEndgame", "pawnEndgame"),
            )

        with when("the benchmark is built"):
            benchmark = load_benchmark(
                build_benchmark(make_config(tmp_path, tmp_path / "bench"))
            )

        with then("only themes with cross-puzzle matches become queries"):
            assert "rookEndgame" in benchmark.strategic_queries
            assert "pawnEndgame" not in benchmark.strategic_queries
            rook_ids = benchmark.strategic_queries["rookEndgame"]
            puzzles = {benchmark.positions[index].puzzle_id for index in rook_ids}
            assert len(puzzles) == 2
        return


class TestBuildBenchmarkCli:
    def test_build_cli_writes_a_loadable_benchmark(self, tmp_path):
        with given("a shard with two cross-puzzle duplicates"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                ],
            )
            output_dir = tmp_path / "bench"
            argv = [
                "build_retrieval_benchmark.py",
                "--data-dir",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "--val-ratio",
                "0.5",
                "--max-positions",
                "50",
                "--max-queries",
                "5",
                "--near-hamming",
                "2",
                "--strategic-positives",
                "4",
                "--strategic-negatives",
                "6",
                "--cluster-themes",
                "rookEndgame,pawnEndgame",
                "--seed",
                "3",
            ]

        with when("the build entry point runs"):
            import src.run.build_retrieval_benchmark as cli

            with patch("sys.argv", argv):
                exit_code = cli.main()

        with then("the artifacts load back"):
            assert exit_code == 0
            assert (output_dir / "positions.parquet").exists()
            assert (output_dir / "manifest.json").exists()
            benchmark = load_benchmark(output_dir)
            assert benchmark.near_queries.size > 0
            assert benchmark.manifest["strategic_positives"] == 4
            assert benchmark.manifest["strategic_negatives"] == 6
        return


class TestHuggingFaceBenchmark:
    def test_local_directory_is_used_directly(self, tmp_path):
        with given("a benchmark built on the local filesystem"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                ],
            )
            bench_dir = build_benchmark(
                make_config(tmp_path, tmp_path / "bench", max_positions=10)
            )

        with when("the benchmark source is resolved"):
            resolved = resolve_benchmark(bench_dir)

        with then("the local directory is returned unchanged"):
            assert resolved == bench_dir
        return

    def test_hub_repo_downloads_only_the_benchmark_subfolder(
        self, tmp_path, monkeypatch
    ):
        with given("a built benchmark and a stubbed Hub snapshot"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                ],
            )
            bench_dir = build_benchmark(
                make_config(tmp_path, tmp_path / "bench", max_positions=10)
            )
            snapshot = tmp_path / "snapshot"
            target = snapshot / BENCHMARK_SUBDIR
            target.mkdir(parents=True)
            for item in bench_dir.iterdir():
                (target / item.name).write_bytes(item.read_bytes())

            calls: dict = {}

            def fake_snapshot_download(**kwargs):
                calls.update(kwargs)
                return str(snapshot)

            monkeypatch.setattr(
                "src.evaluation.retrieval.snapshot_download",
                fake_snapshot_download,
            )

        with when("the benchmark is loaded by repository id"):
            benchmark = load_benchmark("org/chesspos", revision="v1")

        with then("only the benchmark subfolder is pulled from the dataset repo"):
            assert calls["repo_id"] == "org/chesspos"
            assert calls["repo_type"] == "dataset"
            assert calls["revision"] == "v1"
            assert calls["allow_patterns"] == [f"{BENCHMARK_SUBDIR}/**"]
            assert len(benchmark.positions) == 2
        return

    def test_download_without_a_manifest_is_rejected(self, tmp_path, monkeypatch):
        with given("a snapshot missing the benchmark manifest"):
            snapshot = tmp_path / "snapshot"
            (snapshot / BENCHMARK_SUBDIR).mkdir(parents=True)
            monkeypatch.setattr(
                "src.evaluation.retrieval.snapshot_download",
                lambda **_: str(snapshot),
            )

        with when("the benchmark source is resolved"):
            error = None
            try:
                resolve_benchmark("org/chesspos")
            except ValueError as exc:
                error = exc

        with then("a clear error is raised"):
            assert error is not None
        return

    def test_upload_uses_the_dataset_repo_and_benchmark_subfolder(
        self, tmp_path, monkeypatch
    ):
        with given("a built benchmark and a stubbed Hub API"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                ],
            )
            bench_dir = build_benchmark(
                make_config(tmp_path, tmp_path / "bench", max_positions=10)
            )
            calls: dict = {}

            class DummyApi:
                def upload_folder(self, **kwargs):
                    calls.update(kwargs)
                    return "commit"

            monkeypatch.setattr("src.evaluation.retrieval.HfApi", lambda: DummyApi())

        with when("the benchmark is uploaded"):
            upload_benchmark(
                bench_dir,
                "org/chesspos",
                revision="v1",
                commit_message="retrieval benchmark v1",
            )

        with then("the artifact goes into the dataset benchmark folder"):
            assert calls["repo_id"] == "org/chesspos"
            assert calls["repo_type"] == "dataset"
            assert calls["path_in_repo"] == BENCHMARK_SUBDIR
            assert calls["revision"] == "v1"
            assert calls["commit_message"] == "retrieval benchmark v1"
        return


class TestEvaluateRetrievalCli:
    def test_cli_prints_metrics_json(self, tmp_path, capsys):
        from dataclasses import asdict

        import torch
        from src.modeling.transformer import MaskedStreamTransformer, TransformerConfig

        with given("a built benchmark and a tiny encoder checkpoint"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                ],
            )
            bench_dir = build_benchmark(
                make_config(tmp_path, tmp_path / "bench", max_positions=10)
            )
            config = TransformerConfig(
                vocab_size=52,
                d_model=1,
                n_heads=1,
                n_layers=1,
                d_ff=2,
                dropout=0.0,
                max_seq_len=256,
                max_segments=8,
            )
            encoder = MaskedStreamTransformer(config)
            checkpoint = tmp_path / "encoder.pt"
            torch.save(
                {
                    "model_config": asdict(config),
                    "model_state": encoder.state_dict(),
                },
                checkpoint,
            )
            argv = [
                "evaluate_retrieval.py",
                "--benchmark",
                str(bench_dir),
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cpu",
                "--batch-size",
                "2",
            ]

        with when("the evaluation entry point runs"):
            import src.run.evaluate_retrieval as cli

            with patch("sys.argv", argv):
                exit_code = cli.main()
            output = capsys.readouterr().out

        with then("a JSON metric object is printed"):
            assert exit_code == 0
            assert "near_recall@1" in output
            assert "rank_effective" in output
        return


class TestBuildBenchmarkUpload:
    def test_build_cli_uploads_when_a_repo_id_is_given(
        self, tmp_path, monkeypatch, capsys
    ):
        with given("a shard, a benchmark output and a stubbed upload"):
            first, second = val_puzzle_ids(2)
            write_puzzle_shard(
                tmp_path / "part.parquet",
                [
                    {"puzzle_id": first, "boards": [base_board()]},
                    {"puzzle_id": second, "boards": [base_board()]},
                ],
            )
            calls: dict = {}

            def fake_upload(path, repo_id, **kwargs):
                calls["path"] = path
                calls["repo_id"] = repo_id
                calls.update(kwargs)
                return "url"

            monkeypatch.setattr(
                "src.run.build_retrieval_benchmark.upload_benchmark",
                fake_upload,
            )
            argv = [
                "build_retrieval_benchmark.py",
                "--data-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "bench"),
                "--val-ratio",
                "0.5",
                "--max-positions",
                "10",
                "--repo-id",
                "org/chesspos",
                "--revision",
                "v1",
                "--commit-message",
                "benchmark v1",
            ]

        with when("the build entry point runs"):
            import src.run.build_retrieval_benchmark as cli

            with patch("sys.argv", argv):
                exit_code = cli.main()
            output = capsys.readouterr().out

        with then("the built artifacts are uploaded to the repo"):
            assert exit_code == 0
            assert calls["repo_id"] == "org/chesspos"
            assert calls["revision"] == "v1"
            assert calls["commit_message"] == "benchmark v1"
            assert "uploaded" in output
        return
