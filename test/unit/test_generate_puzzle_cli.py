from unittest.mock import MagicMock, patch

from bdd import given, then, when


class TestGeneratePuzzleCli:
    def test_cli_builds_config_from_arguments_and_runs_the_generator(self):
        with given("the CLI invoked with explicit input, output and shard size"):
            argv = [
                "generate_puzzle_dataset.py",
                "--input",
                "data/raw/lichess_db_puzzle.csv",
                "--output-dir",
                "data/processed/puzzles",
                "--shard-size",
                "50000",
                "--workers",
                "2",
                "--memory",
                "2048",
                "--debug",
            ]

        with when("main() runs with the dataset generator stubbed out"):
            import src.run.generate_puzzle_dataset as cli

            stub = MagicMock()
            with patch.object(cli, "PuzzleDataset", stub):
                with patch("sys.argv", argv):
                    exit_code = cli.main()

        with then("the generator receives a config matching the CLI arguments"):
            assert exit_code == 0
            config = stub.call_args.args[0]
            assert config.input_path == "data/raw/lichess_db_puzzle.csv"
            assert config.output_dir == "data/processed/puzzles"
            assert config.shard_size == 50000
            assert config.worker_count == 2
            assert config.memory_limit_mb == 2048
            assert config.debug is True
            stub.return_value.generate.assert_called_once()
        return

    def test_defaults_point_at_the_lichess_dump_and_processed_dir(self):
        with given("the CLI invoked without any arguments"):
            argv = ["generate_puzzle_dataset.py"]

        with when("main() runs with the dataset generator stubbed out"):
            import src.run.generate_puzzle_dataset as cli

            stub = MagicMock()
            with patch.object(cli, "PuzzleDataset", stub):
                with patch("sys.argv", argv):
                    exit_code = cli.main()

        with then("the default paths and worker settings are used"):
            assert exit_code == 0
            config = stub.call_args.args[0]
            assert config.input_path == "data/raw/lichess_db_puzzle.csv"
            assert config.output_dir == "data/processed/puzzles"
            assert config.shard_size == 100_000
            assert config.worker_count == 4
            assert config.memory_limit_mb == 4096
            assert config.debug is False
        return
