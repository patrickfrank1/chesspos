from unittest.mock import MagicMock, patch

from bdd import given, then, when

from src.dataset.archive_pipeline import RunResult


class TestCliOutputDirDefault:
    def test_dry_run_defaults_output_dir_to_data_processed(self):
        with given("the CLI invoked with --archive-urls --dry-run and no --output-dir"):
            argv = [
                "generate_hf_dataset.py",
                "--repo",
                "local/chesspos-dryrun",
                "--archive-urls",
                "https://x/a.pgn.zst",
                "--dry-run",
            ]

        with when("main() runs with the pipeline stubbed out"):
            import src.run.generate_hf_dataset as cli

            stub = MagicMock()
            stub.return_value.run.return_value = RunResult()
            with patch.object(cli, "ArchivePipeline", stub):
                with patch("sys.argv", argv):
                    exit_code = cli.main()

        with then("the pipeline is asked to write to the default data/processed dir"):
            assert exit_code == 0
            kwargs = stub.return_value.run.call_args.kwargs
            assert kwargs["dry_run"] is True
            assert kwargs["output_dir"] == "data/processed"
        return
