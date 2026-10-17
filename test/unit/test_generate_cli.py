from unittest.mock import MagicMock, patch

from src.dataset.archive_pipeline import RunResult


class TestCliOutputDirDefault:
    def test_dry_run_defaults_output_dir_to_data_processed(self):
        # Given the CLI invoked with --archive-urls --dry-run and no --output-dir
        argv = [
            "generate_hf_dataset.py",
            "--repo",
            "local/chesspos-dryrun",
            "--archive-urls",
            "https://x/a.pgn.zst",
            "--dry-run",
        ]

        # When main() runs (with the pipeline stubbed out)
        with patch("sys.argv", argv):
            import src.run.generate_hf_dataset as cli

            stub = MagicMock()
            stub.return_value.run.return_value = RunResult()
            with patch.object(cli, "ArchivePipeline", stub):
                exit_code = cli.main()

        # Then the pipeline is asked to write to the default data/processed dir
        assert exit_code == 0
        kwargs = stub.return_value.run.call_args.kwargs
        assert kwargs["dry_run"] is True
        assert kwargs["output_dir"] == "data/processed"
