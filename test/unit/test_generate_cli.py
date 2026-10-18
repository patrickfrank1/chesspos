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


class TestCliLocalArchivePipeline:
    def test_pipeline_archive_uses_local_archive_pipeline(self):
        with given("the CLI invoked with --pipeline archive and a local data path"):
            argv = [
                "generate_hf_dataset.py",
                "--repo",
                "local/chesspos-test",
                "--data-path",
                "data/raw",
                "--pipeline",
                "archive",
                "--dry-run",
            ]

        with when("main() runs with both pipelines stubbed out"):
            import src.run.generate_hf_dataset as cli

            stub = MagicMock()
            stub.return_value.run.return_value = RunResult()
            with patch.object(cli, "ArchivePipeline", stub):
                with patch.object(cli, "ChessPositionDataset") as legacy_stub:
                    with patch("sys.argv", argv):
                        exit_code = cli.main()

        with then(
            "the archive pipeline is constructed from the local data path "
            "and the legacy dataset is not used"
        ):
            assert exit_code == 0
            source_config = stub.call_args.kwargs["source_config"]
            assert source_config.data_path == "data/raw"
            assert source_config.archive_urls is None
            assert stub.return_value.run.call_args.kwargs["dry_run"] is True
            legacy_stub.assert_not_called()
        return

    def test_pipeline_archive_from_yaml_config(self, tmp_path):
        with given("a YAML config selecting the archive pipeline for a local dir"):
            config = tmp_path / "pipeline.yaml"
            config.write_text(
                "repo_name: local/chesspos-test\n"
                "data_path: data/raw\n"
                "pipeline: archive\n"
                "dry_run: true\n"
            )
            argv = ["generate_hf_dataset.py", "--config", str(config)]

        with when("main() runs with both pipelines stubbed out"):
            import src.run.generate_hf_dataset as cli

            stub = MagicMock()
            stub.return_value.run.return_value = RunResult()
            with patch.object(cli, "ArchivePipeline", stub):
                with patch.object(cli, "ChessPositionDataset") as legacy_stub:
                    with patch("sys.argv", argv):
                        exit_code = cli.main()

        with then("the archive pipeline is selected from the config file"):
            assert exit_code == 0
            source_config = stub.call_args.kwargs["source_config"]
            assert source_config.data_path == "data/raw"
            legacy_stub.assert_not_called()
        return
