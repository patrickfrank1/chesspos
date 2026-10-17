from bdd import given, then, when
from src.dataset.config import SourceConfig


class TestSourceConfig:
    def test_local_directory_source(self):
        with given("a source pointing to a local directory"):
            config = SourceConfig(data_path="/data/raw")

        with when("the config is validated"):
            config.__post_init__()

        with then("the local source is accepted"):
            assert config.data_path == "/data/raw"
            assert config.archive_urls is None
        return

    def test_archive_urls_source(self):
        with given("a source pointing to one or more lichess archive URLs"):
            config = SourceConfig(
                archive_urls=[
                    "https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst",
                    "https://database.lichess.org/standard/lichess_db_standard_rated_2023-02.pgn.zst",
                ]
            )

        with when("the config is validated"):
            config.__post_init__()

        with then("the url source is accepted"):
            assert len(config.archive_urls) == 2
            assert config.data_path is None
        return

    def test_no_source_rejected(self):
        with given("a source with neither data_path nor archive_urls"):
            error = None

        with when("the config is constructed"):
            try:
                SourceConfig()
            except ValueError as exc:
                error = exc

        with then("validation fails"):
            assert error is not None
        return

    def test_both_sources_rejected(self):
        with given("a source with both data_path and archive_urls"):
            error = None

        with when("the config is constructed"):
            try:
                SourceConfig(
                    data_path="/data/raw",
                    archive_urls=[
                        "https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst"
                    ],
                )
            except ValueError as exc:
                error = exc

        with then("validation fails because sources are mutually exclusive"):
            assert error is not None
        return

    def test_empty_url_rejected(self):
        with given("a source whose archive_urls contains an empty string"):
            error = None

        with when("the config is constructed"):
            try:
                SourceConfig(archive_urls=[""])
            except ValueError as exc:
                error = exc

        with then("validation fails"):
            assert error is not None
        return

    def test_chunk_target_mb_must_be_positive(self):
        with given("a source with a non-positive chunk target"):
            error = None

        with when("the config is constructed"):
            try:
                SourceConfig(data_path="/data/raw", chunk_target_mb=0)
            except ValueError as exc:
                error = exc

        with then("validation fails"):
            assert error is not None
        return

    def test_default_chunk_target_mb(self):
        with given("a source without an explicit chunk target"):
            config = SourceConfig(data_path="/data/raw")

        with when("the config is inspected"):
            target = config.chunk_target_mb

        with then("a positive default chunk target is used"):
            assert target > 0
        return

    def test_download_workers_default(self):
        with given("a url source without an explicit download worker count"):
            config = SourceConfig(
                archive_urls=[
                    "https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst"
                ]
            )

        with when("the download worker count is inspected"):
            workers = config.download_workers

        with then("it defaults to 8 parallel connections"):
            assert workers == 8
        return
