from bdd import given, then, when
from src.dataset.config import SourceConfig
from src.dataset.source import ArchivePlanner, WorkItem, completed_archives


class TestArchivePlanner:
    def test_plan_from_local_directory(self, temp_pgn_dir):
        with given("a local directory containing .pgn files"):
            config = SourceConfig(data_path=temp_pgn_dir)
            planner = ArchivePlanner(config)

        with when("the plan is built with nothing completed"):
            items = planner.plan(completed=set())

        with then("one local work item per PGN file is returned, sorted"):
            assert len(items) == 1
            assert items[0].provider == "local"
            assert items[0].path == temp_pgn_dir + "/test.pgn"
            assert items[0].archive_id == "test"
        return

    def test_plan_from_local_directory_ignores_other_files(self, tmp_path):
        with given("a local directory containing pgn, zstd and unrelated files"):
            (tmp_path / "a.pgn").write_bytes(b"")
            (tmp_path / "b.pgn.zst").write_bytes(b"")
            (tmp_path / "notes.txt").write_bytes(b"")
            planner = ArchivePlanner(SourceConfig(data_path=str(tmp_path)))

        with when("the plan is built"):
            items = planner.plan(completed=set())

        with then("only .pgn and .pgn.zst files become work items"):
            assert [item.archive_id for item in items] == ["a", "b"]
            assert items[1].path.endswith("b.pgn.zst")
        return

    def test_plan_from_local_directory_ignores_non_pgn_zst(self, tmp_path):
        with given("a local directory containing a .pgn file and a non-pgn .zst file"):
            (tmp_path / "a.pgn").write_bytes(b"")
            (tmp_path / "lichess_db_puzzle.csv.zst").write_bytes(b"")
            planner = ArchivePlanner(SourceConfig(data_path=str(tmp_path)))

        with when("the plan is built"):
            items = planner.plan(completed=set())

        with then("only the .pgn file becomes a work item"):
            assert [item.archive_id for item in items] == ["a"]
        return

    def test_plan_from_urls(self):
        with given("a source with two lichess archive URLs"):
            urls = [
                "https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2023-02.pgn.zst",
            ]
            planner = ArchivePlanner(SourceConfig(archive_urls=urls))

        with when("the plan is built with nothing completed"):
            items = planner.plan(completed=set())

        with then("one lichess work item per URL is returned with the stem as id"):
            assert [item.archive_id for item in items] == [
                "lichess_db_standard_rated_2023-01",
                "lichess_db_standard_rated_2023-02",
            ]
            assert all(item.provider == "lichess" for item in items)
            assert items[0].url == urls[0]
            assert all(item.path is None for item in items)
        return

    def test_plan_skips_completed_archives(self, temp_pgn_dir):
        with given("a plan where one archive is already complete on the hub"):
            planner = ArchivePlanner(
                SourceConfig(
                    archive_urls=[
                        "https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst",
                        "https://database.lichess.org/standard/lichess_db_standard_rated_2023-02.pgn.zst",
                    ]
                )
            )

        with when("the plan is built with the first archive marked complete"):
            items = planner.plan(completed={"lichess_db_standard_rated_2023-01"})

        with then("only the incomplete archive is planned"):
            assert [item.archive_id for item in items] == [
                "lichess_db_standard_rated_2023-02"
            ]
        return

    def test_completed_archives_from_repo_listing(self):
        with given("a hub repo file listing with one complete and one partial archive"):
            files = [
                "archives/arch-01/_COMPLETE.json",
                "train/arch-01/batch_0000_0000.parquet",
                "train/arch-01/batch_0000_0000.meta.json",
                "train/arch-02/batch_0000_0000.parquet",
                "train/arch-02/batch_0000_0000.meta.json",
            ]

        with when("completed archives are derived from the listing"):
            completed = completed_archives(files)

        with then("only the archive with a completion marker is reported"):
            assert completed == {"arch-01"}
        return


class TestWorkItem:
    def test_archive_id_from_lichess_url(self):
        with given("a lichess archive URL"):
            url = "https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst"

        with when("the work item is constructed"):
            item = WorkItem(
                archive_id="lichess_db_standard_rated_2023-01",
                url=url,
                provider="lichess",
            )

        with then("the archive id equals the URL stem without extension"):
            assert item.archive_id == "lichess_db_standard_rated_2023-01"
        return
