import os
from unittest.mock import MagicMock, patch

import pytest
import ray
import ray.data

from bdd import given, then, when
from etl_support import make_batch_with_split


@pytest.fixture(scope="module")
def local_ray():
    """Run all tests in this module against a single local Ray instance.

    Forces a fresh local cluster instead of autodetecting a running one
    (RAY_ADDRESS or /tmp/ray leftovers), skips the dashboard, and keeps the
    session alive across tests by no-oping ``ray.shutdown`` while the tests
    run, so startup cost is only paid once.
    """
    with patch.dict(os.environ, {"RAY_ADDRESS": "local"}):
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        try:
            with patch.object(ray, "shutdown"):
                yield
        finally:
            ray.shutdown()
    return


class TestGenerate:
    def test_generate_dry_run(self, dataset, local_ray):
        with given("a dataset backed by one mocked encoded batch"):
            chunk = make_batch_with_split(["deadbeefdeadbeef"])
            mock_encoded = MagicMock(spec=ray.data.Dataset)
            mock_encoded.iter_batches.return_value = iter([chunk])
            dataset._build_encoded = MagicMock(return_value=mock_encoded)

        with when("one batch is generated in dry-run mode"):
            batches = list(dataset.generate(num_batches=1, dry_run=True))

        with then("exactly one batch is yielded with all games split across splits"):
            assert len(batches) == 1
            train_ds, test_ds = batches[0]
            assert train_ds.count() + test_ds.count() == 1
        return

    def test_generate_resumes_from_start_batch(self, dataset, local_ray):
        with given("a dataset backed by two mocked batches and a resume point of 2"):
            chunks = [
                make_batch_with_split([f"aaaaaaaa{i:04d}" for i in range(10)]),
                make_batch_with_split([f"bbbbbbbb{i:04d}" for i in range(10)]),
            ]
            mock_encoded = MagicMock(spec=ray.data.Dataset)
            mock_encoded.iter_batches.return_value = iter(chunks)
            dataset._build_encoded = MagicMock(return_value=mock_encoded)

            mock_client = MagicMock()
            mock_client.get_next_batch_number.return_value = 2
            dataset.hf_client = mock_client

        with when("generation is resumed inside the yield loop"):
            counts = None
            game_ids = None
            num_yielded = 0
            # Consume inside the loop, while the generator is paused at `yield`
            # and the Ray session is still alive.
            for train_ds, test_ds in dataset.generate(
                num_batches=1, dry_run=True, resume=True
            ):
                num_yielded += 1
                counts = (train_ds.count(), test_ds.count())
                game_ids = [row["game_id"] for row in train_ds.take_all()]

        with then("only the second batch is yielded, entirely in the train split"):
            assert num_yielded == 1
            assert counts == (10, 0)
            assert game_ids == [f"bbbbbbbb{i:04d}" for i in range(10)]
        return
