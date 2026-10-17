from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bdd import given, then, when
from src.dataset.huggingface_client import HuggingFaceClient


@pytest.fixture
def mock_hf_api():
    with patch("src.dataset.huggingface_client.HfApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.whoami.return_value = {"name": "test_user"}
        mock_api.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_repo_exists():
    with patch("src.dataset.huggingface_client.repo_exists", return_value=True):
        yield


@pytest.fixture
def mock_create_repo():
    with patch("src.dataset.huggingface_client.create_repo") as mock:
        yield mock


class TestHuggingFaceClient:
    def test_init_checks_authentication(self, mock_hf_api):
        with when("a client is constructed"):
            HuggingFaceClient(repo_name="test/repo")

        with then("authentication is checked with the HF API"):
            mock_hf_api.whoami.assert_called_once()

    def test_init_raises_on_auth_failure(self):
        with given("an HF API that fails authentication"):
            with patch("src.dataset.huggingface_client.HfApi") as mock_api:
                mock_instance = MagicMock()
                mock_instance.whoami.side_effect = Exception("Not authenticated")
                mock_api.return_value = mock_instance

                with when("a client is constructed"):
                    client = None
                    with pytest.raises(RuntimeError) as excinfo:
                        client = HuggingFaceClient(repo_name="test/repo")

        with then("a RuntimeError is raised and no client is created"):
            assert client is None
            assert "Not authenticated" in str(excinfo.value)

    def test_get_existing_batches(self, mock_hf_api):
        with given("a repo containing two train batches and unrelated files"):
            mock_hf_api.list_repo_files.return_value = [
                "train/batch_0001.parquet",
                "train/batch_0002.parquet",
                "test/batch_0001.parquet",
                "README.md",
            ]
            client = HuggingFaceClient(repo_name="test/repo")

        with when("the existing train batches are listed"):
            batches = client.get_existing_batches(split="train")

        with then("only the train batch numbers are returned"):
            assert batches == [1, 2]

    def test_get_existing_batches_empty(self, mock_hf_api):
        with given("a repo containing no batch files"):
            mock_hf_api.list_repo_files.return_value = ["README.md"]
            client = HuggingFaceClient(repo_name="test/repo")

        with when("the existing train batches are listed"):
            batches = client.get_existing_batches(split="train")

        with then("an empty list is returned"):
            assert batches == []

    def test_get_next_batch_number(self, mock_hf_api):
        with given("a repo containing train batches 1 and 2"):
            mock_hf_api.list_repo_files.return_value = [
                "train/batch_0001.parquet",
                "train/batch_0002.parquet",
            ]
            client = HuggingFaceClient(repo_name="test/repo")

        with when("the next batch number is queried"):
            next_batch = client.get_next_batch_number(split="train")

        with then("the number after the highest existing batch is returned"):
            assert next_batch == 3

    def test_get_next_batch_number_empty_repo(self, mock_hf_api):
        with given("an empty repo"):
            mock_hf_api.list_repo_files.return_value = []
            client = HuggingFaceClient(repo_name="test/repo")

        with when("the next batch number is queried"):
            next_batch = client.get_next_batch_number(split="train")

        with then("the first batch number is returned"):
            assert next_batch == 1

    def test_create_version_tag(self, mock_hf_api):
        with given("a client backed by the mocked HF API"):
            client = HuggingFaceClient(repo_name="test/repo")

        with when("a version tag is created"):
            url = client.create_version_tag("v1.0.0", tag_message="Release v1")

        with then("the tag is created on the dataset repo and its url returned"):
            mock_hf_api.create_tag.assert_called_once_with(
                repo_id="test/repo",
                tag="v1.0.0",
                repo_type="dataset",
                tag_message="Release v1",
                exist_ok=True,
            )
            assert "v1.0.0" in url

    def test_create_dataset_card(self, mock_hf_api):
        with given("a client and feature descriptions to document"):
            client = HuggingFaceClient(repo_name="test/repo")
            features = {
                "window": {
                    "shape": "(10, 69)",
                    "dtype": "int8",
                    "description": "Position window",
                },
                "scalars": {
                    "shape": "(3,)",
                    "dtype": "int8",
                    "description": "Scalar features",
                },
            }

        with when("the dataset card is created"):
            card = client.create_dataset_card(
                description="Test dataset",
                features=features,
                usage_example="dataset = load_dataset('test/repo')",
            )

        with then("the card contains the description and feature details"):
            assert "Test dataset" in card
            assert "window" in card
            assert "(10, 69)" in card

    def test_push_batch(self, mock_hf_api):
        with given("a client and a mocked HF Dataset factory"):
            with patch("src.dataset.huggingface_client.Dataset") as mock_dataset:
                mock_ds_instance = MagicMock()
                mock_ds_instance.push_to_hub.return_value = (
                    "https://huggingface.co/datasets/test/repo"
                )
                mock_dataset.from_dict.return_value = mock_ds_instance

                client = HuggingFaceClient(repo_name="test/repo")

                with when("a batch of window and scalar data is pushed"):
                    data = {
                        "window": np.zeros((100, 10, 69), dtype=np.int8),
                        "scalars": np.zeros((100, 3), dtype=np.int8),
                    }
                    url = client.push_batch(data, split="train", batch_number=1)

        with then("the data is uploaded once and the hub url is returned"):
            mock_dataset.from_dict.assert_called_once()
            mock_ds_instance.push_to_hub.assert_called_once()
            assert url == "https://huggingface.co/datasets/test/repo"
