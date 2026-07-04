from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
        client = HuggingFaceClient(repo_name="test/repo")
        mock_hf_api.whoami.assert_called_once()

    def test_init_raises_on_auth_failure(self):
        with patch("src.dataset.huggingface_client.HfApi") as mock_api:
            mock_instance = MagicMock()
            mock_instance.whoami.side_effect = Exception("Not authenticated")
            mock_api.return_value = mock_instance

            with pytest.raises(RuntimeError, match="Not authenticated"):
                HuggingFaceClient(repo_name="test/repo")

    def test_get_existing_batches(self, mock_hf_api):
        mock_hf_api.list_repo_files.return_value = [
            "train/batch_0001.parquet",
            "train/batch_0002.parquet",
            "test/batch_0001.parquet",
            "README.md",
        ]

        client = HuggingFaceClient(repo_name="test/repo")
        batches = client.get_existing_batches(split="train")

        assert batches == [1, 2]

    def test_get_existing_batches_empty(self, mock_hf_api):
        mock_hf_api.list_repo_files.return_value = ["README.md"]

        client = HuggingFaceClient(repo_name="test/repo")
        batches = client.get_existing_batches(split="train")

        assert batches == []

    def test_get_next_batch_number(self, mock_hf_api):
        mock_hf_api.list_repo_files.return_value = [
            "train/batch_0001.parquet",
            "train/batch_0002.parquet",
        ]

        client = HuggingFaceClient(repo_name="test/repo")
        next_batch = client.get_next_batch_number(split="train")

        assert next_batch == 3

    def test_get_next_batch_number_empty_repo(self, mock_hf_api):
        mock_hf_api.list_repo_files.return_value = []

        client = HuggingFaceClient(repo_name="test/repo")
        next_batch = client.get_next_batch_number(split="train")

        assert next_batch == 1

    def test_create_version_tag(self, mock_hf_api):
        client = HuggingFaceClient(repo_name="test/repo")
        url = client.create_version_tag("v1.0.0", tag_message="Release v1")

        mock_hf_api.create_tag.assert_called_once_with(
            repo_id="test/repo",
            tag="v1.0.0",
            repo_type="dataset",
            tag_message="Release v1",
            exist_ok=True,
        )
        assert "v1.0.0" in url

    def test_create_dataset_card(self, mock_hf_api):
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

        card = client.create_dataset_card(
            description="Test dataset",
            features=features,
            usage_example="dataset = load_dataset('test/repo')",
        )

        assert "Test dataset" in card
        assert "window" in card
        assert "(10, 69)" in card

    def test_push_batch(self, mock_hf_api):
        with patch("src.dataset.huggingface_client.Dataset") as mock_dataset:
            mock_ds_instance = MagicMock()
            mock_ds_instance.push_to_hub.return_value = (
                "https://huggingface.co/datasets/test/repo"
            )
            mock_dataset.from_dict.return_value = mock_ds_instance

            client = HuggingFaceClient(repo_name="test/repo")
            data = {
                "window": np.zeros((100, 10, 69), dtype=np.int8),
                "scalars": np.zeros((100, 3), dtype=np.int8),
            }

            url = client.push_batch(data, split="train", batch_number=1)

            mock_dataset.from_dict.assert_called_once()
            mock_ds_instance.push_to_hub.assert_called_once()
            assert url == "https://huggingface.co/datasets/test/repo"
