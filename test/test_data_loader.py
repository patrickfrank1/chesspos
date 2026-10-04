from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.dataset.data_loader import TrainingDataGenerator


@pytest.fixture
def mock_dataset():
    mock = MagicMock()
    mock.__iter__ = MagicMock(
        return_value=iter(
            [
                {
                    "window": np.ones((10, 69), dtype=np.int8),
                    "scalars": np.zeros((3,), dtype=np.int8),
                },
                {
                    "window": np.ones((10, 69), dtype=np.int8) * 2,
                    "scalars": np.zeros((3,), dtype=np.int8),
                },
            ]
        )
    )
    return mock


class TestTrainingDataGenerator:
    def test_initialization(self):
        gen = TrainingDataGenerator(repo_name="test/repo")
        assert gen.repo_name == "test/repo"
        assert gen.split == "train"
        assert gen.batch_size == 32

    def test_with_split(self):
        gen = TrainingDataGenerator(repo_name="test/repo")
        test_gen = gen.with_split("test")
        assert test_gen.split == "test"
        assert gen.split == "train"

    def test_with_masking(self):
        gen = TrainingDataGenerator(repo_name="test/repo")
        masked_gen = gen.with_masking(num_tokens=5)
        assert masked_gen.mask_tokens == 5
        assert gen.mask_tokens == 0

    def test_apply_mask(self):
        gen = TrainingDataGenerator(
            repo_name="test/repo", mask_tokens=10, mask_token_id=16
        )
        window = np.zeros((5, 69), dtype=np.int8)

        masked = gen._apply_mask(window.copy())

        assert masked.shape == window.shape
        assert np.sum(masked == 16) > 0

    def test_apply_mask_zero_tokens(self):
        gen = TrainingDataGenerator(repo_name="test/repo", mask_tokens=0)
        window = np.ones((5, 69), dtype=np.int8)

        masked = gen._apply_mask(window.copy())

        np.testing.assert_array_equal(masked, window)

    def test_on_epoch_end(self):
        gen = TrainingDataGenerator(repo_name="test/repo")
        assert gen.epoch_count == 0

        gen.on_epoch_end()
        assert gen.epoch_count == 1

        gen.on_epoch_end()
        assert gen.epoch_count == 2

    def test_with_transformation(self):
        gen = TrainingDataGenerator(repo_name="test/repo")

        def double_fn(x):
            return x * 2

        transformed = gen.with_transformation(double_fn)
        assert transformed.repo_name == gen.repo_name

    @patch("src.dataset.data_loader.load_dataset")
    def test_to_tf_dataset_streaming(self, mock_load_dataset, mock_dataset):
        mock_load_dataset.return_value = mock_dataset

        gen = TrainingDataGenerator(
            repo_name="test/repo",
            batch_size=2,
            shuffle_buffer_size=0,
        )

        ds = gen.to_tf_dataset(streaming=True)

        mock_load_dataset.assert_called_once_with(
            "test/repo",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        assert ds is not None

    @patch("src.dataset.data_loader.load_dataset")
    def test_to_tf_dataset_memory(self, mock_load_dataset):
        mock_ds = [
            {
                "window": np.ones((10, 69), dtype=np.int8),
                "scalars": np.zeros((3,), dtype=np.int8),
            },
            {
                "window": np.ones((10, 69), dtype=np.int8),
                "scalars": np.zeros((3,), dtype=np.int8),
            },
        ]
        mock_load_dataset.return_value = mock_ds

        gen = TrainingDataGenerator(
            repo_name="test/repo",
            batch_size=2,
            shuffle_buffer_size=0,
        )

        ds = gen.to_tf_dataset(streaming=False)

        mock_load_dataset.assert_called_once_with(
            "test/repo",
            split="train",
            trust_remote_code=True,
        )
        assert ds is not None

    def test_epoch_count_property(self):
        gen = TrainingDataGenerator(repo_name="test/repo")
        gen._epoch_count = 5
        assert gen.epoch_count == 5
