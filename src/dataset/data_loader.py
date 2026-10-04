from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import tensorflow as tf
from datasets import load_dataset

from src.dataset.types import EncodingFormat, TOKEN_SEQUENCE


@dataclass
class TrainingDataGenerator:
    repo_name: str
    split: str = "train"
    encoding: EncodingFormat = TOKEN_SEQUENCE
    shuffle_buffer_size: int = 10_000
    batch_size: int = 32
    mask_tokens: int = 0
    mask_token_id: int = 16
    _dataset: tf.data.Dataset | None = field(default=None, repr=False)
    _epoch_count: int = field(default=0, repr=False)

    def to_tf_dataset(self, streaming: bool = True) -> tf.data.Dataset:
        if streaming:
            ds = load_dataset(
                self.repo_name,
                split=self.split,
                streaming=True,
                trust_remote_code=True,
            )
            self._dataset = self._streaming_to_tf_dataset(ds)
        else:
            ds = load_dataset(
                self.repo_name,
                split=self.split,
                trust_remote_code=True,
            )
            self._dataset = self._memory_to_tf_dataset(ds)

        return self._dataset

    def _streaming_to_tf_dataset(self, dataset) -> tf.data.Dataset:
        def generator() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for sample in dataset:
                window = np.array(sample["window"], dtype=np.int8)
                scalars = np.array(sample["scalars"], dtype=np.int8)

                if self.mask_tokens > 0:
                    train_window = self._apply_mask(window.copy())
                    yield train_window, window
                else:
                    yield window, window

        output_signature = (
            tf.TensorSpec(shape=(None, 69), dtype=tf.int8),
            tf.TensorSpec(shape=(None, 69), dtype=tf.int8),
        )

        tf_ds = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )

        if self.shuffle_buffer_size > 0:
            tf_ds = tf_ds.shuffle(self.shuffle_buffer_size)

        tf_ds = tf_ds.batch(self.batch_size)
        tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE)

        return tf_ds

    def _memory_to_tf_dataset(self, dataset) -> tf.data.Dataset:
        windows = np.array([s["window"] for s in dataset], dtype=np.int8)

        if self.mask_tokens > 0:
            train_windows = np.array([self._apply_mask(w.copy()) for w in windows])
            tf_ds = tf.data.Dataset.from_tensor_slices((train_windows, windows))
        else:
            tf_ds = tf.data.Dataset.from_tensor_slices((windows, windows))

        if self.shuffle_buffer_size > 0:
            tf_ds = tf_ds.shuffle(self.shuffle_buffer_size)

        tf_ds = tf_ds.batch(self.batch_size)
        tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE)

        return tf_ds

    def _apply_mask(self, window: np.ndarray) -> np.ndarray:
        if self.mask_tokens <= 0:
            return window

        num_positions = len(window)
        for i in range(num_positions):
            tokens = np.arange(69)
            np.random.shuffle(tokens)
            mask_tokens = tokens[: self.mask_tokens]
            window[i, mask_tokens] = self.mask_token_id

        return window

    def with_transformation(
        self,
        transform_fn: Callable[[np.ndarray], np.ndarray],
    ) -> "TrainingDataGenerator":
        transformed_gen = TrainingDataGenerator(
            repo_name=self.repo_name,
            split=self.split,
            encoding=self.encoding,
            shuffle_buffer_size=self.shuffle_buffer_size,
            batch_size=self.batch_size,
            mask_tokens=self.mask_tokens,
            mask_token_id=self.mask_token_id,
        )

        if self._dataset is not None:
            transformed_gen._dataset = self._dataset.map(
                lambda x, y: (transform_fn(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        return transformed_gen

    def with_masking(self, num_tokens: int) -> "TrainingDataGenerator":
        return TrainingDataGenerator(
            repo_name=self.repo_name,
            split=self.split,
            encoding=self.encoding,
            shuffle_buffer_size=self.shuffle_buffer_size,
            batch_size=self.batch_size,
            mask_tokens=num_tokens,
            mask_token_id=self.mask_token_id,
        )

    def with_split(self, split: str) -> "TrainingDataGenerator":
        return TrainingDataGenerator(
            repo_name=self.repo_name,
            split=split,
            encoding=self.encoding,
            shuffle_buffer_size=self.shuffle_buffer_size,
            batch_size=self.batch_size,
            mask_tokens=self.mask_tokens,
            mask_token_id=self.mask_token_id,
        )

    def on_epoch_end(self) -> None:
        self._epoch_count += 1

        if self._dataset is not None and self.shuffle_buffer_size > 0:
            self._dataset = self._dataset.shuffle(self.shuffle_buffer_size)

    @property
    def epoch_count(self) -> int:
        return self._epoch_count

    def __iter__(self):
        if self._dataset is None:
            self.to_tf_dataset()
        return iter(self._dataset)
