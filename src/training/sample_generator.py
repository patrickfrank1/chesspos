import numpy as np
import tensorflow as tf

from src.types import IOTensorPair
from src.utils.fileops import file_paths_from_directory


class AutoencoderDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory_path: str, batch_size: int) -> None:
        self.dtype = np.float16
        self.all_data_seen = False
        self.directory_path = directory_path
        self.files = file_paths_from_directory(self.directory_path, ".npz")
        self.current_file = self.files.pop()
        self.visited_files = []
        self.train_positions = None
        self.label_positions = np.load(self.current_file)['data'].astype(self.dtype)
        self.batch_size = batch_size

    def __len__(self) -> int:
        # Calculate the total number of batches based on the number of samples in the file and the batch size.
        return int(np.ceil(len(self.label_positions) / self.batch_size))

    def __getitem__(self, idx: int) -> IOTensorPair:
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.label_positions))
        label_batch = self.label_positions[low:high]
        if self.train_positions is None:
            return label_batch, label_batch
        else:
            train_batch = self.train_positions[low:high]
            return train_batch, label_batch

    def on_epoch_end(self) -> None:
        self.visited_files.append(self.current_file)
        if len(self.files) == 0:
            self.all_data_seen = True
            self.files = self.visited_files
            self.visited_files = []
        self.current_file = self.files.pop()
        self.label_positions = np.load(self.current_file)['data'].astype(self.dtype)

    def total_dataset_length(self) -> int:
        # Calculate the total number of batches based on the number of samples in all files and the batch size.
        number_samples = 0
        for file_path in file_paths_from_directory(self.directory_path, ".npz"):
            number_samples += len(np.load(file_path)['data'])
        return number_samples


class ReconstructAutoencoderDataGenerator(AutoencoderDataGenerator):
    def __init__(self, *args, number_squares, **kwargs) -> None:
        self.number_squares = number_squares
        self.mask_token = 16
        self.encoding = "token_sequence"  # token_sequence | tensor
        self.mask = None
        super().__init__(*args, **kwargs)

        if self.encoding == "tensor":
            self.mask = self._mask_tensor
        elif self.encoding == "token_sequence":
            self.mask = self._mask_token_sequence
        self.mask()

    def on_epoch_end(self) -> None:
        super().on_epoch_end()
        self.mask()

    def _mask_tensor(self) -> None:
        squares = np.arange(64)
        num_positions = len(self.label_positions)
        tmp_positions: np.ndarray = self.label_positions.reshape((num_positions, 64, 15))
        for i in range(num_positions):
            np.random.shuffle(squares)
            mask_squares = squares[:self.number_squares]
            tmp_positions[i, mask_squares, :] = self.mask_token
        self.train_positions = tmp_positions.reshape(num_positions, 8, 8, 15)

    def _mask_token_sequence(self) -> None:
        tmp_positions: np.ndarray = np.copy(self.label_positions)
        num_positions = len(tmp_positions)
        tokens = np.arange(69)
        for i in range(num_positions):
            np.random.shuffle(tokens)
            mask_squares = tokens[:self.number_squares]
            tmp_positions[i, mask_squares] = self.mask_token
        self.train_positions = tmp_positions
