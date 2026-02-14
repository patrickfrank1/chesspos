from abc import abstractmethod
from functools import wraps
from typing import Callable, Dict, List, overload
import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import chess


import tensorflow as tf
from tensorflow import keras

from chesspos.models.saveable_model import SaveableModel
from chesspos.preprocessing.sample_generator import SampleGenerator

class TrainableModel(SaveableModel):
	@wraps(SaveableModel.__init__)
	def __init__(
		self,
		train_generator: SampleGenerator,
		test_generator: SampleGenerator,
		train_steps_per_epoch: int,
		test_steps_per_epoch: int,
		optimizer = 'rmsprop',
		board_to_input: Callable[[chess.Board], np.ndarray] = None,
		loss = None,
		metrics = None,
		hide_tf_warnings: bool = True,
		tf_callbacks = None,
		**kwargs
	) -> None:

		self.EXCESS_VALIDATION_EPOCHS = 10

		self.board_to_input = board_to_input
		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics

		self.train_generator = train_generator
		self.test_generator = test_generator
		self.train_steps_per_epoch = train_steps_per_epoch
		self.test_steps_per_epoch = test_steps_per_epoch
		self.hide_tf_warnings = hide_tf_warnings

		super(TrainableModel, self).__init__(**kwargs)

		self.tf_callbacks = self._set_tf_callbacks(tf_callbacks)
		print(self.loss)

	@abstractmethod
	def _define_model(self) -> keras.Model:
		pass

	def _set_tf_callbacks(self, callback_array) -> List[keras.callbacks.Callback]:
		callbacks = []
		if callback_array is None:
			pass
		elif isinstance(callback_array, list):
			for callback in callback_array:
				if isinstance(callback, (
					keras.callbacks.EarlyStopping,
					keras.callbacks.ModelCheckpoint,
					keras.callbacks.TensorBoard,
					keras.callbacks.Callback)
				):
					callbacks.append(callback)
				elif isinstance(callback, str):
					if callback == 'early_stopping':
						callbacks.append(
							keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=10,
							verbose=0, mode='min', restore_best_weights=True)
						)
					elif callback == 'checkpoints':
						save_dir = os.path.abspath(self.save_dir)
						callbacks.append(
							keras.callbacks.ModelCheckpoint(filepath=save_dir+"/checkpoints/cp-{epoch:04d}.ckpt",
							save_weights_only=False, save_best_only=True, mode='min', verbose=1)
						)
					else:
						print(f"WARNING: illegal argument {callback} in tf_callbacks, skipping.")
		else:
			raise ValueError("'callback_array' needs to be a list.")
		
		return callbacks


	def _train_samples(self) -> int:
		return  1.0 * self.train_generator.number_samples


	def _test_samples(self) -> int:
		return 1.0 * self.test_generator.number_samples


	def _train_epochs(self) -> int:
		return self._train_samples() / self.train_generator.batch_size / self.train_steps_per_epoch


	def _test_epochs(self) -> int:
		return self._test_samples() / self.test_generator.batch_size / self.test_steps_per_epoch


	def get_model(self) -> keras.Model:
		return self.model


	def _compile(self) -> None:
		self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
		keras.utils.plot_model(self.model, to_file=f"{self.save_dir}/model.png", show_shapes=True)
		self.model.summary()


	def _check_train_test_ratio(self):
		train_epochs = self._train_epochs()
		test_epochs = self._test_epochs()
		print(f'You have enough training samples for {train_epochs} epochs and enough validation samples for {test_epochs} epochs.')

		if train_epochs > test_epochs:
			raise ValueError("Not enough validation samples provided to start training! Cancelling.")
		else:
			print(f"\nTraining on {self._train_samples() / 1.e6} million training samples.")
			print(f"Validating on {math.floor(train_epochs) * self.test_generator.batch_size * self.test_steps_per_epoch / 1.e6} million validation samples.")
			if test_epochs > self.EXCESS_VALIDATION_EPOCHS + train_epochs:
				print("WARNING: your are providing much more validation samples than necessary. Those could be used for training instead.")

	def _plot_train_history(self, history: keras.callbacks.History) -> None:
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.save_dir+'/loss.png')

	def train(self) -> Dict:
		if self.hide_tf_warnings:
			os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

		# TODO: benchmark
		#train_generator = self.train_generator.get_generator()
		#test_generator = self.test_generator.get_generator()
		train_generator = self.train_generator.get_tf_dataset()
		test_generator = self.test_generator.get_tf_dataset()

		self._check_train_test_ratio()

		history = self.model.fit(
			train_generator,
			steps_per_epoch = self.train_steps_per_epoch,
			epochs = math.floor(self._train_epochs()),
			validation_data = test_generator,
			validation_steps = self.test_steps_per_epoch,
			callbacks = self.tf_callbacks
		)
		return history

	def predict(self, samples: np.ndarray) -> np.ndarray:
		return self.model.predict(samples)

	def _check_input_converter(self) -> None:
		if self.board_to_input is None:
			raise ValueError("No board to model input converter set.")

		start_board = chess.Board()
		converter_shape = self.board_to_input(start_board).shape
		input_shape = self.train_generator.sample_shape
		if not converter_shape == input_shape:
			raise ValueError(f"board_to_input converter is not compatible with model input shape.\n"
				f"Expected shape: {converter_shape}\n"
				f"Input shape: {input_shape}")

	def predict_from_board(self, boards: List[chess.Board]) -> np.ndarray:
		self._check_input_converter()
		inputs = np.empty((len(boards), *self.train_generator.sample_shape))
		for i, board in enumerate(boards):
			inputs[i] = self.board_to_input(board)
		return self.predict(inputs)

	def evaluate(self, samples: np.ndarray, labels: np.ndarray) -> np.float32:
		return self.model.evaluate(samples, labels)

	def evaluate_from_board(self, boards: List[chess.Board], labels) -> np.float32:
		self._check_input_converter()
		inputs = np.empty((len(boards), *self.train_generator.sample_shape))
		for i, board in enumerate(boards):
			inputs[i] = self.board_to_input(board)
		return self.evaluate(inputs, labels)
