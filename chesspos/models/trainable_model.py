import os
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard

from chesspos.models.saveable_model import SaveableModel

class TrainableModel(SaveableModel):
	def __init__(
		self,
		save_dir,
		train_generator,
		test_generator,
		train_steps_per_epoch,
		test_steps_per_epoch,
		optimizer='rmsprop',
		loss=None,
		metrics=None,
		hide_tf_warnings = False,
		tf_callbacks = None
	):
		super().__init__(save_dir)

		self.EXCESS_VALIDATION_EPOCHS = 10

		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics

		self.save_dir = save_dir
		self.train_generator = train_generator
		self.test_generator = test_generator
		self.train_steps_per_epoch = train_steps_per_epoch
		self.test_steps_per_epoch = test_steps_per_epoch
		self.hide_tf_warnings = hide_tf_warnings,
		self.tf_callbacks = self._set_tf_callbacks(tf_callbacks)


	def _set_tf_callbacks(self, callback_array):
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
							keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10,
							verbose=0, mode='min', restore_best_weights=True)
						)
					elif callback == 'checkpoints':
						save_dir = os.path.abspath(self.save_dir)
						callbacks.append(
							keras.callbacks.ModelCheckpoint(filepath=save_dir+"/checkpoints/cp-{epoch:04d}.ckpt",
							save_weights_only=False, save_best_only=True, mode='min', verbose=1)
						)
					else:
						print(f"WARNING: illegal arguent {callback} in tf_callbacks, skipping.")
		else:
			raise ValueError("'callback_array' needs to be a list.")

		return callbacks


	def _train_samples(self):
		return  1.0 * self.train_generator.number_samples() * len(self.train_generator.subsampling_functions)


	def _test_samples(self):
		return 1.0 * self.test_generator.number_samples() * len(self.test_generator.subsampling_functions)


	def _train_epochs(self):
		return self._train_samples() / self.train_generator.batch_size / self.train_steps_per_epoch


	def _test_epochs(self):
		return self._test_samples() / self.test_generator.batch_size / self.test_steps_per_epoch


	def get_model(self):
		return self.model


	def compile(self):
		if self.model is None:
			raise Exception("No model to compile.")

		self.model.compile(
			optimizer=self.optimizer,
			loss=self.loss,
			metrics=self.metrics
		)


	def _check_train_test_ratio(self):
		train_epochs = self._train_epochs()
		test_epochs = self._test_epochs()
		print(f'You have enough training samples for {train_epochs} epochs and enought validation samples for {test_epochs} epochs.')

		if train_epochs > test_epochs:
			raise ValueError("Not enought validation samples provided to start training! Cancelling.")
		else:
			print(f"\nTraining on {self._train_samples() / 1.e6} million training samples.")
			print(f"Validating on {math.floor(train_epochs) * self.test_generator.batch_size * self.test_steps_per_epoch / 1.e6} million validation samples.")
			if test_epochs > self.EXCESS_VALIDATION_EPOCHS + train_epochs:
				print("WARNING: your are providing much more validation samples than nessecary. Those could be used for training instead.")


	def train(self):
		if self.hide_tf_warnings:
			os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

		train_generator = self.train_generator.get_generator()
		test_generator = self.test_generator.get_generator()

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

	def predict(self, samples):
		return self.model.predict(samples)

