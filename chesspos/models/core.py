import tensorflow as tf
from tensorflow import keras

class TrainableModel():
	def __init__(self, optimizer='rmsprop', loss=None, metrics=None):
		self.model = None
		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics

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
