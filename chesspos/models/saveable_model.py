from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model, Model

class SaveableModel(ABC):
	def __init__(self, save_dir: str, **kwargs) -> None:
		self.save_dir = save_dir
		self.model = self._define_model()
		for key, value in kwargs.items():
			print(f"Unknown argument: {key}={value} encountered and ignored for model instantiation.")

	@abstractmethod
	def _define_model(self) -> Model:
		pass

	def get_model(self) -> Model:
		return self.model

	def save(self) -> None:
		self.model.save(f"{self.save_dir}/model")


	def load(self) -> None:
		self.model = load_model(f"{self.save_dir}/model")