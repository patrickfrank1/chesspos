import tensorflow as tf
from tensorflow import keras
from keras.models import load_model, Model

class SaveableModel():
	def __init__(self,
		save_dir: str,
	) -> None:
		self.save_dir = save_dir

		self.model: Model = None


	def save(self) -> None:
		self.model.save(f"{self.save_dir}/model")


	def load(self) -> None:
		self.model = load_model(f"{self.save_dir}/model")