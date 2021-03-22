import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DenseNetwork():
	def __init__(
		self,
		input_size,
		output_size,
		hidden_layers=[],
		name="dense network"
	):
		self.name = name
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_layers = hidden_layers

		self.model = self.build_model()

	def build_model(self):
		input_layer = layers.Input(shape=self.input_size, name=self.name, dtype=tf.float32)
		output_layer = None

		if len(self.hidden_layers) == 0:
			output_layer = layers.Dense(self.output_size, activation='relu')(input_layer)
		else:
			if self.hidden_layers[0] > 0.0 and self.hidden_layers[0] < 1.0:
				x = layers.Dropout(rate=self.hidden_layers[0])(input_layer)
			else:
				x = layers.Dense(self.hidden_layers[0], activation='relu')(input_layer)

			for i in range(1, len(self.hidden_layers)):
				if self.hidden_layers[i] > 0.0 and self.hidden_layers[i] < 1.0:
					x = layers.Dropout(rate=self.hidden_layers[i])(x)
				else:
					x = layers.Dense(self.hidden_layers[i], activation='relu')(x)

			output_layer = layers.Dense(self.output_size, activation='relu')(x)

		model = keras.Model(inputs=input_layer, outputs=output_layer, name=self.name)
		return model

	def get_model(self):
		return self.model

