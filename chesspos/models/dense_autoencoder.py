import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from chesspos.models.core import TrainableModel
from chesspos.models import DenseNetwork

class DenseAutoencoder(TrainableModel):
	def __init__(
		self,
		input_size,
		embedding_size,
		hidden_layers=[],
		optimizer='rmsprop',
		loss=None,
		metrics=None
	):
		super(DenseAutoencoder, self).__init__(optimizer, loss, metrics)
		self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_layers = hidden_layers
		self.build_model()

	def build_model(self):
		encoder = DenseNetwork(self.input_size, self.embedding_size, self.hidden_layers, name='encoder').get_model()
		decoder = DenseNetwork(self.embedding_size, self.input_size, self.hidden_layers[::-1], name='decoder').get_model()
		encoder.summary()
		decoder.summary()

		encoder_input = layers.Input(shape=self.input_size, dtype=tf.float32, name='autoencoder')
		decoder_input = encoder(encoder_input)
		decoder_output = decoder(decoder_input)
		self.model = keras.Model(inputs=encoder_input, outputs=decoder_output, name='autoencoder')
		self.model.summary()


