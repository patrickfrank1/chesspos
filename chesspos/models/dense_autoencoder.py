from functools import wraps
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from chesspos.models.autoencoder import AutoencoderModel

class DenseAutoencoder(AutoencoderModel):
	@wraps(AutoencoderModel.__init__)
	def __init__(self, **kwargs):
		self.embedding_size = 256
		super().__init__(**kwargs)

	def _model_helper(self) -> dict:
		encoder_input = layers.Input(shape=(8,8,15,1), dtype=tf.float16)
		encoder = layers.Reshape((8*8*15,))(encoder_input)
		encoder = layers.Dense(2*self.embedding_size, activation='relu')(encoder)
		encoder = layers.Dense(self.embedding_size, activation='relu')(encoder)

		decoder_input = layers.Input(shape=(self.embedding_size,))
		decoder = layers.Dense(2*self.embedding_size, activation='relu')(decoder_input)
		decoder = layers.Dense(8*8*15, activation='relu')(decoder_input)
		decoder = layers.Reshape((8,8,15,1))(decoder)

		encoder = keras.Model(inputs=encoder_input, outputs=encoder, name='encoder')
		decoder = keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')
		autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

		return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}

	def _define_encoder(self) -> Model:
		return self._model_helper()['encoder']

	def _define_decoder(self):
		return self._model_helper()['decoder']

	def _define_model(self) -> Model:
		return self._model_helper()['autoencoder']