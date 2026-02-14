from functools import wraps
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from chesspos.models.autoencoder import AutoencoderModel

class CnnAutoencoder(AutoencoderModel):
	@wraps(AutoencoderModel.__init__)
	def __init__(self, **kwargs):
		self._final_conv_shape = None
		self.embedding_size = 256
		super().__init__(**kwargs)

	def _model_helper(self) -> dict:
		encoder_input = layers.Input(shape=(8,8,15,1), dtype=tf.float16)

		# Encoder
		x = layers.Conv3D(32, (3, 3, 15), activation="relu", padding="same")(encoder_input)
		x = layers.MaxPooling3D((2, 2, 1), padding="same")(x)
		x = layers.Conv3D(32, (3, 3, 15), activation="relu", padding="same")(x)
		x = layers.MaxPooling3D((2, 2, 1), padding="same")(x)

		# Decoder
		decoder_input = layers.Input(shape=(2,2,15,32), dtype=tf.float16)
		y = layers.Conv3DTranspose(32, (3, 3, 15), strides=(2,2,1), activation="relu", padding="same")(decoder_input)
		y = layers.Conv3DTranspose(32, (3, 3, 15), strides=(2,2,1), activation="relu", padding="same")(y)
		y = layers.Conv3D(1, (8, 8, 15), activation="sigmoid", padding="same")(y)

		encoder = keras.Model(inputs=encoder_input, outputs=x, name='encoder')
		decoder = keras.Model(inputs=decoder_input, outputs=y, name='decoder')
		autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

		return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}

	def _define_encoder(self) -> Model:
		return self._model_helper()['encoder']

	def _define_decoder(self):
		return self._model_helper()['decoder']

	def _define_model(self) -> Model:
		return self._model_helper()['autoencoder']