from functools import wraps
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from chesspos.models.autoencoder import AutoencoderModel

class ResnetAutoencoder(AutoencoderModel):
	@wraps(AutoencoderModel.__init__)
	def __init__(self, **kwargs):
		self._final_conv_shape = None
		self.embedding_size = 256
		super().__init__(**kwargs)

	def _model_helper(self) -> dict:
		encoder_input = layers.Input(shape=(8,8,15,1), dtype=tf.float16)

		conv_1 = layers.Conv3D(16, (2,2,15), activation="relu", padding="same")(encoder_input)
		conv_1 = layers.BatchNormalization()(conv_1)
		conv_2 = layers.Conv3D(32, (2,2,15), activation="relu", padding="same")(conv_1)
		conv_2 = layers.BatchNormalization()(conv_2)
		#conv_3 = layers.Conv3D(64, (2,2,15), activation="relu", padding="same")(conv_2)
		#conv_3 = layers.BatchNormalization()(conv_3)
		#conv_4 = layers.Conv3D(64, (2,2,15), activation="relu", padding="same")(conv_3)
		#conv_4 = layers.BatchNormalization()(conv_4)
		#conv_5 = layers.Conv3D(128, (2,2,15), activation="relu", padding="same")(conv_4)
		#conv_5 = layers.BatchNormalization()(conv_5)
		#conv_6 = layers.Conv3D(256, (2,2,15), activation="relu", padding="same")(conv_5)
		#conv_6 = layers.BatchNormalization()(conv_6)
		#conv_7 = layers.Conv3D(512, (2,2,15), activation="relu", padding="same")(conv_6)
		#conv_7 = layers.BatchNormalization()(conv_7)
		
		self._final_conv_shape = conv_2.shape[1:]
		print("final_conv_shape:", self._final_conv_shape)
		embedding = layers.Flatten()(conv_2)
		embedding = layers.Dense(self.embedding_size, activation="relu")(embedding)

		decoder_input = layers.Input(shape=(self.embedding_size,))
		decoder_dense = layers.Dense(np.prod(self._final_conv_shape), activation="relu")(decoder_input)
		decoder_dense = layers.Reshape(self._final_conv_shape)(decoder_dense)
		decoder_conv_1 = layers.Conv3DTranspose(32, (3,3,15), activation="relu", padding="same")(decoder_dense)
		decoder_conv_1 = layers.BatchNormalization()(decoder_conv_1)
		decoder_conv_2 = layers.Conv3DTranspose(16, (3,3,15), activation="relu", padding="same")(decoder_conv_1)
		decoder_conv_2 = layers.BatchNormalization()(decoder_conv_2)
		#decoder_conv_3 = layers.Conv3DTranspose(1, (3,3,15), activation="relu", padding="same")(decoder_conv_2)
		#decoder_conv_3 = layers.BatchNormalization()(decoder_conv_3)
		#decoder_conv_4 = layers.Conv3DTranspose(1, (2,2,15), activation="sigmoid", padding="same")(decoder_conv_3)
		#decoder_conv_4 = layers.BatchNormalization()(decoder_conv_4)
		#decoder_conv_5 = layers.Conv3DTranspose(16, (2,2,15), activation="relu", padding="same")(decoder_conv_1)
		#decoder_conv_5 = layers.BatchNormalization()(decoder_conv_5)
		#decoder_conv_6 = layers.Conv3DTranspose(8, (2,2,15), activation="relu", padding="same")(decoder_conv_5)
		#decoder_conv_6 = layers.BatchNormalization()(decoder_conv_6)
		#decoder_conv_7 = layers.Conv3DTranspose(1, (2,2,15), activation="sigmoid", padding="same")(decoder_conv_6)
		reconstructed_input = decoder_conv_2

		encoder = keras.Model(inputs=encoder_input, outputs=embedding, name='encoder')
		decoder = keras.Model(inputs=decoder_input, outputs=reconstructed_input, name='decoder')
		autoencoder = keras.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)), name='autoencoder')

		return {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}

	def _define_encoder(self) -> Model:
		return self._model_helper()['encoder']

	def _define_decoder(self):
		return self._model_helper()['decoder']

	def _define_model(self) -> Model:
		return self._model_helper()['autoencoder']