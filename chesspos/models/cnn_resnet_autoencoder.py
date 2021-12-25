import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from chesspos.models.trainable_model import TrainableModel
from chesspos.models.chessposition_inspectable_autoencoder import ChesspositionInspectableAutoencoderMixin

class CnnResnetAutoencoder(TrainableModel, ChesspositionInspectableAutoencoderMixin):
	def __init__(
		self,
		input_size,
		embedding_size,
		train_generator,
		test_generator,
		train_steps_per_epoch,
		test_steps_per_epoch,
		save_dir,
		hidden_layers=[],
		optimizer='rmsprop',
		loss=None,
		metrics=None,
		tf_callbacks = None
	):
		super().__init__(
			save_dir, train_generator, test_generator, train_steps_per_epoch,
			test_steps_per_epoch, optimizer, loss, metrics, tf_callbacks = tf_callbacks
		)

		self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_layers = hidden_layers
		self.filters = 128

		self.encoder = None
		self.decoder = None

		self.build_model()

	def build_model(self):
		input_layer = layers.Input(shape=773, dtype=tf.float16)

		board_layer = layers.Lambda(lambda x: x[:,:768], output_shape=(768,))(input_layer)
		board_layer = layers.Reshape((8,8,12))(board_layer)

		metadata_layer = layers.Lambda(lambda x: x[:,768:], output_shape=(5,))(input_layer)

		
		encoder = layers.Conv2D(self.filters, (3, 3), activation="relu", padding="same")(board_layer)
		encoder_1 = layers.BatchNormalization()(encoder)
		encoder = layers.Conv2D(self.filters, (3, 3), activation="relu", padding="same")(encoder_1)
		encoder = layers.BatchNormalization()(encoder) + encoder_1
		encoder = layers.Flatten()(encoder)
		encoder = layers.Dense(128, activation="relu")(encoder)
		encoder = layers.Dropout(0.3)(encoder)
		encoder = layers.Dense(64, activation="relu")(encoder)
		
		encoder_model = keras.Model(inputs=input_layer, outputs=encoder, name='encoder')
		encoder_model.summary()
		self.encoder = encoder_model

		decoder_input = layers.Input(shape=64)
		decoder = layers.Dense(128, activation="relu")(decoder_input)
		decoder = layers.Dropout(0.3)(decoder)
		decoder = layers.Dense(8192, activation="relu")(decoder)
		decoder_1 = layers.Reshape((8,8,self.filters))(decoder)
		decoder = layers.Conv2DTranspose(self.filters, (3, 3), activation="relu", padding="same")(decoder_1)
		decoder = layers.BatchNormalization()(decoder) + decoder_1
		decoder = layers.Conv2DTranspose(12, (3, 3), activation="relu", padding="same")(decoder)
		decoder = layers.Reshape((768,))(decoder)

		decoder_model = keras.Model(inputs=decoder_input, outputs=decoder, name='decoder')
		decoder_model.summary()
		self.decoder = decoder_model

		autoencoder = encoder_model(input_layer)
		autoencoder = decoder_model(autoencoder)
		output_layer = layers.Concatenate()([autoencoder, metadata_layer])

		autoencoder_model = keras.Model(inputs=input_layer, outputs=output_layer, name='autocoder')
		autoencoder_model.summary()
		self.model = autoencoder_model

	def compile(self):
		super().compile()
		self.encoder.compile(optimizer='rmsprop', loss=None, metrics=None)
		self.decoder.compile(optimizer='rmsprop', loss=None, metrics=None)


	def get_encoder(self):
		if self.encoder is None:
			raise Exception("No encoder model defined.")
		else:
			return self.encoder


	def get_decoder(self):
		if self.decoder is None:
			raise Exception("No decoder model defined.")
		else:
			return self.decoder
