import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from chesspos.models.trainable_model import TrainableModel
from chesspos.models.dense import DenseNetwork
from chesspos.models.chessposition_inspectable_autoencoder import ChesspositionInspectableAutoencoderMixin

class DenseAutoencoder(TrainableModel, ChesspositionInspectableAutoencoderMixin):
	def __init__(
		self,
		input_size,
		embedding_size,
		train_generator,
		test_generator,
		train_steps_per_epoch,
		test_steps_per_epoch,
		safe_dir,
		hidden_layers=[],
		optimizer='rmsprop',
		loss=None,
		metrics=None,
		tf_callbacks = None
	):
		super().__init__(
			safe_dir, train_generator, test_generator, train_steps_per_epoch,
			test_steps_per_epoch, optimizer, loss, metrics, tf_callbacks=tf_callbacks
		)

		self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_layers = hidden_layers

		self.encoder = None
		self.decoder = None

		self.build_model()


	def build_model(self):
		encoder = DenseNetwork(self.input_size, self.embedding_size, self.hidden_layers, name='encoder').get_model()
		encoder.summary()
		self.encoder = encoder

		decoder = DenseNetwork(self.embedding_size, self.input_size, self.hidden_layers[::-1], name='decoder').get_model()
		decoder.summary()
		self.decoder = decoder

		encoder_input = layers.Input(shape=self.input_size, dtype=tf.float32, name='autoencoder')
		decoder_input = encoder(encoder_input)
		decoder_output = decoder(decoder_input)
		model = keras.Model(inputs=encoder_input, outputs=decoder_output, name='autoencoder')
		model.summary()
		self.model = model


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
