import tensorflow as tf
from tensorflow import keras


class TripletLossLayer(keras.layers.Layer):
	def __init__(self, alpha, **kwargs):
		self.alpha = alpha
		super(TripletLossLayer, self).__init__(**kwargs)

	def triplet_loss(self, inputs):
		anchor, positive, negative = inputs
		pos_dist = tf.reduce_sum(tf.square(anchor-positive), axis=-1)
		neg_dist = tf.reduce_sum(tf.square(anchor-negative), axis=-1)
		return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + self.alpha, 0), axis=0)

	def call(self, inputs):
		loss = self.triplet_loss(inputs)
		self.add_loss(loss)
		return loss

class AutoencoderTripletLossLayer(keras.layers.Layer):
	def __init__(self, triplet_weight_ratio, **kwargs):
		self.triplet_weight_ratio = triplet_weight_ratio
		super(AutoencoderTripletLossLayer, self).__init__(**kwargs)

	def triplet_loss(self, inputs):
		return inputs[0]

	def autoencoder_loss(self, inputs):
		_, inp_a, inp_p, inp_n, out_a, out_p, out_n = inputs 
		reconstruct_anc = tf.reduce_sum(tf.square(inp_a-out_a), axis=-1)
		reconstruct_pos = tf.reduce_sum(tf.square(inp_p-out_p), axis=-1)
		reconstruct_neg = tf.reduce_sum(tf.square(inp_n-out_n), axis=-1)
		return tf.multiply(1./3., tf.reduce_mean(reconstruct_anc, axis=0) +\
			tf.reduce_mean(reconstruct_pos, axis=0) + tf.reduce_mean(reconstruct_neg, axis=0)
		)

	def call(self, inputs):
		loss = self.triplet_weight_ratio * self.triplet_loss(inputs) + self.autoencoder_loss(inputs)
		self.add_loss(loss)
		return loss

def embedding_network(input_shape, embedding_size, hidden_layers=None, name="embedding_model"):
	if hidden_layers is None:
		return keras.layers.Dense(
			embedding_size, activation='relu', input_shape=input_shape,name="embedding_layer"
		)
	else:
		embedding = keras.Sequential(name="embedding_model")
		embedding.add(
			keras.layers.Dense(hidden_layers[0], activation='relu',
				input_shape=input_shape,name="embedding_layer_0"
			)
		)
		for i in range(1, len(hidden_layers)-1):
			embedding.add(
				keras.layers.Dense(hidden_layers[i], activation='relu',name=f"embedding_layer_{i}")
			)
		embedding.add(
			keras.layers.Dense(embedding_size, activation='relu',
				name=f"embedding_layer_{len(hidden_layers)}"
			)
		)
		return embedding


def triplet_network_model(input_shape, embedding_size, hidden_layers=None, alpha=0.2):
	# Input layers
	anchor_input = keras.layers.Input(input_shape, name="anchor_input", dtype=float)
	positive_input = keras.layers.Input(input_shape, name="positive_input", dtype=float)
	negative_input = keras.layers.Input(input_shape, name="negative_input", dtype=float)

	# Generate the encodings (feature vectors) for the three positions
	embedding = embedding_network(input_shape, embedding_size, hidden_layers=hidden_layers)
	embedding.summary()

	# Embeddings for the three inputs
	embedding_a = embedding(anchor_input)
	embedding_p = embedding(positive_input)
	embedding_n = embedding(negative_input)

	# TripletLoss Layer, initialize and incorporate into network, tie embeddings together
	loss_layer = TripletLossLayer(alpha=alpha, name='triplet_loss_layer')([embedding_a, embedding_p, embedding_n])

	# Cast as tf model
	triplet_network = keras.models.Model(
		inputs=[anchor_input, positive_input, negative_input],
		outputs=[loss_layer, embedding_a, embedding_p, embedding_n]
	)

	# Compile the model
	optimizer = keras.optimizers.Adam(lr=0.00006)

	def mean_pred(y_true, y_pred): # pylint: disable=unused-argument,dangerous-default-value
		return print("hello")

	triplet_network.compile(
		loss=None,
		optimizer=optimizer,
		metrics=[mean_pred] # call to any metric not working, why?
	)

	# Print model summary
	triplet_network.summary()

	return triplet_network

def triplet_network_autoencoder(input_shape, embedding_size, hidden_layers=None, alpha=0.2):
	# Input layers
	anchor_input = keras.layers.Input(input_shape, name="anchor_input", dtype=float)
	positive_input = keras.layers.Input(input_shape, name="positive_input", dtype=float)
	negative_input = keras.layers.Input(input_shape, name="negative_input", dtype=float)

	# Generate the encodings (feature vectors) for the three positions
	encoder = embedding_network(input_shape, embedding_size, hidden_layers=hidden_layers, name="encoder_network")
	encoder.summary()

	# Embeddings for the three inputs
	encoder_a = encoder(anchor_input)
	encoder_p = encoder(positive_input)
	encoder_n = encoder(negative_input)

	# TripletLoss Layer, initialize and incorporate into network, tie embeddings together
	loss_layer = TripletLossLayer(alpha=alpha, name='triplet_loss_layer')([encoder_a, encoder_p, encoder_n])

	# Initialise decoder
	decoder = embedding_network((embedding_size,), input_shape[0], hidden_layers=hidden_layers[::-1], name="decoder_network")
	decoder.summary()

	# decode embeddings
	decoder_a = decoder(encoder_a)
	decoder_p = decoder(encoder_p)
	decoder_n = decoder(encoder_n)

	# Autoencoder Loss Layer
	autoencoder_layer = AutoencoderLossLayer(
		triplet_weight_ratio=triplet_weight_ratio,
		name="autoencoder_layer"
	)
	autoencoder_loss = autoencoder_layer(
		[triplet_loss, anchor_input, positive_input, negative_input, decoder_a, decoder_p, decoder_n]
	)

	# Cast as tf model
	autoencoder_triplet_network = keras.models.Model(
		inputs=[anchor_input, positive_input, negative_input],
		outputs=[final_loss, encoder_a, encoder_p, encoder_n]
	)
	autoencoder_triplet_network.summary()

	# Compile the model
	optimizer = keras.optimizers.Adam(lr=0.00006)

	def mean_pred(y_true, y_pred): # pylint: disable=unused-argument,dangerous-default-value
		return print("hello")

	autoencoder_triplet_network.compile(
		loss=None,
		optimizer=optimizer,
		metrics=[mean_pred] # call to any metric not working, why?
	)

	return autoencoder_triplet_network

