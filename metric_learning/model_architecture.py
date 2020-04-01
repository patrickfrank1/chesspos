import tensorflow as tf
from tensorflow import keras
import numpy as np


class TripletLossLayer(keras.layers.Layer):
	def __init__(self, alpha, **kwargs):
		self.alpha = alpha
		super(TripletLossLayer, self).__init__(**kwargs)

	def triplet_loss(self, inputs):
		anchor, positive, negative = inputs
		pos_dist = tf.reduce_sum(tf.square(anchor-positive), axis=-1)
		neg_dist = tf.reduce_sum(tf.square(anchor-negative), axis=-1)
		return tf.reduce_sum(tf.maximum(pos_dist - neg_dist + self.alpha, 0), axis=0)

	def call(self, inputs):
		loss = self.triplet_loss(inputs)
		self.add_loss(loss)
		return loss

def triplet_accuracy(embeddings):
	'''
	Wrapper function that returns a loss function: loss(y_true, y_pred).
	Neither of which is actually used for calculating the metric.
	See: https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
	'''
	print("called triplet accuracy")
	def loss(y_true, y_pred):
		print("called triplet accuracy metric function")
		return 2
		anchor, positive, negative = embeddings[0], embeddings[1], embeddings[2]
		pos_dist = tf.reduce_sum(tf.square(anchor-positive), axis=-1)
		neg_dist = tf.reduce_sum(tf.square(anchor-negative), axis=-1)
		#return tf.reduce_sum(tf.cast(tf.cast(pos_dist < neg_dist, dtype=tf.int32), dtype=tf.float32), axis=0)

	return loss

def mean_pred(y_true, y_pred):
	return tf.reduce_mean(y_pred)


def embedding_network(input_shape, embedding_size, hidden_layers=None):
	if hidden_layers is None:
		return keras.layers.Dense(embedding_size, activation='relu', input_shape=input_shape,
									name="embedding_layer")
	else:
		embedding = keras.Sequential(name="embedding_model")
		embedding.add(keras.layers.Dense(hidden_layers[0], activation='relu',
						input_shape=input_shape,name="embedding_layer_0")
		)
		for i in range(1, len(hidden_layers)-1):
			embedding.add(keras.layers.Dense(hidden_layers[i], activation='relu',
							name=f"embedding_layer_{i}")
			)
		embedding.add(keras.layers.Dense(embedding_size, activation='relu',
						name=f"embedding_layer_{len(hidden_layers)}")
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

	# three embeddings
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

	# compile the model
	optimizer = keras.optimizers.Adam(lr=0.00006)
	#t_acc = triplet_accuracy([embedding_a, embedding_p, embedding_n])
	def test(y_true, y_pred):
		print(y_true)
		print(y_pred)
		return 0


	triplet_network.compile(
		loss=None,
		optimizer=optimizer,
		metrics=[test] # not working, why?
	)

	triplet_network.summary()
	keras.utils.plot_model(triplet_network, 'triplet_network.png', show_shapes=True)

	return triplet_network
