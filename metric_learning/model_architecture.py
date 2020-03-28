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
		return tf.reduce_sum(tf.maximum(pos_dist - neg_dist + self.alpha, 0), axis=0)

	def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss