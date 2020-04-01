import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from triplet_preparation import inputs_from_tuples, tuples_from_file_array, train_inputs_file_array_generator, train_inputs_length
from model_architecture import triplet_network_model

print(f"tf.Version.__version__: {tf.__version__}")
print(f"tf.keras.__version__: {tf.keras.__version__}")

class SkMetrics(keras.callbacks.Callback):
	def __init__(self, valid_data, batch_size, steps_per_callback=10):
		super(SkMetrics, self).__init__()
		self.valid_data = valid_data
		self.batch_size = batch_size
		self.steps_per_callback = steps_per_callback

	def predict_correct(self, predictions):
		anchor = predictions[1]
		positive = predictions[2]
		negative = predictions[3]
		pos_dist = tf.reduce_sum(tf.square(anchor-positive), axis=-1)
		neg_dist = tf.reduce_sum(tf.square(anchor-negative), axis=-1)
		return tf.reduce_sum(tf.cast(pos_dist < neg_dist, dtype=tf.int32), axis=0)

	def on_train_begin(self, logs={}):
		self.num_correct = []
		self.frac_correct = []
		self.diagnostics = []

	def on_epoch_end(self, epoch, logs={}):
		correct = tf.Variable(0)
		self.diagnostics.append("correct variable initialized")
		for i in range(self.steps_per_callback):
			predictions = self.model.predict_on_batch(next(self.valid_data))
			self.diagnostics.append(f"prediction {i} successful")
			correct.assign_add(self.predict_correct(predictions))
			self.diagnostics.append("correct variable updated")
		self.num_correct.append(correct)
		frac = tf.cast(correct, dtype=tf.float16)/tf.Variable(self.batch_size*self.steps_per_callback, dtype=tf.float16)
		self.frac_correct.append(frac.numpy())
		print(f" triplet_acc: {self.frac_correct[-1]}")

samples_generator = train_inputs_file_array_generator(
	[os.path.abspath('data/samples/lichess_db_standard_rated_2013-01-tuples.h5')],
	table_id_prefix="tuples",
	tuple_indices=[0,1,2,3,4,5,6],
	batch_size=16
)

skmetrics = SkMetrics(samples_generator, batch_size=16, steps_per_callback=10)




'''Initialize triplet network for training'''
input_shape = (773,)
embedding_size = 10
model = triplet_network_model(input_shape, embedding_size, hidden_layers=[512,256,64])

# input arguments
train_files = [
	#os.path.abspath('data/samples/lichess_db_standard_rated_2020-02-06-tuples-strong.h5'),
	os.path.abspath('data/samples/lichess_db_standard_rated_2013-02-tuples.h5')
]

validation_files = [
	os.path.abspath('data/samples/lichess_db_standard_rated_2013-01-tuples.h5')
]

batch_size = 16
steps_per_epoch = 1000
yield_augmented = 1

train_len = train_inputs_length(train_files, table_id_prefix="tuples")
print(f"{train_len} training samples.")

# generators for trian and test data
train_generator = train_inputs_file_array_generator(train_files, table_id_prefix="tuples",
					tuple_indices=[0,1,2,3,4,5,6], batch_size=batch_size)
validation_generator = train_inputs_file_array_generator(validation_files, table_id_prefix="tuples",
					tuple_indices=[0,1,2,3,4,5,6], batch_size=batch_size)


# train model
history = model.fit(
	train_generator,
	steps_per_epoch=steps_per_epoch,
	epochs=int(yield_augmented*train_len/steps_per_epoch/batch_size),
	validation_data=validation_generator,
	validation_steps=10,
	callbacks=[skmetrics]
)

print('\nhistory dict:', history.history.keys())
l = skmetrics.frac_correct
print(skmetrics.frac_correct)

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(np.arange(len(loss)), loss)
plt.plot(np.arange(len(val_loss)), val_loss)
plt.plot(np.arange(len(l)), l)
plt.show()
