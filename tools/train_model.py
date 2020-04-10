import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from chesspos.preprocessing import input_generator, input_length
from chesspos.preprocessing import easy_triplets, hard_triplets
from chesspos.models import triplet_autoencoder

print(f"tf.__version__: {tf.__version__}") # pylint: disable=no-member
print(f"tf.keras.__version__: {tf.keras.__version__}")

'''
Inputs
'''
# environment
model_dir = os.path.abspath('metric_learning/model/simple_triplet')
train_dir = os.path.abspath('data/train_small')
validation_dir = os.path.abspath('data/validation_small')
save_metrics = True
hide_warnings = True
plot_model = True
# model specs
input_size = 773
embedding_size = 32
# training specs
train_batch_size = 16
validation_batch_size = 16
train_steps_per_epoch = 1000
validation_steps_per_epoch = 50

'''
Training environment
'''
if hide_warnings:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # hide warnings during training

'''
Define custom callback to monitor a custom metric on epoch end
'''
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

	def on_train_begin(self, logs={}): # pylint: disable=unused-argument,dangerous-default-value
		self.num_correct = [] # pylint: disable=attribute-defined-outside-init
		self.frac_correct = [] # pylint: disable=attribute-defined-outside-init
		self.diagnostics = [] # pylint: disable=attribute-defined-outside-init

	def on_epoch_end(self, epoch, logs={}): # pylint: disable=unused-argument,dangerous-default-value
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

'''
Initialize triplet network
'''
model = triplet_autoencoder(
	input_size,
	embedding_size,
	alpha=0.2,
	triplet_weight_ratio=20.0,
	hidden_layers=[512,0.2,256,0.2,64]
)

if plot_model:
	keras.utils.plot_model(model, model_dir+'/triplet_network.png', show_shapes=True)

'''
Initialise trainig, and validation data
'''
train_files = [
	os.path.abspath('data/train_small/lichess_db_standard_rated_2013-02-tuples.h5'),
	os.path.abspath('data/train_large/lichess_db_standard_rated_2013-03-tuples.h5'),
	os.path.abspath('data/train_large/lichess_db_standard_rated_2013-04-tuples.h5')
]
validation_files = [
	os.path.abspath('data/validation_small/lichess_db_standard_rated_2020-02-07-tuples-strong.h5')
]

# TODO: print WARNING if too few validation examples
train_len = input_length(train_files, table_id_prefix="tuples")
val_len = input_length(validation_files, table_id_prefix="tuples")
print(f"\n{train_len} training samples.")
print(f"{val_len} validation samples.")

# sampling functions
train_fn = [easy_triplets, hard_triplets]
validation_fn = [hard_triplets]

# generators for train and test data
train_generator = input_generator(
	train_files,
	table_id_prefix="tuples",
	selector_fn=train_fn,
	batch_size=train_batch_size
)
validation_generator = input_generator(
	validation_files,
	table_id_prefix="tuples",
	selector_fn=validation_fn,
	batch_size=validation_batch_size
)
metric_generator = input_generator(
	validation_files,
	table_id_prefix="tuples",
	selector_fn=validation_fn,
	batch_size=validation_batch_size
)

# instantiate callbacks
skmetrics = SkMetrics(metric_generator, batch_size=validation_batch_size, steps_per_callback=10)
early_stopping = keras.callbacks.EarlyStopping(
	monitor='val_loss', min_delta=0.05, patience=10, verbose=0, mode='min'
)
cp = keras.callbacks.ModelCheckpoint(
	filepath=model_dir+"/checkpoints/cp-{epoch:04d}.ckpt",
	save_weights_only=False,
	save_best_only=True,
	mode='min',
	verbose=1
)

'''Train  the model'''
history = model.fit(
	train_generator,
	steps_per_epoch=train_steps_per_epoch,
	epochs=int(len(train_fn)*train_len/train_steps_per_epoch/train_batch_size),
	validation_data=validation_generator,
	validation_steps=validation_steps_per_epoch,
	callbacks=[skmetrics, early_stopping] #, cp]
)

'''
Visualise results
'''
print('history dict:', history.history.keys())

loss = history.history['loss']
val_loss = history.history['val_loss']
triplet_acc = skmetrics.frac_correct

def plot_metrics(train_loss, validation_loss, triplet_accuracy=None):
	fig, ax1 = plt.subplots()

	ax1.set_xlabel('epoch')
	ax1.set_ylabel('loss')
	ax1.plot(np.arange(len(train_loss)), train_loss, 'gx-', label="training loss")
	ax1.plot(np.arange(len(validation_loss)), validation_loss, 'rx-', label="validation loss")
	plt.legend()

	if triplet_accuracy is not None:
		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		color = 'tab:blue'
		ax2.set_ylabel('triplet accuracy', color=color)
		ax2.plot(np.arange(len(triplet_accuracy)), triplet_accuracy, 'bo-', label="triplet_accuracy")
		ax2.tick_params(axis='y', labelcolor=color)
		plt.legend()

	fig.tight_layout()
	plt.savefig(model_dir+"/train_loss.png")
	return 1

plot_metrics(loss, val_loss, triplet_accuracy=triplet_acc)
