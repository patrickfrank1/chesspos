import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class SkMetrics(tf.keras.callbacks.Callback):
	'''
	Custom callback to monitor triplet classification accuracy on epoch end.
	'''
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

def save_metrics(metric_arrays, save_dir, plot=True):
