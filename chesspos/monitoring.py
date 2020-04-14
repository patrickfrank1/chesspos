from os.path import abspath

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
		self.frac_correct = [] # pylint: disable=attribute-defined-outside-init

	def on_epoch_end(self, epoch, logs={}): # pylint: disable=unused-argument,dangerous-default-value
		correct = tf.Variable(0, dtype=tf.int32)
		for i in range(self.steps_per_callback):
			predictions = self.model.predict_on_batch(next(self.valid_data))
			correct.assign_add(self.predict_correct(predictions))
		frac = tf.cast(correct, dtype=tf.float32)/tf.Variable(self.batch_size*self.steps_per_callback, dtype=tf.float32)
		self.frac_correct.append(frac.numpy())
		print(f" triplet_acc: {self.frac_correct[-1]}")

def plot_metrics(save_dir, loss_arr, loss_labels, other_metric=None, other_label=None):
	assert len(loss_arr) == len(loss_labels)

	fig, ax1 = plt.subplots()
	ax1.set_title('Training progress')
	ax1.set_xlabel('epoch')
	ax1.set_ylabel('loss')
	for (i, el) in enumerate(loss_arr):
		ax1.plot(np.arange(len(el)), el, 'x-', label=loss_labels[i])
	plt.legend()

	if other_metric is not None and other_label is not None:
		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		color = 'tab:red'
		ax2.set_ylabel(other_label, color=color)
		ax2.plot(np.arange(len(other_metric)), other_metric, 'ro-', label=other_label)
		ax2.tick_params(axis='y', labelcolor=color)
		plt.legend()

	fig.tight_layout()
	plt.savefig(save_dir+"/train_progress.png")

def save_metrics(metric_arrays, metric_labels, save_dir, plot=True):
	# get absolute path
	path = abspath(save_dir)
	# save metrics to file
	if len(metric_labels) == len(metric_arrays):
		# this is recommended
		d = dict(zip(metric_labels, metric_arrays))
		np.savez_compressed(f"{path}/metrics",**d)
	else:
		raise ValueError("No metric labels provided.")

	if plot:
		*loss, special = metric_arrays
		*label, special_label = metric_labels
		plot_metrics(path, loss, label, other_metric=special, other_label=special_label)
