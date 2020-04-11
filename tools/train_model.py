import os
import math
import argparse
import json

import tensorflow as tf
from tensorflow import keras

from chesspos.utils import files_from_directory
from chesspos.preprocessing import input_generator, input_length
from chesspos.preprocessing import easy_triplets, semihard_triplets, hard_triplets
from chesspos.models import triplet_autoencoder
from chesspos.monitoring import SkMetrics, save_metrics

def train_embedding(train_dir, validation_dir, save_dir, input_size=773,
	embedding_size=32, alpha=0.2, triplet_weight_ratio=10.0, hidden_layers=None,
	train_batch_size=16, validation_batch_size=16, train_steps_per_epoch=1000,
	validation_steps_per_epoch=100, train_sampling=['easy','semihard','hard'],
	validation_sampling=['easy','semihard','hard'],
	tf_callbacks=['early_stopping','triplet_accuracy', 'checkpoints'],
	save_stats=True, hide_tf_warnings=True):
	print(f"tf.__version__: {tf.__version__}") # pylint: disable=no-member
	print(f"tf.keras.__version__: {tf.keras.__version__}")

	'''
	Inputs: as in function declaration, only paths are converted to absolute paths
	'''
	save_dir = os.path.abspath(save_dir)
	train_dir = os.path.abspath(train_dir)
	validation_dir = os.path.abspath(validation_dir)
	# metrics_save = True
	# hide_warnings = True
	# plot_model = True
	# # model specs
	# input_size = 773
	# embedding_size = 32
	# alpha = 0.2
	# triplet_weight_ratio = 20.0
	# hidden_layers = [512,0.4,256,0.4,64]
	# # training specs
	# train_batch_size = 16
	# validation_batch_size = 16
	# train_steps_per_epoch = 1000
	# validation_steps_per_epoch = 110


	'''
	Training environment
	'''
	if hide_tf_warnings:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # hide warnings during training


	'''
	Initialise trainig, and validation data
	'''
	# get files from directory
	train_files = files_from_directory(train_dir, file_type="h5")
	validation_files = files_from_directory(validation_dir, file_type="h5")
	# sampling functions
	train_fn = []
	for el in train_sampling:
		if el == 'easy':
			train_fn.append(easy_triplets)
		elif el == 'semihard':
			train_fn.append(semihard_triplets)
		elif el == 'hard':
			train_fn.append(hard_triplets)
	validation_fn = []
	for el in validation_sampling:
		if el == 'easy':
			validation_fn.append(easy_triplets)
		elif el == 'semihard':
			validation_fn.append(semihard_triplets)
		elif el == 'hard':
			validation_fn.append(hard_triplets)
	# generators for train and test data
	train_generator = input_generator(train_files, table_id_prefix="tuples",
		selector_fn=train_fn, batch_size=train_batch_size
	)
	validation_generator = input_generator(validation_files, table_id_prefix="tuples",
		selector_fn=validation_fn, batch_size=validation_batch_size
	)
	metric_generator = input_generator(validation_files, table_id_prefix="tuples",
		selector_fn=validation_fn, batch_size=validation_batch_size
	)

	# check if there are enough validation samples
	train_len = input_length(train_files, table_id_prefix="tuples")
	val_len = input_length(validation_files, table_id_prefix="tuples")
	train_epochs = 1.0*train_len*len(train_fn)/train_batch_size/train_steps_per_epoch
	val_epochs = 1.0*val_len*len(validation_fn)/validation_batch_size/validation_steps_per_epoch
	print(f'You have enough training samples for {train_epochs} epochs and enought validation samples for {val_epochs} epochs.')

	if train_epochs > val_epochs:
		raise ValueError("Not enought validation samples provided to start training! Cancelling.")
	else:
		print(f"\nTraining on {1.0*train_len*len(train_fn)/1.e6} million training samples.")
		print(f"Validating on {1.0*math.floor(train_epochs)*validation_batch_size*validation_steps_per_epoch/1.e6} million validation samples.")
		if val_epochs > 10 + train_epochs:
			print("WARNING: your are providing much more validation samples than nessecary. Those could be used for training instead.")


	'''
	Initialise tensorflow callbacks
	'''
	skmetrics = SkMetrics(metric_generator, batch_size=validation_batch_size,
		steps_per_callback=validation_steps_per_epoch
	)
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
		min_delta=0.05, patience=10, verbose=0, mode='min', restore_best_weights=True
	)
	cp = keras.callbacks.ModelCheckpoint(
		filepath=save_dir+"/checkpoints/cp-{epoch:04d}.ckpt",
		save_weights_only=False,
		save_best_only=True,
		mode='min',
		verbose=1
	)
	callbacks = []
	for el in tf_callbacks:
		if el == 'early_stopping':
			callbacks.append(early_stopping)
		elif el == 'triplet_accuracy':
			callbacks.append(skmetrics)
		elif el == 'checkpoints':
			callbacks.append(cp)


	'''
	Initialize model
	'''
	model = triplet_autoencoder(
		input_size,
		embedding_size,
		alpha=alpha,
		triplet_weight_ratio=triplet_weight_ratio,
		hidden_layers=hidden_layers
	)
	if save_stats:
		keras.utils.plot_model(model, save_dir+'/triplet_network.png', show_shapes=True)


	'''
	Train the model
	'''
	history = model.fit(
		train_generator,
		steps_per_epoch=train_steps_per_epoch,
		epochs=math.floor(train_epochs),
		validation_data=validation_generator,
		validation_steps=validation_steps_per_epoch,
		callbacks=callbacks
	)

	'''
	Save the trained model
	'''
	model.save(f"{save_dir}/model", save_format='tf')

	'''
	Visualise and save results
	'''
	if save_stats:
		loss = history.history['loss']
		val_loss = history.history['val_loss']
		triplet_acc = None
		if 'triplet_accuracy' in tf_callbacks:
			triplet_acc = skmetrics.frac_correct
		save_metrics(
			[loss, val_loss, triplet_acc],
			["train loss", "validation loss", "triplet accuracy"],
			save_dir,
			plot=True
		)

		return 0

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Train a chess position embedding with tensorflow.')
	parser.add_argument('config', type=str, action="store", help='json config file to read settings from')
	args = parser.parse_args()

	print(f"JSON config file at: {args.config}")

	with open(args.config) as json_data:
		data = json.load(json_data)
	
	print("The following settings are used for training:")
	print(data)

	train_embedding(**data)
