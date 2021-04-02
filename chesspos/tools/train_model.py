import argparse
import json
import math
import os
import pickle

import tensorflow as tf
from tensorflow import keras

from chesspos.ml.models.models import autoencoder, triplet_autoencoder
from chesspos.monitoring import SkMetrics, save_metrics
from chesspos.preprocessing import (easy_triplets, hard_triplets,
                                    input_generator, input_length,
                                    semihard_triplets, singlet_factory,
                                    triplet_factory)
from chesspos.utils import files_from_directory

# pylint: disable=dangerous-default-value
def train_triplet_autoencoder(train_dir, validation_dir, save_dir,
	input_size=773, embedding_size=32, alpha=0.2, triplet_weight_ratio=10.0, hidden_layers=[],
	train_batch_size=16, validation_batch_size=16,
	train_steps_per_epoch=1000, validation_steps_per_epoch=100,
	train_sampling=['easy','semihard','hard'],
	validation_sampling=['easy','semihard','hard'],
	tf_callbacks=['early_stopping','triplet_accuracy', 'checkpoints'],
	save_stats=True, hide_tf_warnings=True, **kwargs):

	'''
	Inputs: as in function declaration, only paths are converted to absolute paths
	'''
	save_dir = os.path.abspath(save_dir)
	train_dir = os.path.abspath(train_dir)
	validation_dir = os.path.abspath(validation_dir)


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
	# train sampling functions
	samples = [[] for _ in range(2)]
	sample_args = [train_sampling, validation_sampling]
	for i in range(2):
		for element in sample_args[i]:
			if element == 'easy':
				samples[i].append(easy_triplets)
			elif element == 'semihard':
				samples[i].append(semihard_triplets)
			elif element == 'hard':
				samples[i].append(hard_triplets)
			elif element == 'custom_hard':
				samples[i] = samples[i] + [triplet_factory([1,2,3]), triplet_factory([2,3,4])]
	train_fn, validation_fn = samples
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
	checkpoint = keras.callbacks.ModelCheckpoint(
		filepath=save_dir+"/checkpoints/cp-{epoch:04d}.ckpt",
		save_weights_only=False,
		save_best_only=True,
		mode='min',
		verbose=1
	)
	callbacks = []
	for element in tf_callbacks:
		if element == 'early_stopping':
			callbacks.append(early_stopping)
		elif element == 'triplet_accuracy':
			callbacks.append(skmetrics)
		elif element == 'checkpoints':
			callbacks.append(checkpoint)


	'''
	Initialize model
	'''
	models = triplet_autoencoder(
		input_size,
		embedding_size,
		alpha=alpha,
		triplet_weight_ratio=triplet_weight_ratio,
		hidden_layers=hidden_layers
	)
	model = models['autoencoder']
	# if save_stats:
	# 	keras.utils.plot_model(model, save_dir+'/triplet_network.png', show_shapes=True)


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
	Save the trained model: save in 3 different ways to increase robustness
	'''
	for key in models:
		try:
			models[key].save(f"{save_dir}/model_{key}", save_format="tf")
		except ImportError as err:
			print(f"Error when saving {key} in format tf:")
			print(err)
		try:
			models[key].save(f"{save_dir}/model_{key}.h5")
		except ImportError as err:
			print(f"Error when saving {key} in format h5:")
			print(err)
		try:
			config = models[key].get_config()
			weights = models[key].get_weights()
			pickle.dump(config, open(f"{save_dir}/model_{key}_config.pk", 'wb'))
			pickle.dump(weights, open(f"{save_dir}/model_{key}_weights.pk", 'wb'))
		except ImportError as err:
			print(f"Error when saving weights and architecture of {key}:")
			print(err)

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

def train_autoencoder(train_dir, validation_dir, save_dir,
	input_size=773, embedding_size=32, hidden_layers=[], hidden_decoder=[],
	train_batch_size=16, validation_batch_size=16,
	train_steps_per_epoch=1000, validation_steps_per_epoch=100,
	train_sampling=[0], validation_sampling=[0],
	tf_callbacks=['early_stopping','checkpoints'],
	save_stats=True, hide_tf_warnings=True, **kwargs):

	'''
	Inputs: as in function declaration, only paths are converted to absolute paths
	'''
	save_dir = os.path.abspath(save_dir)
	train_dir = os.path.abspath(train_dir)
	validation_dir = os.path.abspath(validation_dir)


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
	# train sampling functions
	samples = [[] for _ in range(2)]
	sample_args = [train_sampling, validation_sampling]
	for i in range(2):
		for j in sample_args[i]:
			samples[i].append(singlet_factory(j))
	train_fn, validation_fn = samples
	# generators for train and test data
	train_generator = input_generator(train_files, table_id_prefix="tuples",
		selector_fn=train_fn, batch_size=train_batch_size
	)
	validation_generator = input_generator(validation_files, table_id_prefix="tuples",
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
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
		min_delta=0.005, patience=10, verbose=0, mode='min', restore_best_weights=True
	)
	checkpoint = keras.callbacks.ModelCheckpoint(
		filepath=save_dir+"/checkpoints/cp-{epoch:04d}.ckpt",
		save_weights_only=False,
		save_best_only=True,
		mode='min',
		verbose=1
	)
	callbacks = []
	for element in tf_callbacks:
		if element == 'early_stopping':
			callbacks.append(early_stopping)
		elif element == 'checkpoints':
			callbacks.append(checkpoint)


	'''
	Initialize model
	'''
	models = autoencoder(
		input_size,
		embedding_size,
		hidden_layers=hidden_layers,
		hidden_decoder=hidden_decoder
	)
	model = models['autoencoder']
	# if save_stats:
	# 	keras.utils.plot_model(model, save_dir+'/triplet_network.png', show_shapes=True)


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
	Save the trained model: save in 3 different ways to increase robustness
	'''
	for key in models:
		try:
			models[key].save(f"{save_dir}/model_{key}", save_format="tf")
		except ImportError as err:
			print(f"Error when saving {key} in format tf:")
			print(err)
		try:
			models[key].save(f"{save_dir}/model_{key}.h5")
		except ImportError as err:
			print(f"Error when saving {key} in format h5:")
			print(err)
		try:
			config = models[key].get_config()
			weights = models[key].get_weights()
			pickle.dump(config, open(f"{save_dir}/model_{key}_config.pk", 'wb'))
			pickle.dump(weights, open(f"{save_dir}/model_{key}_weights.pk", 'wb'))
		except ImportError as err:
			print(f"Error when saving weights and architecture of {key}:")
			print(err)

	'''
	Visualise and save results
	'''
	if save_stats:
		loss = history.history['loss']
		val_loss = history.history['val_loss']
		save_metrics(
			[loss, val_loss],
			["train loss", "validation loss"],
			save_dir,
			plot=True
		)

	return 0

if __name__ == "__main__":

	print(f"tf.__version__: {tf.__version__}") # pylint: disable=no-member
	print(f"tf.keras.__version__: {tf.keras.__version__}")

	PARSER = argparse.ArgumentParser(description='Train a chess position embedding with tensorflow.')
	PARSER.add_argument('config', type=str, action="store",
		help='json config file to read settings from'
	)
	ARGS = PARSER.parse_args()

	print(f"JSON config file at: {ARGS.config}")

	with open(ARGS.config) as json_data:
		DATA = json.load(json_data)

	print("The following settings are used for training:")
	print(DATA)

	if DATA['model_type'] == 'triplet_autoencoder':
		train_triplet_autoencoder(**DATA)
	elif DATA['model_type'] == 'autoencoder':
		train_autoencoder(**DATA)
	else:
		raise ValueError("Network not implemented.")
