import yaml
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras

from chesspos.preprocessing import SampleGenerator
from chesspos.models import DenseAutoencoder, CnnAutoencoder

data_params = yaml.safe_load(open(Path(__file__).with_name('params.yaml')))['train']['data']
model_params = yaml.safe_load(open(Path(__file__).with_name('params.yaml')))['train']['model']


train_generator = SampleGenerator(
    sample_dir = data_params['train_dir'],
    batch_size = data_params['train_batch_size']
)
train_generator.set_subsampling_functions(['singlets'])
train_generator.construct_generator()

test_generator = SampleGenerator(
    sample_dir = data_params['test_dir'],
    batch_size = data_params['test_batch_size']
)
test_generator.set_subsampling_functions(['singlets'])
test_generator.construct_generator()

autoencoder = CnnAutoencoder(
    input_size = model_params['input_size'],
    embedding_size = model_params['embedding_size'],
    hidden_layers = model_params['hidden_layers'],
    loss = model_params['loss'],
    train_generator = train_generator,
    test_generator = test_generator,
	save_dir = model_params['save_dir'],
    train_steps_per_epoch = model_params['train_steps_per_epoch'],
    test_steps_per_epoch = model_params['test_steps_per_epoch'],
    tf_callbacks = model_params['tf_callbacks'] + [
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, verbose=0, mode='min', restore_best_weights=True),
        keras.callbacks.TensorBoard(log_dir=model_params['save_dir'], histogram_freq=1, write_images=True, embeddings_freq=1)
    ]
)
autoencoder.build_model()
autoencoder.compile()
history = autoencoder.train()
autoencoder.save()
