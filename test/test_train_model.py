import numpy as np
import tensorflow as tf
from tensorflow import keras

from chesspos.preprocessing import singlets

from chesspos.ml.preprocessing.sample_generator import SampleGenerator
from chesspos.ml.models.dense_autoencoder import DenseAutoencoder

train_generator = SampleGenerator(
    sample_dir = "/home/pafrank/Documents/coding/chess-position-embedding/data/train_small/",
    batch_size = 16
)
train_generator.set_subsampling_functions(['singlets'])
train_generator.construct_generator()

test_generator = SampleGenerator(
    sample_dir = "/home/pafrank/Documents/coding/chess-position-embedding/data/validation_small/",
    batch_size = 16
)
test_generator.set_subsampling_functions(['singlets'])
test_generator.construct_generator()

simple_autoencoder = DenseAutoencoder(
    input_size = 773,
    embedding_size = 32,
    hidden_layers = [512, 0.5, 128],
    loss = 'binary_crossentropy',
    train_generator = train_generator,
    test_generator = test_generator,
	safe_dir = "/home/pafrank/Documents/coding/chess-position-embedding/metric_learning/new_autoencoder_test/",
    train_steps_per_epoch = 1000,
    test_steps_per_epoch = 10,
    tf_callbacks = [
        'checkpoints',
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, verbose=0, mode='min', restore_best_weights=True)
    ]
)

simple_autoencoder.build_model()
simple_autoencoder.compile()
history = simple_autoencoder.train()

print(history.)
