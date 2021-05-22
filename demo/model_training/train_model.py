import numpy as np
import tensorflow as tf
from tensorflow import keras

from chesspos.utils import bitboard_to_board
from chesspos.preprocessing import SampleGenerator
from chesspos.models import DenseAutoencoder, CnnAutoencoder

train_generator = SampleGenerator(
    sample_dir = "/home/pafrank/Documents/coding/chess-position-embedding/data/bitboards/train_small/",
    batch_size = 32
)
train_generator.set_subsampling_functions(['singlets'])
train_generator.construct_generator()

test_generator = SampleGenerator(
    sample_dir = "/home/pafrank/Documents/coding/chess-position-embedding/data/bitboards/validation_small/",
    batch_size = 32
)
test_generator.set_subsampling_functions(['singlets'])
test_generator.construct_generator()

autoencoder = CnnAutoencoder(
    input_size = 773,
    embedding_size = 32,
    # hidden_layers = [1025, 0.3, 1024, 0.3, 512, 0.3, 512, 0.3, 256, 0.3, 256, 0.3, 256, 0.3, 128, 0.3, 128, 0.3, 128, 0.3, 128, 0.3],
    hidden_layers = [512, 0.3, 256, 0.3, 128, 0.3],
    loss = 'binary_crossentropy',
    train_generator = train_generator,
    test_generator = test_generator,
	safe_dir = "/home/pafrank/Documents/coding/chess-position-embedding/data/models/new_autoencoder_test/",
    train_steps_per_epoch = 100,
    test_steps_per_epoch = 20,
    tf_callbacks = [
        'checkpoints',
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, verbose=0, mode='min', restore_best_weights=True),
        keras.callbacks.TensorBoard(log_dir="/home/pafrank/Documents/coding/chess-position-embedding/data/models/new_autoencoder_test/", histogram_freq=1, write_images=True, embeddings_freq=1)
    ]
)

autoencoder.build_model()
autoencoder.compile()
history = autoencoder.train()
autoencoder.save()

examples = 10
best_samples = autoencoder.get_best_samples(examples, 10)
for i in range(examples):
    best_sample_bitboard = best_samples[i]['position'].reshape((773, -1))
    best_sample_position = bitboard_to_board(best_sample_bitboard)
    print(best_samples[i]['loss'])
    autoencoder.compare_sample_to_prediction(best_sample_bitboard)

worst_samples = autoencoder.get_worst_samples(examples, 10)
for i in range(examples):
    worst_sample_bitboard = worst_samples[i]['position'].reshape((773, -1))
    worst_sample_position = bitboard_to_board(worst_sample_bitboard)
    print(worst_samples[i]['loss'])
    autoencoder.compare_sample_to_prediction(worst_sample_bitboard)