import yaml
import json
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras

from chesspos.utils.board_bitboard_converter import bitboard_to_board
from chesspos.preprocessing import SampleGenerator
from chesspos.models import DenseAutoencoder, CnnAutoencoder

evaluate_data_params = yaml.safe_load(open(Path(__file__).with_name('params.yaml')))['evaluate']['data']
evaluate_eval_params = yaml.safe_load(open(Path(__file__).with_name('params.yaml')))['evaluate']['eval']

test_generator = SampleGenerator(
    sample_dir = evaluate_data_params['test_dir'],
    batch_size = evaluate_data_params['test_batch_size']
)
test_generator.set_subsampling_functions(['singlets'])
test_generator.construct_generator()

autoencoder = CnnAutoencoder(
    input_size = None,
    embedding_size = None,
    hidden_layers = None,
    loss = None,
    train_generator = None,
    test_generator = test_generator,
	save_dir = evaluate_data_params['model_dir'],
    train_steps_per_epoch = None,
    test_steps_per_epoch = None,
    tf_callbacks = None
)

autoencoder.load()
examples = evaluate_eval_params['number_examples']
examples_out = ""

best_samples = autoencoder.get_best_samples(examples, 10*examples)
for i in range(examples):
    best_sample_bitboard = best_samples[i]['position'].reshape((773, -1))
    best_sample_position = bitboard_to_board(best_sample_bitboard)
    examples_out += str(best_samples[i]['loss']) + '\n\n'
    examples_out += autoencoder.compare_sample_to_prediction(best_sample_bitboard)

worst_samples = autoencoder.get_worst_samples(examples, 10)
for i in range(examples):
    worst_sample_bitboard = worst_samples[i]['position'].reshape((773, -1))
    worst_sample_position = bitboard_to_board(worst_sample_bitboard)
    examples_out += str(worst_samples[i]['loss']) + '\n'
    examples_out += autoencoder.compare_sample_to_prediction(worst_sample_bitboard)

print(examples_out)

with open(Path(__file__).with_name('examples.out'), 'w') as file:
    file.write(examples_out)


with open(Path(__file__).with_name('scores.json'), 'w') as file:
    json.dump(
		{
			'train_loss': autoencoder.train_history['loss'][-1],
			'test_loss': autoencoder.train_history['val_loss'][-1]
		},
		file,
		indent=2
	)

with open(Path(__file__).with_name('test_loss.json'), 'w') as file:
    json.dump(
		{
			'val_loss': [{
				'epoch': i,
				'train_loss': autoencoder.train_history['loss'][i],
				'val_loss':  autoencoder.train_history['val_loss'][i]
			} for i in range(len(autoencoder.train_history['val_loss']))]
		},
		file,
		indent=2
	)
