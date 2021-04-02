from functools import cmp_to_key
from colorama import Fore, Style
import chess

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

from chesspos.utils import bitboard_to_board, board_to_bitboard
from chesspos.models.trainable_model import TrainableModel

class ChesspositionInspectableAutoencoderMixin():
	"""
	Requires the extended class to have the following attributes:
	self.model
	self.encoder
	self.decoder
	"""

	def __init__(self):
		super().__init__()

	@staticmethod
	def reshape_bitboards_for_model(bitboards):
		if bitboards.shape[0] == 773:
			bitboards = bitboards.reshape((-1, 773))
		return bitboards

	@staticmethod
	def reshape_bitboards_for_parsing(bitboards):
		if bitboards.shape[1] == 773:
			bitboards = bitboards.reshape((773, -1))
		return bitboards

	@staticmethod
	def binarize_array(array, threshold=0.5):
		return np.where(array > threshold, True, False)

	
	def get_embedding_of_fen(self, fen):
		board = chess.Board(fen)
		bitboard = board_to_bitboard(board)
		bitboard = self.reshape_bitboards_for_model(bitboard)
		return self.encoder.predict(bitboard)

	def get_board_of_embedding(self, embedding):
		reconstructed_bitboard = self.decoder.predict(embedding)
		reconstructed_bitboard = self.reshape_bitboards_for_parsing(reconstructed_bitboard)
		reconstructed_bitboard = self.binarize_array(reconstructed_bitboard)
		board = bitboard_to_board(reconstructed_bitboard)
		return board

	def _get_sorted_samples(self, sort_fn, number_samples, steps=None):
		test_generator = self.test_generator.get_generator()
		stored_samples = []

		for i in range(steps):
			x, y = test_generator.__next__()
			for j in range(x.shape[0]):
				bitboard = self.reshape_bitboards_for_model(x[j])
				sample_loss = self.model.test_on_batch(bitboard, bitboard)
				stored_samples.append({'position':bitboard, 'loss': sample_loss})

			sort_wrapper = lambda x, y: sort_fn(x['loss'], y['loss'])
			stored_samples.sort(key=cmp_to_key(sort_wrapper))
			stored_samples = stored_samples[:number_samples]

		return stored_samples

	def get_best_samples(self, number_samples, steps=None):
		max_loss = lambda x, y : x - y
		return self._get_sorted_samples(max_loss, number_samples, steps)

	def get_worst_samples(self, number_samples, steps=None):
		min_loss = lambda x, y : y - x
		return self._get_sorted_samples(min_loss, number_samples, steps)


	def compare_sample_to_prediction(self, bitboard):
		input_bitboard = self.reshape_bitboards_for_model(bitboard)
		predicted_bitboard = self.model.predict(input_bitboard)
		predicted_bitboard = self.binarize_array(predicted_bitboard)
		predicted_bitboard = self.reshape_bitboards_for_parsing(predicted_bitboard)

		input_board = self.reshape_bitboards_for_parsing(bitboard)
		input_board = bitboard_to_board(input_board)
		input_board = input_board.__str__().split("\n")

		output_board = self.reshape_bitboards_for_parsing(predicted_bitboard)
		output_board = bitboard_to_board(output_board)
		output_board = output_board.__str__().split("\n")
		for i in range(len(output_board)):
			output_board[i] = ''.join([f'{Fore.RED}{output_board[i][j]}{Style.RESET_ALL}' if input_board[i][j] != output_board[i][j] else output_board[i][j] for j in range(len(output_board[i]))])

		output_str = [f"{input_board[i]}    {output_board[i]}\n" for i in range(len(input_board))]
		print("Original:          Reconstructed:")
		print(''.join(output_str))


