from typing import Callable
import numpy as np
from chess import Board

import chesspos.custom_types as ct

def no_filter(board: Board) -> bool:
	return True

def subsample_positions(selection_probability: Callable[[int], float]) -> ct.PositionFilter:
	"""Subsample positions based on the selection probability function"""
	def filter(board: Board) -> bool:
		ply = board.ply()
		return selection_probability(ply) > np.random.rand()
	return filter

def subsample_opening_10_linear(board: Board) -> bool:
	selection_probability = lambda x: min(x, 20) / 20.0
	return subsample_positions(board, selection_probability)

def subsample_opening_20_linear(board: Board) -> bool:
	selection_probability = lambda x: min(x, 40) / 40.0
	return subsample_positions(board, selection_probability)

def subsample_opening_10_quadratic(board: Board) -> bool:
	selection_probability = lambda x: min(x, 20)**2 / 20.0
	return subsample_positions(board, selection_probability)

def filter_piece_count(min_pieces: int, max_pieces: int) -> ct.PositionFilter:
	"""Discard positions with less or equal than min_pieces or more than max_pieces"""
	def filter(board: Board) -> bool:
		return min_pieces <= len(board.piece_map()) <= max_pieces
	return filter