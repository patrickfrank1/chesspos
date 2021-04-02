import numpy as np
import chess

from chesspos.utils.board_bitboard_converter import board_to_bitboard

start_board = chess.Board(chess.STARTING_FEN)
start_bb = np.load("test/startpos.npy")

def test_board_to_bitboard():
	board_result = board_to_bitboard(start_board)
	assert board_result.shape == (773,)
	assert board_result.dtype == 'bool'
	assert np.all(board_result == start_bb)
