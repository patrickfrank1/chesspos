import numpy as np
import chess
import chesspos.convert as conv

start_board = chess.Board(chess.STARTING_FEN)
start_bb = np.load("test/startpos.npy")

def test_board_to_bitboard():
	board_result = conv.board_to_bitboard(start_board)
	assert board_result.shape == (773,)
	assert board_result.dtype == 'bool'
	assert np.all(board_result == start_bb)
