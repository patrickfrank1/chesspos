import numpy as np
import chess

from chesspos.preprocessing.position_processors import bitboard_to_board, board_to_bitboard
import chesspos.search.binary_index as searchpos

def test_bitboard_to_uint8():
	BITBOARD = np.round(np.random.random_sample((773,))).astype(bool)
	BITBOARDS = np.round(np.random.random_sample((2,773))).astype(bool)

	single = searchpos.bitboard_to_uint8(BITBOARD)
	assert single.dtype == np.uint8
	assert single.shape == (1+int(773/8),)

	multiple = searchpos.bitboard_to_uint8(BITBOARDS)
	assert multiple.dtype == np.uint8
	assert multiple.shape == (2,1+int(773/8))

def test_unit8_to_bitboard():
	BITBOARD = np.round(np.random.random_sample((773,))).astype(bool)
	BITBOARDS = np.round(np.random.random_sample((2,773))).astype(bool)

	single_converted = searchpos.bitboard_to_uint8(BITBOARD)
	multiple_converted = searchpos.bitboard_to_uint8(BITBOARDS)

	bitboard_restored = searchpos.uint8_to_bitboard(single_converted, trim_last_bits=3)
	assert bitboard_restored.dtype == bool
	assert np.all(bitboard_restored == BITBOARD)

	bitboards_restored = searchpos.uint8_to_bitboard(multiple_converted, trim_last_bits=3)
	assert bitboards_restored.dtype == bool
	assert np.all(bitboards_restored == BITBOARDS)

def test_bitboard_to_board():
	FEN = "rnb1kb1r/pp2pppp/2p2n2/3qN3/2pP4/6P1/PP2PP1P/RNBQKB1R w KQkq - 2 6"

	board = chess.Board(FEN)
	bitboard = board_to_bitboard(board)
	assert bitboard.dtype == bool 
	assert bitboard.shape == (773,)

	reconstructed_board = bitboard_to_board(bitboard)
	reconstructed_fen = reconstructed_board.fen()
	assert FEN[:-3] == reconstructed_fen[:-3]
