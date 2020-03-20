#!/usr/bin/env python3

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import chess
import pgn2pos


start_board = chess.Board(chess.STARTING_FEN)
start_bb = np.load("test/startpos.npy")


def test_correct_file_ending():
	assert pgn2pos.correct_file_ending("data/hello", "txt") == "data/hello.txt"
	assert pgn2pos.correct_file_ending("one_file.pyc", "pyc") == "one_file.pyc"

def test_board_to_bb():
	board_result = pgn2pos.board_to_bb(start_board)
	assert board_result.shape == (773,)
	assert board_result.dtype == 'bool'
	assert np.all(board_result == start_bb)
