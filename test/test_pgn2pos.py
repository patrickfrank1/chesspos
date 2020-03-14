#!/usr/bin/env python3

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pgn2pos

def test_correct_file_ending():
	assert pgn2pos.correct_file_ending("data/hello", "txt") == "data/hello.txt"
	assert pgn2pos.correct_file_ending("one_file.pyc", "pyc") == "one_file.pyc"
