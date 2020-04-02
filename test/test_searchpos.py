import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import chess
import faiss

import searchpos

BITBOARD = np.round(np.random.random((773,))).astype(bool)
BITBOARDS = np.round(np.random.random((2,773))).astype(bool)

def test_bitboard_to_uint8():
	single = searchpos.bitboard_to_uint8(BITBOARD)
	assert(single.dtype) == np.uint8
	assert(single.shape) == (1+int(773/8),)

	multiple = searchpos.bitboard_to_uint8(BITBOARDS)
	assert(multiple.dtype) == np.uint8
	assert(multiple.shape) == (2,1+int(773/8))
