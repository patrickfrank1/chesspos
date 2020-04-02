import math

import faiss
import numpy as np
import h5py
import chess

from pgn2pos import correct_file_ending, board_to_bb

def init_binary_index(dim, threads=4):
	# set threads
	faiss.omp_set_num_threads(threads)

	# build index
	return faiss.IndexBinaryFlat(dim)

# def bb_convert_bool_uint8(bb_array):
# 	bb_len = None
# 	vec_len = None
# 	if len(bb_array.shape) == 1:
# 		bb_len = 1
# 		vec_len = bb_array.shape[0]
# 	else:
# 		bb_len, vec_len = bb_array.shape
# 	uint = np.copy(bb_array).reshape((bb_len, int(vec_len/8), 8)).astype(bool)
# 	return np.reshape(np.packbits(uint, axis=-1), (bb_len, int(vec_len/8)))

def bitboard_to_uint8(bb_array):
	bb_arr = np.asarray(bb_array, dtype=bool)
	if len(bb_arr.shape) == 1: #reshape if single vector provided
		bb_arr = bb_arr.reshape((1,bb_arr.shape[0]))

	arr_len, vec_len = bb_arr.shape

	if vec_len % 8 != 0: # bitboard padding
		bb_arr = np.hstack(( bb_arr, np.zeros((arr_len,8-vec_len%8), dtype=bool) ))

	uint = bb_arr.reshape((arr_len, 1+int(vec_len/8), 8)).astype(bool)

	if arr_len == 1:
		uint = np.reshape(np.packbits(uint, axis=-1), (1+int(vec_len/8,)))
	else:
		uint = np.reshape(np.packbits(uint, axis=-1), (arr_len, 1+int(vec_len/8)))
	return uint

def index_add_bitboards(bb_array, faiss_index):
	uint = bitboard_to_uint8(bb_array)
	faiss_index.add(uint)
	print(f"{faiss_index.ntotal / 1.e6} million positions stored", end="\r")
	return faiss_index

def index_load_bitboard_file(file, id_string, faiss_index, chunks=int(1e6)):
	fname = correct_file_ending(file, "h5")
	chunks = int(chunks)

	with h5py.File(fname, 'r') as hf:
		print(f"File {fname} has keys {hf.keys()}")
		for key in hf.keys():
			if id_string in key:
				hf_len = len(hf[key])

				for i in range(math.floor(hf_len/chunks)):
					faiss_index = index_add_bitboards(hf[key][i*chunks:(i+1)*chunks,:], faiss_index)
				rest_len = hf_len % chunks
				
				faiss_index = index_add_bitboards(hf[key][math.floor(hf_len/chunks)*chunks:,:], faiss_index)

	return faiss_index

def index_load_bitboard_file_array(file_list, id_string, faiss_index, chunks=int(1e6)):
	for file in file_list:
		faiss_index = index_load_bitboard_file(file, id_string, faiss_index, chunks=chunks)
	return faiss_index

# def index_search_bitboard(query_array, faiss_index, num_results=10):
# 	D, I = faiss_index.search(query_array, k=num_results)
# 	return D, I

def bitboard_to_board(bb):
	# set up empty board
	reconstructed = chess.Board()
	reconstructed.clear()
	# loop over all pieces and squares
	for color in [1, 0]: # white, black
		for i in range(1, 7): # P N B R Q K
			idx = (1-color)*6+i-1
			piece = chess.Piece(i,color)

			bitmask = bb[idx*64:(idx+1)*64]
			squares = np.argwhere(bitmask)
			squares = [square for sublist in squares for square in sublist] # flatten list of lists

			for square in squares:
				reconstructed.set_piece_at(square,piece)
	# set global board information
	reconstructed.turn = bb[768]

	castling_rights = ''
	if bb[770]: # castling_h1
		castling_rights += 'K'
	if bb[769]: # castling_a1
		castling_rights += 'Q'
	if bb[772]: # castling_h8
		castling_rights += 'k'
	if bb[771]: # castling_a8
		castling_rights += 'q'
	reconstructed.set_castling_fen(castling_rights)

	return reconstructed

def index_query_positions(query_array, faiss_index, input_format='fen',
	output_format='fen', num_results=10):
	"""
	Query the faiss index of stored bitboards and retrieve nearest neighbors for
	each provided position.

	:param input_format: format that the query is provided in valid choices 'fen' | 'bitboard'
	:param output_format: format that the results are provided in valid choices 'fen' | 'bitboard'
	"""
	query = []
	if input_format == 'fen':
		for fen in query_array:
			tmp = chess.Board(fen) # fen -> chess.Board
			tmp = board_to_bb(tmp) # chess.Board -> bitboard
			query.append(tmp)
		query = np.asarray(query)
		query = bitboard_to_uint8(query)
		print("\n",query.shape)

	elif input_format == 'bitboard':
		print("hello")
	else:
		raise ValueError("Invalid input format provided.")

	D, I = faiss_index.search(query, k=num_results)

	return D, I

if __name__ == "__main__":
	# test querying
	print("\nTesting index")
	index = init_binary_index(776)
	index = index_load_bitboard_file_array(
		["data/bitboards/lichess_db_standard_rated_2013-01-bb"],
		"position_1",
		index
	)
	test_queries = [
		"rnb1kb1r/pp2pppp/2p2n2/3qN3/2pP4/6P1/PP2PP1P/RNBQKB1R w KQkq - 2 6",
		"r2qkb1r/ppp2pp1/2np1n1p/4p3/2B1P1b1/2NPBN2/PPP2PPP/R2Q1RK1 b kq - 3 7"
	]
	D, I = index_query_positions(test_queries, index, input_format='fen',
	output_format='fen', num_results=10)
	print(D, I)

	

	# with h5py.File("data/bitboards/lichess_db_standard_rated_2013-01-bb.h5", 'r') as hf:
	# 	for id in idx[0]:
	# 		near_board = hf["position_1"][id]
	# 		print(near_board)
