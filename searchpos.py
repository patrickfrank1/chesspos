import math
import faiss
import numpy as np
import h5py
from pgn2pos import correct_file_ending, board_to_bb
import chess

def init_binary_index(dim, threads=4):
	# set threads
	faiss.omp_set_num_threads(threads)

	# build index
	return faiss.IndexBinaryFlat(dim)

def bb_convert_bool_uint8(bb_array):
	bb_len = None
	vec_len = None
	if len(bb_array.shape) == 1:
		bb_len = 1
		vec_len = bb_array.shape[0]
	else:
		bb_len, vec_len = bb_array.shape
	uint = np.copy(bb_array).reshape((bb_len, int(vec_len/8), 8)).astype(bool)
	return np.reshape(np.packbits(uint, axis=-1), (bb_len, int(vec_len/8)))

def load_bb(bb_array, faiss_index):
	uint = bb_convert_bool_uint8(bb_array)
	faiss_index.add(uint)
	print(f"{faiss_index.ntotal / 1.e6} million positions stored", end="\r")
	return faiss_index

def load_h5_bb(file, id_string, faiss_index, chunks=int(1e6)):
	fname = correct_file_ending(file, "h5")
	with h5py.File(fname, 'r') as hf:
		print(f"File {fname} has keys {hf.keys()}")
		for key in hf.keys():
			if id_string in key:
				hf_len = len(hf[key])
				chunks = int(chunks)
				vectors = np.zeros(shape=(chunks, 776), dtype=np.bool_)
				for i in range(math.floor(hf_len/chunks)):
					vectors[:, :773] = hf[key][i*chunks:(i+1)*chunks, :773]
					faiss_index = load_bb(vectors, faiss_index)
				rest_len = hf_len % chunks
				vectors = np.zeros(shape=(rest_len, 776), dtype=np.bool_)
				vectors[:, :773] = hf[key][math.floor(hf_len/chunks)*chunks:, :773]
				faiss_index = load_bb(vectors, faiss_index)
	return faiss_index

def load_h5_array(file_list, id_string, faiss_index, chunks=int(1e6)):
	for file in file_list:
		faiss_index = load_h5_bb(file, id_string, faiss_index, chunks=chunks)
	return faiss_index

def search_bb(query_array, faiss_index, num_results=10):
	D = faiss_index.search(query_array, k=num_results)
	return D

if __name__ == "__main__":

	# test loading
	index = init_binary_index(776)
	index = load_h5_array(["data/db/lichess_db_standard_rated_2013-01-bb"], "position_1", index)

	# test querying
	print("Testing.............")
	board = chess.Board("rnb1kb1r/pp2pppp/2p2n2/3qN3/2pP4/6P1/PP2PP1P/RNBQKB1R w KQkq - 2 6")
	print(board)
	bitboard = board_to_bb(board)
	print(bitboard.shape)
	bitboard = np.concatenate((bitboard, np.zeros((3,),dtype=bool)))
	print(bitboard.shape)
	search_uint8 = bb_convert_bool_uint8(bitboard)
	print(search_uint8)
	dist, idx = search_bb(search_uint8, index)
	print(dist, idx)

	with h5py.File("data/db/lichess_db_standard_rated_2013-01-bb.h5", 'r') as hf:
		for i in idx[0]:
			near_board = hf["position_1"][i]
			print(near_board)
			# TODO: convert bitboard to chess.board
