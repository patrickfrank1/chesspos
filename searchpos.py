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

def convert_bb_to_board(bb):
	rec_board = chess.Board()
	rec_board.clear()
	pieces = list(zip(6*["white"],["pawn","knight","bishop","rook","queen","king"]))
	pieces = pieces + list(zip(6*["black"],["pawn","knight","bishop","rook","queen","king"]))
	#print(pieces)
	for color in [1, 0]:
		for i in range(1, 7): # P N B R Q K / white
			index = (1-color)*6+i-1
			print(f"bitmask for pieces: {pieces[index]}")
			print(f"start index: {index*64}")
			print(f"end index: {(index+1)*64}")
			bitmask = bb[index*64:(index+1)*64]
			print(len(bitmask))
			squares = np.argwhere(bitmask)
			squares = [square for sublist in squares for square in sublist]
			print(squares)
			piece = chess.Piece(i,color)
			for square in squares:
				rec_board.set_piece_at(square,piece)
	turn = bb[768]
	castling_a1 = bb[769]
	castling_h1 = bb[770]
	castling_a8 = bb[771]
	castling_h8 = bb[772]
	print(turn,castling_a1,castling_a8,castling_h1,castling_h8)
	print(bb[773:])
	print(rec_board)

if __name__ == "__main__":

	# test loading
	print("\nTesting bitboard conversion")
	fen = "rnb1kb1r/pp2pppp/2p2n2/3qN3/2pP4/6P1/PP2PP1P/RNBQKB1R w KQkq - 2 6"
	board = chess.Board(fen)
	print(board)
	bitboard = board_to_bb(board)
	print(bitboard.shape)
	bitboard = np.concatenate((bitboard, np.zeros((3,),dtype=bool)))
	print(bitboard.shape)

	reconstructed_board = convert_bb_to_board(bitboard)
	print(board)
	

	# test querying
	# print("\nTesting index")
	# index = init_binary_index(776)
	# index = load_h5_array(["data/bitboards/lichess_db_standard_rated_2013-01-bb"], "position_1", index)
	# # search board from previous test
	# search_uint8 = bb_convert_bool_uint8(bitboard)
	# print(search_uint8)
	# dist, idx = search_bb(search_uint8, index)
	# print(dist, idx)

	# with h5py.File("data/bitboards/lichess_db_standard_rated_2013-01-bb.h5", 'r') as hf:
	# 	for id in idx[0]:
	# 		near_board = hf["position_1"][id]
	# 		print(near_board)
	# 		# TODO: convert bitboard to chess.board
