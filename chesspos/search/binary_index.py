import math

import faiss
import numpy as np
import h5py
import chess

from chesspos.utils.utils import correct_file_ending
from chesspos.utils.board_bitboard_converter import bitboard_to_board, board_to_bitboard

def init_binary_index(dim, threads=4):
	# set threads
	faiss.omp_set_num_threads(threads)
	# build index
	return faiss.IndexBinaryFlat(dim)

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

def uint8_to_bitboard(uint8_array, trim_last_bits=3):
	unpacked = np.unpackbits(uint8_array, axis=-1, count=-int(trim_last_bits))
	return np.asarray(unpacked, dtype=bool)


def index_add_bitboards(bb_array, faiss_index):
	uint = bitboard_to_uint8(bb_array)
	faiss_index.add(uint)
	print(f"{faiss_index.ntotal / 1.e6} million positions stored", end="\r")
	return faiss_index

def index_load_bitboard_file(file, id_string, faiss_index, chunks=int(1e6)):
	fname = correct_file_ending(file, "h5")
	chunks = int(chunks)
	table_id = []

	with h5py.File(fname, 'r') as hf:
		print(f"File {fname} has keys {hf.keys()}")
		for key in hf.keys():
			if id_string in key:
				hf_len = len(hf[key])

				for i in range(math.floor(hf_len/chunks)):
					faiss_index = index_add_bitboards(hf[key][i*chunks:(i+1)*chunks,:], faiss_index)

				faiss_index = index_add_bitboards(hf[key][math.floor(hf_len/chunks)*chunks:,:], faiss_index)
				table_id.append(faiss_index.ntotal)

	return faiss_index, table_id

def index_load_bitboard_file_array(file_list, id_string, faiss_index, chunks=int(1e6)):
	file_ids = []
	for file in file_list:
		faiss_index, t_id = index_load_bitboard_file(file, id_string, faiss_index, chunks=chunks)
		file_ids.append(t_id)
	return faiss_index, file_ids

def index_search_and_retrieve(queries, faiss_index, num_results=10):
	D, I = faiss_index.search(queries, k=num_results)
	results = []
	for q_nr in range(len(queries)): # loop over queries
		q_res = []
		for res_nr in range(num_results): # loop over results for that query
			R = faiss_index.reconstruct(int(I[q_nr,res_nr]))
			q_res.append(R)
		results.append(q_res)
	return D, I, results

def index_query_positions(query_array, faiss_index, input_format='fen',
	output_format='fen', num_results=10):
	"""
	Query the faiss index of stored bitboards and retrieve nearest neighbors for
	each provided position.

	:param input_format: format that the query is provided in valid choices 'fen' | 'bitboard'
	:param output_format: format that the results are provided in valid choices 'fen' | 'bitboard' | 'board'
	"""
	#prepare input
	query = []
	if input_format in ['fen','bitboard']:
		if input_format == 'fen':
			for fen in query_array:
				tmp = chess.Board(fen) # fen -> chess.Board
				tmp = board_to_bitboard(tmp) # chess.Board -> bitboard
				query.append(tmp)
		query = np.asarray(query)
		query = bitboard_to_uint8(query)
	else:
		raise ValueError("Invalid input format provided.")

	#reshape if only single query
	if len(query.shape) == 1:
		query = query.reshape((1,-1))
	# search faiss index and retrieve closest bitboards
	distance, _, results = index_search_and_retrieve(query, faiss_index, num_results=num_results)

	# prepare output
	if output_format in ['fen','bitboard','board']:
		for q_nr in range(len(results)): # loop over queries
			for res_nr in range(num_results): # loop over resluts per query
				results[q_nr][res_nr] = uint8_to_bitboard(results[q_nr][res_nr], trim_last_bits=3)
				if output_format in ['fen','board']:
					results[q_nr][res_nr] = bitboard_to_board(results[q_nr][res_nr])
					if output_format == 'fen':
						results[q_nr][res_nr] = results[q_nr][res_nr].fen()
	else:
		raise ValueError("Invalid input format provided.")

	return distance, results

def index_save(faiss_index, save_name, is_binary):
	save = correct_file_ending(save_name, 'faiss')

	if is_binary:
		faiss.write_index_binary(faiss_index, save)
	else:
		faiss.write_index(faiss_index, save)

	return f"Index saved to file {save}"

def index_load(load_name, is_binary):
	load = correct_file_ending(load_name, 'faiss')
	faiss_index = None
	if is_binary:
		faiss_index = faiss.read_index_binary(load)
	else:
		faiss_index = faiss.read_index(load)

	return faiss_index
