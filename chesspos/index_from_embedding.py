import os
import math
import json

import h5py
import faiss
import numpy as np
import chess
import tensorflow as tf

from chesspos.convert import bitboard_to_board
from chesspos.binary_index import board_to_bitboard
from chesspos.utils import correct_file_ending, files_from_directory

def index_load_file(file, id_string, faiss_index, chunks=int(1e5), train=False):
	fname = correct_file_ending(file, "h5")
	chunks = int(chunks)
	table_dict = {}

	with h5py.File(fname, 'r') as hf:
		print(f"File {fname} has keys {hf.keys()}")
		for key in hf.keys():
			if id_string in key:
				hf_len = len(hf[key])

				for i in range(math.floor(hf_len/chunks)):
					if train:
						faiss_index = index_train_embeddings(hf[key][i*chunks:(i+1)*chunks,:], faiss_index)
					else:
						faiss_index = index_add_embeddings(hf[key][i*chunks:(i+1)*chunks,:], faiss_index)

				if train:
					faiss_index = index_train_embeddings(hf[key][math.floor(hf_len/chunks)*chunks:,:], faiss_index)
				else:
					faiss_index = index_add_embeddings(hf[key][math.floor(hf_len/chunks)*chunks:,:], faiss_index)
				# add info for reconstruction
				table_dict[faiss_index.ntotal] = [fname, key]

	return faiss_index, table_dict

def index_load_file_array(file_list, id_string, faiss_index, chunks=int(1e5), train=False):
	table_dict = {}
	for f in file_list:
		faiss_index, t_id = index_load_file(f, id_string, faiss_index, chunks=chunks, train=train)
		table_dict = {**table_dict, **t_id}
	return faiss_index, table_dict

def index_add_embeddings(embedding_array, faiss_index):
	faiss_index.add(np.asarray(embedding_array, dtype=np.float32))
	print(f"{faiss_index.ntotal / 1.e6} million positions stored", end="\r")
	return faiss_index

def index_train_embeddings(embedding_array, faiss_index):
	faiss_index.train(np.asarray(embedding_array, dtype=np.float32))
	print(f"faiss index trained? {faiss_index.is_trained}")
	return faiss_index

def index_search_and_retrieve(query_arr, faiss_index, num_results=10):
	D, I = faiss_index.search(query_arr, k=num_results)
	results = []
	for q_nr in range(len(query_arr)): # loop over queries
		q_res = []
		for res_nr in range(num_results): # loop over results for that query
			R = faiss_index.reconstruct(int(I[q_nr,res_nr]))
			q_res.append(R)
		results.append(q_res)
	return D, I, np.asarray(results)

def encode_bitboard(query, model_path):
	model = tf.keras.models.load_model(model_path)
	embedding = model.predict_on_batch(query)
	return np.asarray(embedding, dtype=np.float32)

def index_query_positions(query_array, faiss_index, encoder_path, table_dict,
	input_format='fen', num_results=10):
	"""
	Query the faiss index of stored bitboards and retrieve nearest neighbors for
	each provided position.

	:param input_format: format that the query is provided in valid choices 'fen' | 'bitboard'
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
		query = encode_bitboard(query, encoder_path)
	else:
		raise ValueError("Invalid input format provided.")

	#reshape if only single query
	if len(query.shape) == 1:
		query = query.reshape((1,-1))

	# search faiss index and retrieve closest bitboards
	distance, idx, results = index_search_and_retrieve(query, faiss_index, num_results=num_results)
	return distance, idx, results

def sort_dict_keys(table_dict):
	keys = np.asarray(list(table_dict.keys()), dtype=np.int32)
	print(keys)
	sorted_keys = np.sort(keys)
	return sorted_keys

def location_from_index(id_lists, table_dict):

	sorted_keys = sort_dict_keys(table_dict)
	num_lists = len(id_lists)
	len_lists = len(id_lists[0])

	key = np.zeros_like(id_lists, dtype=np.int64)
	for i in range(num_lists):
		for j in range(len_lists):
			key_idx = np.min(np.argwhere(id_lists[i][j] < sorted_keys))
			key[i][j] = sorted_keys[key_idx]

	offset_from_end = key - id_lists

	name_file = [[None for _ in range(len_lists)] for _ in range(num_lists)]
	name_table = [[None for _ in range(len_lists)] for _ in range(num_lists)]

	for i in range(num_lists):
		for j in range(len_lists):
			name_file[i][j] = table_dict[str(key[i][j])][0]
			name_table[i][j] = table_dict[str(key[i][j])][1]

	return np.asarray(name_file), np.asarray(name_table), np.asarray(offset_from_end)

def manipulate_prefix(identifier, new_prefix):

	identifier = np.asarray(identifier)

	def swap_prefix(table_name, new_prefix):
		num = table_name.split('_')[-1]
		return f"{new_prefix}_{num}"

	swap_all = np.vectorize(swap_prefix)
	new_identifier = swap_all(identifier, new_prefix)
	return new_identifier


def retrieve_elements_from_file(files, tables, offsets):

	files = np.asarray(files)
	tables = np.asarray(tables)
	offsets = np.asarray(offsets)

	dt = None
	d_len = None
	with h5py.File(files[0][0], 'r') as hf:
		dt = hf[tables[0][0]][0].dtype
		d_len = len(hf[tables[0][0]][0])
	def get_el(f, t, o):
		with h5py.File(f, 'r') as hf:
			d_len = len(hf[t])
			return hf[t][d_len-o]

	vf = np.vectorize(get_el, signature=f'(),(),()->({d_len})', otypes=[dt for _ in range(d_len)])
	results = vf(files, tables, offsets)

	return results

if __name__ == "__main__":

	embedding_path = "/media/pafrank/Backup/other/Chess/lichess/embeddings/add"
	train_path = "/media/pafrank/Backup/other/Chess/lichess/embeddings/train"
	model_path = "/home/pafrank/Documents/coding/chess-position-embedding/metric_learning/model7/model_encoder.h5"
	decoder_path ="/home/pafrank/Documents/coding/chess-position-embedding/metric_learning/model7/model_decoder.h5"
	save_path = "/media/pafrank/Backup/other/Chess/lichess/embeddings"
	table_id = "test_embedding"
	queries = [
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"8/1R6/4p1k1/1p6/p1b2K2/P1Br4/1P6/8 b - - 8 49"
	]
	num_results = 10

	# # create index
	# print("create index")
	# index = faiss.index_factory(8, "SQ4")

	# # add table to faiss index
	# print("train index")
	# train_file_list = files_from_directory(train_path, file_type="h5")
	# index, ids = index_load_file_array(train_file_list, table_id, index, chunks=int(1e6), train=True)
	# print(ids)

	# # add table to faiss index
	# print("populate index")
	# file_list = files_from_directory(embedding_path, file_type="h5")
	# index, ids = index_load_file_array(file_list, table_id, index, chunks=int(1e5), train=False)
	# print(ids)

	# # save index
	# faiss.write_index(index, f"{save_path}/SQ4.faiss")
	# del index
	# import json
	# json.dump( ids, open( f"{save_path}/table_dict.json", 'w' ) )

	import json
	# Read data from file:
	table_dict = json.load( open( f"{save_path}/table_dict.json" ) )

	index = faiss.read_index(f"{save_path}/SQ4.faiss")
	# search index
	print("search index")
	D, I, E = index_query_positions(queries, index, model_path, table_dict,
	input_format='fen', num_results=num_results)

	# get location from index
	file, table, offset = location_from_index(I, table_dict)

	print(table)
	bb_table = manipulate_prefix(table, "position")
	print(bb_table)
	bitboards = retrieve_elements_from_file(file, bb_table, offset)
	print(bitboards.shape, bitboards.dtype)

	# # look at the decoders performance
	# embeddings = retrieve_elements_from_file(file, table, offset)
	# decoder = tf.keras.models.load_model(decoder_path)
	# decoder.summary()

	# decoded_pos = decoder(embeddings[1][1].reshape((1,-1)))
	# print(decoded_pos.shape, decoded_pos.dtype)
	# decoded_pos = decoded_pos[0]
	# print("bitboard")
	# print(bitboard_to_board(bitboards[1][1]))
	# print("decoded")
	# print(bitboard_to_board(decoded_pos))

	print("query")
	print(chess.Board(queries[1]))
	for r in range(num_results):
		print(f"Result {r+1}")
		print(bitboard_to_board(bitboards[1][r]))