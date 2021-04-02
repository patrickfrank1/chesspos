import math

import h5py
import numpy as np
import chess
import tensorflow as tf

from chesspos.utils.utils import correct_file_ending
from chesspos.utils.board_bitboard_converter import board_to_bitboard

def index_load_file(file, id_string, faiss_index, chunks=int(1e5)):
	fname = correct_file_ending(file, "h5")
	chunks = int(chunks)
	table_dict = {}

	with h5py.File(fname, 'r') as hf:
		print(f"File {fname} has keys {hf.keys()}")
		for key in hf.keys():
			if id_string in key:
				hf_len = len(hf[key])
				# TODO: rewrite to pass function to do stuff
				for i in range(math.floor(hf_len/chunks)):
					faiss_index = index_add_embeddings(hf[key][i*chunks:(i+1)*chunks,:], faiss_index)

				faiss_index = index_add_embeddings(hf[key][math.floor(hf_len/chunks)*chunks:,:], faiss_index)
				# add info for reconstruction
				table_dict[faiss_index.ntotal] = [fname, key]

	return faiss_index, table_dict

def index_load_file_array(file_list, id_string, faiss_index, chunks=int(1e5)):
	table_dict = {}
	for f in file_list:
		faiss_index, t_id = index_load_file(f, id_string, faiss_index, chunks=chunks)
		table_dict = {**table_dict, **t_id}
	return faiss_index, table_dict

def index_add_embeddings(embedding_array, faiss_index):
	faiss_index.add(np.asarray(embedding_array, dtype=np.float32))
	print(f"{faiss_index.ntotal / 1.e6} million positions stored", end="\r")
	return faiss_index

def index_train_embeddings(file_list, id_string, faiss_index, train_frac=1e-3, chunks=int(1e4)):
	chunks = int(chunks)
	train_samples = int(chunks*train_frac)
	if train_samples < 1:
		raise ValueError("Not enought training samples. Increase train_frac.")

	# get embedding dimension
	fname = correct_file_ending(file_list[0], "h5")
	dim = None
	with h5py.File(fname, 'r') as hf:
		for key in hf.keys():
			if id_string in key:
				dim = len(hf[key][0])
				break
	train_set = np.empty((0, dim))

	for file in file_list:
		fname = correct_file_ending(file, "h5")
		# get training vectors train_frac controls how many
		with h5py.File(fname, 'r') as hf:
			print(f"File {fname} has keys {hf.keys()}")
			for key in hf.keys():
				if id_string in key:
					hf_len = len(hf[key])
					for i in range(math.floor(hf_len/chunks)):
						train_set = np.concatenate((train_set,hf[key][i*chunks:i*chunks+train_samples,:]))
					train_set = np.concatenate((train_set,hf[key][math.floor(hf_len/chunks)*chunks:math.floor(hf_len/chunks)*chunks+train_samples,:]))

	print(f"Training on {len(train_set)} positions")
	faiss_index.train(np.asarray(train_set, dtype=np.float32))
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

def index_query_positions(query_array, faiss_index, encoder_path,
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
