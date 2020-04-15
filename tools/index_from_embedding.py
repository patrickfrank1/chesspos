import os
import math

import h5py
import faiss
import numpy as np
import chess
import tensorflow as tf

from chesspos.binary_index import board_to_bitboard
from chesspos.utils import correct_file_ending, files_from_directory

def index_load_file(file, id_string, faiss_index, chunks=int(1e5), train=False):
	fname = correct_file_ending(file, "h5")
	chunks = int(chunks)
	table_id = []

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
					faiss_index = index_train_embeddings(hf[key][i*chunks:(i+1)*chunks,:], faiss_index)
				else:
					faiss_index = index_add_embeddings(hf[key][i*chunks:(i+1)*chunks,:], faiss_index)
				table_id.append(faiss_index.ntotal)

	return faiss_index, table_id

def index_load_file_array(file_list, id_string, faiss_index, chunks=int(1e5), train=False):
	file_ids = []
	for file in file_list:
		faiss_index, t_id = index_load_file(file, id_string, faiss_index, chunks=chunks, train=train)
		file_ids.append(t_id)
	return faiss_index, file_ids

def index_add_embeddings(embedding_array, faiss_index):
	faiss_index.add(np.asarray(embedding_array, dtype=np.float32))
	print(f"{faiss_index.ntotal / 1.e6} million positions stored", end="\r")
	return faiss_index

def index_train_embeddings(embedding_array, faiss_index):
	faiss_index.train(np.asarray(embedding_array, dtype=np.float32))
	print(f"faiss index trained? {faiss_index.is_trained}")
	return faiss_index

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

def encode_bitboard(query, model_path):
	model = tf.keras.models.load_model(model_path)
	print("MODEL LOADED")
	model.summary()
	print(query[0].shape, query[0].dtype)
	embedding = model.predict_on_batch(query)
	print(embedding[0].shape, embedding[0].dtype)
	return np.asarray(embedding, dtype=np.float32)

def index_query_positions(query_array, faiss_index, model_path, input_format='fen',
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
		query = encode_bitboard(query, model_path)
	else:
		raise ValueError("Invalid input format provided.")

	#reshape if only single query
	if len(query.shape) == 1:
		query = query.reshape((1,-1))
	# search faiss index and retrieve closest bitboards
	distance, index, results = index_search_and_retrieve(query, faiss_index, num_results=num_results)

	# prepare output
	# if output_format in ['fen','bitboard','board']:
	# 	for q_nr in range(len(results)): # loop over queries
	# 		for res_nr in range(num_results): # loop over resluts per query
	# 			results[q_nr][res_nr] = uint8_to_bitboard(results[q_nr][res_nr], trim_last_bits=3)
	# 			if output_format in ['fen','board']:
	# 				results[q_nr][res_nr] = bitboard_to_board(results[q_nr][res_nr])
	# 				if output_format == 'fen':
	# 					results[q_nr][res_nr] = results[q_nr][res_nr].fen()
	# else:
	# 	raise ValueError("Invalid input format provided.")

	return distance, index, results

if __name__ == "__main__":

	embedding_path = "/media/pafrank/Backup/other/Chess/lichess/embeddings/bitboards16"
	train_path = "/media/pafrank/Backup/other/Chess/lichess/embeddings/train"
	model_path = "/home/pafrank/Documents/coding/chess-position-embedding/metric_learning/new_model7.h5"
	save_path = "/media/pafrank/Backup/other/Chess/lichess/embeddings/quantized_index.faiss"
	table_id = "test_embedding"
	queries = [
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"8/1R6/4p1k1/1p6/p1b2K2/P1Br4/1P6/8 b - - 8 49"
	]

	# #embedding dimension -> 8
	# with h5py.File(f"{train_path}/lichess_db_standard_rated_2013-01-bb.h5", 'r') as hf:
	# 	print(f"Keys {hf.keys()}")
	# 	sample = hf[list(hf.keys())[4]][0]
	# 	print(sample.shape)

	# # create index
	# print("create index")
	# index = faiss.index_factory(8, "SQ4")

	# # add table to faiss index
	# print("train index")
	# train_file_list = files_from_directory(train_path, file_type="h5")
	# index, ids = index_load_file_array(train_file_list, table_id, index, chunks=int(1e5), train=True)

	# # add table to faiss index
	# print("populate index")
	# file_list = files_from_directory(embedding_path, file_type="h5")
	# index, ids = index_load_file_array(file_list, table_id, index, chunks=int(1e5), train=False)

	# # save index
	# faiss.write_index(index, save_path)
	# del index
	# #WORKS!

	index = faiss.read_index(save_path)
	# search index
	print("search index")
	dist, idx, embedding = index_query_positions(queries, index, model_path, input_format='fen',
	output_format='fen', num_results=5)
	print("RESULTS")
	print(dist)
	print(idx)
	print(embedding)