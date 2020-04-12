import math

import h5py
import numpy as np
import tensorflow as tf

from chesspos.utils import correct_file_ending, files_from_directory

def bitboard_from_table(file, table):
	fname = correct_file_ending(file, 'h5')
	with h5py.File(fname, 'r') as hf:
		return np.asarray(hf[table][:])

def bitboard_from_table_generator(file, table, batch_size=128):
	fname = correct_file_ending(file, 'h5')
	with h5py.File(fname, 'r') as hf:
		table_len = len(hf[table])
		iterations = math.floor(table_len/batch_size)
		for i in range(iterations):
			yield np.asarray(hf[table][i*batch_size:(i+1)*batch_size])
		yield np.asarray(hf[table][iterations*batch_size:])

def infer_embeddings(model, inputs, batch_size=128):
	embeddings = model.predict(inputs, batch_size=batch_size)
	return embeddings

def embedding_generator(model, input_generator):
	#embedding_dim = model.get_output_shape_at(-1)[1]
	#embeddings = np.empty(shape=(0,embedding_dim), dtype=np.float32)
	i = 0
	for batch in input_generator:
		i += 1
		print(f"Embedding batch {i}...",end="\r")
		new_embeddings = np.asarray(model.predict_on_batch(batch), dtype=np.float32)
		yield new_embeddings
	# 	embeddings = np.concatenate((embeddings, new_embeddings))
	# return embeddings

def save_embeddings(file, table, data_generator, data_shape, batch_size):
	fname = correct_file_ending(file, 'h5')

	with h5py.File(fname, "a") as hf:
		if table in hf.keys():
			del hf[table]

		data_sheet = hf.create_dataset(table, shape=data_shape,
			dtype=float,  compression="gzip", compression_opts=9
		)
		batch_nr = 0
		to_save = np.empty((0,data_shape[1]), dtype=float)
		for batch in data_generator:
			to_save = np.concatenate((to_save, batch))
			if len(to_save) >= batch_size:
				data_sheet[batch_nr*batch_size:(batch_nr+1)*batch_size,:] = to_save
				to_save = np.empty((0,data_shape[1]), dtype=float)
				batch_nr += 1
		# save rest
		data_sheet[batch_nr*batch_size:,:] = to_save

def get_table_info_from_h5(file, table_prefix):
	fname = correct_file_ending(file, "h5")

	with h5py.File(fname, 'r') as hf:
		keys = [key for key in hf.keys() if table_prefix in key]
		table_len = [len(hf[key]) for key in keys]
		return keys, table_len

def embed_bitboards_from_files():
	model_path = "metric_learning/test_model/model_encoder"
	file_path = "data/bitboards/testdir"
	table_prefix = "position"
	embedding_table_prefix = "test_embedding"
	batch_size_inference=8192
	batch_size_save=8192
	file_arr = files_from_directory(file_path, file_type="h5")
	model = tf.keras.models.load_model(model_path)
	embedding_dim = model.get_output_shape_at(-1)[1]

	for file in file_arr:
		keys, table_len = get_table_info_from_h5(file, table_prefix)
		print(keys, table_len)
		for (i, key) in enumerate(keys):
			table_index = int(key.split("_")[-1])
			print(table_index)
			bb_generator = bitboard_from_table_generator(file, key, batch_size=batch_size_inference)
			embedding_gen = embedding_generator(model, bb_generator)
			save_embeddings(
				file=file,
				table=f"{embedding_table_prefix}_{table_index}",
				data_generator=embedding_gen,
				data_shape=(table_len[i],embedding_dim),
				batch_size=batch_size_save
			)
		with h5py.File(file, 'r') as hf:
			print(hf.keys())


if __name__ == "__main__":

	# encoder = tf.keras.models.load_model('metric_learning/test_model/model_encoder')
	# encoder.summary() # works

	# for key, bb in retireve_bitboard_tables(
	# 	"data/bitboards/lichess_db_standard_rated_2013-01-bb.h5",
	# 	"position"):
	# 	print(key)
	# 	print(len(bb))

	embed_bitboards_from_files()
