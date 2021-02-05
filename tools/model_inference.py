import argparse
import math
import os

import h5py
import numpy as np
import tensorflow as tf
from chesspos.utils import correct_file_ending, files_from_directory


def bitboard_from_table_generator(file, table, batch_size):
	fname = correct_file_ending(file, 'h5')
	with h5py.File(fname, 'r') as hf:
		table_len = len(hf[table])
		iterations = math.floor(table_len/batch_size)
		for i in range(iterations):
			yield np.asarray(hf[table][i*batch_size:(i+1)*batch_size])
		yield np.asarray(hf[table][iterations*batch_size:])

def embedding_generator(model, input_generator):
	for batch in input_generator:
		yield np.asarray(model.predict_on_batch(batch))

def save_embeddings(file, table, data_generator, data_shape, batch_size, float16):
	fname = correct_file_ending(file, 'h5')

	with h5py.File(fname, "a") as hf:
		if table in hf.keys():
			del hf[table]

		if float16:
			print("Writing float16...")
			data_sheet = hf.create_dataset(table, shape=data_shape, dtype=np.float16)
		else:
			print("Writing float32...")
			data_sheet = hf.create_dataset(table, shape=data_shape, dtype=np.float32)

		batch_nr = 0
		for batch in data_generator:
			print(f"Embedding batch {batch_nr+1}/{math.ceil(data_shape[0]/batch_size)} written to file...", end="\r")
			if len(batch) < batch_size:
				data_sheet[batch_nr*batch_size:,:] = batch
			else:
				data_sheet[batch_nr*batch_size:(batch_nr+1)*batch_size,:] = batch
			batch_nr += 1

def get_table_info_from_h5(file, table_prefix):
	fname = correct_file_ending(file, "h5")
	with h5py.File(fname, 'r') as hf:
		keys = [key for key in hf.keys() if table_prefix in key]
		table_len = [len(hf[key]) for key in keys]
		return keys, table_len

def embed_bitboards_from_files(model_dir, bitboard_dir, table_prefix='position',
	embedding_table_prefix='test_embedding', batch_size=8192, float16=False):

	model_dir = os.path.abspath(model_dir)
	bitboard_dir = os.path.abspath(bitboard_dir)

	file_arr = files_from_directory(bitboard_dir, file_type="h5")

	model = tf.keras.models.load_model(model_dir)
	
	embedding_dim = model.get_output_shape_at(-1)[1]

	for file in file_arr:
		print(f"\nCalculating embeddings for file {file}")
		keys, table_len = get_table_info_from_h5(file, table_prefix)
		for (i, key) in enumerate(keys):
			table_index = int(key.split("_")[-1])
			print(f"\nEmbedding table {table_index}...")
			bb_generator = bitboard_from_table_generator(file, key, batch_size=batch_size)
			embedding_gen = embedding_generator(model, bb_generator)
			save_embeddings(
				file=file,
				table=f"{embedding_table_prefix}_{table_index}",
				data_generator=embedding_gen,
				data_shape=(table_len[i],embedding_dim),
				batch_size=batch_size,
				float16=float16
			)
		with h5py.File(file, 'r') as hf:
			print(f"\nNow {file} contains the following tables:")
			print(hf.keys())
	return "Program successfully finished."


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description='Infer bitboard embeddings from trained model and save to existing h5py file.'
	)

	parser.add_argument('model_dir', type=str, action="store",
		help='Directory in which embedding model is stored.'
	)
	parser.add_argument('bitboard_dir', type=str, action="store",
		help='Directory in which h5 files with bitboards -to be converted and appended to same file- are stored.'
	)
	parser.add_argument('--table_prefix', type=str, action="store", default='position',
		help='Prefix to select h5py tables in which bitboards are stored.'
	)
	parser.add_argument('--embedding_table_prefix', type=str, action="store", default='test_embedding',
		help='Prefix of the tables to which the inferred embeddings are written.'
	)
	parser.add_argument('--batch_size', type=int, action="store", default=8192,
		help='Chunk size for reading, inferring and writing embeddings.'
	)
	parser.add_argument('--float16', type=bool, action="store", default=False,
		help='Whether flaot embeddings are stored as float32 or float16 values, to save storage space.'
	)

	args = parser.parse_args()

	print(f"Model director: {args.model_dir}")
	print(f"Directory with bitboards: {args.bitboard_dir}")
	print(f"Bitboard table prefix: {args.table_prefix}")
	print(f"Embedding table prefix: {args.embedding_table_prefix}")
	print(f"Batch size: {args.batch_size}")
	print(f"Float16 enabled: {args.float16}\n\n")

	exit_message = embed_bitboards_from_files(
		args.model_dir,
		args.bitboard_dir,
		args.table_prefix,
		args.embedding_table_prefix,
		args.batch_size,
		args.float16
	)

	print(exit_message)
