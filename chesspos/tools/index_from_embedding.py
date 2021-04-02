from os.path import abspath
import argparse
import json

import h5py
import faiss

from chesspos.utils import files_from_directory, correct_file_ending
import chesspos.embedding_index as iemb

def index_from_embedding(index_factory_string, embedding_dir, table_prefix="test_embedding",
	save_path="", chunks=int(1e4), train_frac=1e-3):

	# get list of embedding files
	embedding_dir = abspath(embedding_dir)
	save_path = abspath(save_path)
	embedding_list = files_from_directory(embedding_dir, file_type="h5")

	# infer embedding dimension from provided embeddings
	embedding_dim = None
	fname = correct_file_ending(embedding_list[0], "h5")
	with h5py.File(fname, 'r') as hf:
		for key in hf.keys():
			if table_prefix in key:
				embedding_dim = len(hf[key][0])
				break

	# create index
	index = faiss.index_factory(embedding_dim, index_factory_string)

	# train faiss index
	index = iemb.index_train_embeddings(embedding_list, table_prefix, index,
		train_frac=train_frac, chunks=chunks
	)

	# populate faiss index
	index, table_dict = iemb.index_load_file_array(embedding_list, table_prefix,
		index, chunks=chunks
	)

	# save index
	faiss.write_index(index, f"{save_path}/{index_factory_string}.faiss")
	json.dump( table_dict, open( f"{save_path}/{index_factory_string}.json", 'w' ) )

	return "\nDone."


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Generate a searchable faiss index from float embeddings.')

	parser.add_argument('index_factory_string', type=str, action="store",
		help='use a valid string to initialize the faiss index factory'
	)
	parser.add_argument('embedding_dir', type=str, action="store",
		help='path to directory that contains all embeddings in h5 files, which should be added to the index'
	)
	parser.add_argument('--table_prefix', type=str, default="test_embedding",
	help='the prefix that selects all h5 tables that store embeddings (default="test_embedding")'
	)
	parser.add_argument('--save_path', type=str, action='store', default="",
		help="full path- and filename to the output index, defaults to the input directory"
	)
	parser.add_argument('--chunks', type=int, default=int(1e4),
		help='chunksize in which files are read, decrease if you run out of RAM'
	)
	parser.add_argument('--train_frac', type=float, default=1e-3,
		help='number of train samples as fraction of the total number of samples (default=1e-3)'
	)

	args = parser.parse_args()

	# print inputs for the user
	print(f"Index factory string: {args.index_factory_string}")
	print(f"Input directory: {args.embedding_dir}")
	print(f"Table key for h5 tables: {args.table_prefix}")
	print(f"Save path: {args.save_path}")
	print(f"Chunks: {args.chunks}")
	print(f"Training fraction: {args.train_frac}\n\n")

	save = ""
	if args.save_path =="":
		save = args.embedding_dir
	else:
		save = args.save_path

	out = index_from_embedding(
		args.index_factory_string,
		args.embedding_dir,
		args.table_prefix,
		save_path=save,
		chunks=args.chunks,
		train_frac=args.train_frac
	)
	print(out)
