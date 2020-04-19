import json

import h5py
import faiss

from chesspos.utils import files_from_directory, correct_file_ending
import chesspos.embedding_index as iemb

# inputs
index_factory_string = "PCA16,SQ4"
embedding_dir = "/media/pafrank/Backup/other/Chess/lichess/embeddings/bb_d64_add"
save_path = "/media/pafrank/Backup/other/Chess/lichess/embeddings"
table_prefix = "test_embedding"
queries = [
	"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
	"8/1R6/4p1k1/1p6/p1b2K2/P1Br4/1P6/8 b - - 8 49"
]
num_results = 10
embedding_dim = None
train_frac = 1e-3
chunks = int(1e4)

# get list of embedding files
embedding_list = files_from_directory(embedding_dir, file_type="h5")

# infer embedding dimension from provided embeddings
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