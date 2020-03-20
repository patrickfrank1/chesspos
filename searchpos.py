import math
import faiss
import numpy as np
import h5py
from pgn2pos import correct_file_ending

def init_binary_index(dim, threads=4):
	# set threads
	faiss.omp_set_num_threads(threads)

	# build index
	return faiss.IndexBinaryFlat(dim)

def load_bb(bb_array, faiss_index):
	bb_len, vec_len = bb_array.shape
	uint = np.copy(bb_array).reshape((bb_len, int(vec_len/8), 8))
	uint = np.reshape(np.packbits(uint, axis=-1), (bb_len, int(vec_len/8)))
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

if __name__ == "__main__":

	#index = faiss_load_bb(["data/db/lichess_db_standard_rated_2013-01-bb"])
	index = init_binary_index(776)
	index = load_h5_array(["data/db/lichess_db_standard_rated_2013-01-bb"], "position", index)





# k = 4                          # we want to see 4 nearest neighbors
# D, I = index.search(xb[:5], k) # sanity check
# print(I)
# print(D)
# D, I = index.search(xq, k)     # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])    