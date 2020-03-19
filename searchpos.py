import math
import faiss
import numpy as np
import h5py
from pgn2pos import correct_file_ending

def faiss_load_bb(file_list):
	d = 776
	index = faiss.IndexBinaryFlat(d)

	for file in file_list:
		fname = correct_file_ending(file, "h5")
		with h5py.File(fname, 'r') as hf:
			print(f"File {fname} has keys {hf.keys()}")

			for key in hf.keys():
				if key[:8] == 'position':
					hf_len = len(hf[key])
					chunks = int(1e6)
					vectors = np.zeros(shape=(chunks, 776), dtype=np.bool_)
					for i in range(math.floor(hf_len/chunks)):
						vectors[:, :773] = hf[key][i*chunks:(i+1)*chunks,:773]
						uint_vec = transform_bb_uint8(vectors)
						index.add(uint_vec)
						print(f"{index.ntotal / 1.e6} million positions stored", end="\r")
					rest_len = hf_len % chunks
					vectors = np.zeros(shape=(rest_len, 776), dtype=np.bool_)
					vectors[:, :773] = hf[key][math.floor(hf_len/chunks)*chunks:,:773]
					uint_vec = transform_bb_uint8(vectors)
					index.add(uint_vec)
					print(f"{index.ntotal / 1.e6} million positions stored")
	return index

def transform_bb_uint8(bb_array):
	bb_len, vec_len = bb_array.shape
	uint = np.copy(bb_array).reshape((bb_len, int(vec_len/8), 8))
	uint = np.reshape(np.packbits(uint, axis=-1), (bb_len, int(vec_len/8)))
	return uint

if __name__ == "__main__":
	
	index = faiss_load_bb(["data/db/lichess_db_standard_rated_2013-01-bb",
					"data/db/lichess_db_standard_rated_2013-02-bb"])

	








# index = faiss.IndexFlatL2(d)   # build the index
# print(index.is_trained)
# index.add(xb)                  # add vectors to the index
# print(index.ntotal)

# k = 4                          # we want to see 4 nearest neighbors
# D, I = index.search(xb[:5], k) # sanity check
# print(I)
# print(D)
# D, I = index.search(xq, k)     # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])    