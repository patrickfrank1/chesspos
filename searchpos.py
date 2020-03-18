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
					chunks = 100000
					vectors = np.zeros(shape=(chunks, 776), dtype=np.bool_)
					for i in range(math.floor(hf_len/chunks)):
						vectors[:, :773] = hf[key][i*chunks:(i+1)*chunks,:773]
						uint_vec = np.zeros(shape=(chunks,97), dtype=np.uint8)
						for j in range(97):
							uint_vec[:,j] = np.packbits(vectors[:,j*8:(j+1)*8])
						print(vectors.shape, index.d, vectors[0], uint_vec[0])
						index.add(uint_vec)
						print(f"{index.ntotal / 1.e6} million positions stored")

def transform_bb_uint8(bb_array):
	pass


if __name__ == "__main__":
	
	faiss_load_bb(["data/db/lichess_db_standard_rated_2013-01-bb"])








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