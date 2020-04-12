import h5py
import numpy as np
import tensorflow as tf

from chesspos.utils import correct_file_ending

def inference_generator_from_bitboards(file, table_id_prefix, batch_size=128):
	bitboards = np.empty(shape=(0, 773), dtype=bool)
	fname = correct_file_ending(file, 'h5')

	with h5py.File(fname, 'r') as hf:
		for key in hf.keys():
			if table_id_prefix in key:
				new_bitboards = np.asarray(hf[key][:], dtype=bool)
				bitboards = np.concatenate((bitboards, new_bitboards))
			while len(bitboards) >= 3*batch_size:
				batch = [
					bitboards[:batch_size],
					bitboards[batch_size:2*batch_size],
					bitboards[2*batch_size:3*batch_size]
				]
				yield batch

def infer_embeddings(model, sample_generator, batch_size=128):
	pass

if __name__ == "__main__":

	encoder = tf.keras.models.load_model('metric_learning/test_model/model_encoder')
	encoder.summary()