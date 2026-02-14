import logging
logger = logging.getLogger(__name__)

import h5py
import numpy as np
import chess

from chesspos.utils.file_utils import correct_file_ending

def samples_from_file(file, table_id_prefix, dtype=np.float32):
	'''
	Return samples from relevant table in a file.
	'''
	fname = correct_file_ending(file, 'h5')
	samples = []
	with h5py.File(fname, 'r') as hf:
		print(f"keys in {fname}: {hf.keys()}")
		for key in hf.keys():
			if table_id_prefix in key:
				samples.extend(hf[key][:])
	return np.asarray(samples, dtype=dtype)

def samples_from_file_array(files, table_id_prefix, dtype=np.float32):
	'''
	Return specified tuples from all relevant tables in a list of files.
	'''
	samples = None
	for i, file in enumerate(files):
		tmp_samples = samples_from_file(file, table_id_prefix, dtype=dtype)
		if i == 0:
			samples = np.empty((len(files), *tmp_samples.shape), dtype=dtype)
		samples[i, ...] = samples_from_file(file, table_id_prefix, dtype=dtype)

	return samples.reshape((-1, *samples.shape[2:]))

def samples_generator_from_file(file, table_id_prefix, dtype=np.float32):
	'''
	Return generator of samples from relevant table in a file.
	'''
	fname = correct_file_ending(file, 'h5')
	samples = []
	with h5py.File(fname, 'r') as hf:
		for key in hf.keys():
			if table_id_prefix in key:
				yield np.asarray(hf[key][:], dtype=dtype)

def sample_generator_from_file_array(files, table_id_prefix, dtype=np.float32):
	'''
	Yield generator of samples from all relevant tables in a list of files.
	'''
	for file in files:
		for sample in samples_generator_from_file(file, table_id_prefix, dtype=dtype):
			yield sample
