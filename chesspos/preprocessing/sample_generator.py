import os
import h5py
from typing import Callable, List, Tuple
import numpy as np
import tensorflow as tf

from chesspos.utils.file_utils import correct_file_ending, files_from_directory
from chesspos.preprocessing.utils import sample_generator_from_file_array


class SampleGenerator():
	def __init__(
		self,
		sample_dir,
		sample_preprocessor: Callable[[np.ndarray], np.ndarray],
		batch_size=16,
		sample_type=np.float32,
	):
		self.H5_COL_KEY = 'encoding'
		self.sample_dir = sample_dir
		self.sample_preprocessor = sample_preprocessor
		self.batch_size = batch_size
		self.sample_type = sample_type
		self.number_samples, self.sample_shape = self._get_sample_dimensions()
		self.generator_function = self._construct_generator_function()


	def _construct_generator_function(self):
		def generator_function():
			assert(self.sample_shape is not None, "SampleGenerator has not been initialized with a sample shape.")
			sample_files = files_from_directory(os.path.abspath(self.sample_dir), file_type="h5")
			for samples in sample_generator_from_file_array(sample_files, self.H5_COL_KEY, self.sample_type):
				tmp_samples = samples
				i = 0
				while len(tmp_samples) >= self.batch_size:
					yield self.sample_preprocessor(samples[:self.batch_size, ...])
					samples = samples[self.batch_size:, ...]
		return generator_function


	def _get_generator_signature(self):
		sample = next(self.generator_function())
		out_shape = None
		if isinstance(sample, np.ndarray):
			out_shape = sample.shape
		elif isinstance(sample, (Tuple, List)):
			out_shape = []
			for s in sample:
				out_shape.append(s.shape)
		return out_shape

	def get_tf_dataset(self):
		out_shape = self._get_generator_signature()
		output_signature = []
		for s in out_shape:
			output_signature.append(tf.TensorSpec(s, self.sample_type))

		return tf.data.Dataset.from_generator(
			self.generator_function,
			output_signature=tuple(output_signature)
		)


	def get_generator(self):
		return self.generator_function()


	def _get_sample_dimensions(self):
		"""
		Return the number of samples (first dimension) and the shape of the samples (other dimensions).
		"""
		samples = 0
		shape = None
		sample_files = files_from_directory(os.path.abspath(self.sample_dir), file_type="h5")
		for i, file in enumerate(sample_files):
			fname = correct_file_ending(file, 'h5')
			with h5py.File(fname, 'r') as hf:
				for key in hf.keys():
					if self.H5_COL_KEY in key:
						samples += hf[key].shape[0]
						if i == 0:
							shape = hf[key].shape[1:]
						else:
							assert(shape == hf[key].shape[1:], "Shape of samples in file {} does not match shape of samples in file {}".format(i, i-1))
		return samples, shape
