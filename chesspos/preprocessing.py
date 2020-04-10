import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from chesspos.utils import correct_file_ending

# REDUNDANT -> DELETE
# def tuples_from_table(file, table, tuple_indices=[0,1,6]):
# 	'''
# 	Fetch tuples form a table in an h5 file.
# 	'''
# 	fname = correct_file_ending(file, 'h5')
# 	tuples = None
# 	with h5py.File(fname, 'r') as hf:
# 		keys = hf.keys()
# 		print(keys)
# 		if table in keys:
# 			tuples = np.asarray(hf[table][:, tuple_indices], dtype=bool)
# 	return tuples

def tuples_from_file(file, table_id_prefix, tuple_indices=[0,1,6]):
	'''
	Return specified tuples from all relevant tables in a file.
	'''
	fname = correct_file_ending(file, 'h5')
	tuples = []
	with h5py.File(fname, 'r') as hf:
		print(hf.keys())
		for key in hf.keys():
			if table_id_prefix in key:
				tuples.extend(hf[key][:, tuple_indices])
	return np.asarray(tuples, dtype=bool)

def tuples_from_file_array(files, table_id_prefix, tuple_indices=[0,1,6]):
	'''
	Return specified tuples from all relevant tables in a list of files.
	'''
	tuples = np.empty(shape=(0, len(tuple_indices), 773), dtype=bool)
	for file in files:
		tmp_tuples = tuples_from_file(file, table_id_prefix, tuple_indices=[0,1,6])
		tuples = np.concatenate((tuples,tmp_tuples))
	return tuples

def inputs_from_tuples(tuple_array, test_split=True, test_size=0.2):
	'''
	Split tuples into train/test and anchor/positive/negative.
	'''
	train_tuples = None
	test_tuples = None
	if test_split:
		train_tuples, test_tuples = train_test_split(tuple_array, test_size=test_size, shuffle=False)
	else:
		train_tuples = tuple_array

	data_train = [train_tuples[:,0,:], train_tuples[:,1,:], train_tuples[:,2,:]]
	data_test = None

	if test_tuples is not None:
		data_test = [test_tuples[:,0,:], test_tuples[:,1,:], test_tuples[:,2,:]]

	return data_train, data_test

def input_generator(file_arr, table_id_prefix, selector_fn, batch_size=16):
	tuples = np.empty(shape=(0, 15, 773), dtype=bool)
	for file in file_arr:
		fname = correct_file_ending(file, 'h5')
		with h5py.File(fname, 'r') as hf:
			for key in hf.keys():
				if table_id_prefix in key:
					new_tuples = np.asarray(hf[key][:], dtype=bool)
					tuples = np.concatenate((tuples, new_tuples))
					while len(tuples) >= batch_size:
						if isinstance(selector_fn, (list, tuple, np.ndarray)):
							for fn in selector_fn:
								triplets = fn(tuples[:batch_size])
								yield triplets
						else:
							triplets = selector_fn(tuples[:batch_size])
							yield triplets
						tuples = tuples[batch_size:]

def triplet_factory(indices):

	assert len(indices) == 3

	def triplets(tuple_batch):
		t = [
			tuple_batch[:,indices[0],:],
			tuple_batch[:,indices[1],:],
			tuple_batch[:,indices[2],:]
		]
		return t

	return triplets

hard_triplets = triplet_factory([0,1,2])
semihard_triplets = triplet_factory([0,1,5])
easy_triplets = triplet_factory([0,1,6])


def input_length(files, table_id_prefix):
	samples = 0
	for file in files:
		fname = correct_file_ending(file, 'h5')
		with h5py.File(fname, 'r') as hf:
			for key in hf.keys():
				if table_id_prefix in key:
					samples += len(hf[key])
	return samples
