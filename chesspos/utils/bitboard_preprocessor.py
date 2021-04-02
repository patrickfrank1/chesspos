import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from chesspos.utils.utils import correct_file_ending


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

def singlet_factory(index):

	def singlets(tuple_batch):
		return (tuple_batch[:,index,:],tuple_batch[:,index,:]) # x, label

	return singlets

hard_triplets = triplet_factory([0,1,2])
semihard_triplets = triplet_factory([0,1,5])
easy_triplets = triplet_factory([0,1,6])
singlets = singlet_factory(0)