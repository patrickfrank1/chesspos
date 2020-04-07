import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from chesspos.utils import correct_file_ending

def tuples_from_table(file, table, tuple_indices=[0, 1, 6]):
	'''
	Fetch tuples form a table in an h5 file.
	'''
	fname = correct_file_ending(file, 'h5')
	tuples = None
	with h5py.File(fname, 'r') as hf:
		keys = hf.keys()
		print(keys)
		if table in keys:
			tuples = np.asarray(hf[table][:, tuple_indices], dtype=bool)
	return tuples

def tuples_from_file(file, table_id_prefix, tuple_indices=[0, 1, 6]):
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

def tuples_from_file_array(files, table_id_prefix, tuple_indices=[0, 1, 6]):
	'''
	Return specified tuples from all relevant tables in a list of files.
	'''
	tuples = np.empty(shape=(0, len(tuple_indices), 773), dtype=bool)
	for file in files:
		tmp_tuples = tuples_from_file(file, table_id_prefix, tuple_indices=[0, 1, 6])
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

def train_inputs_file_array_generator(files, table_id_prefix, tuple_indices=[0,1,2,3,4,5,6], batch_size=16):
	tuples = np.empty(shape=(0, len(tuple_indices), 773), dtype=bool)
	for file in files:
		fname = correct_file_ending(file, 'h5')
		with h5py.File(fname, 'r') as hf:
			for key in hf.keys():
				if table_id_prefix in key:
					new_tuples = np.asarray(hf[key][:, tuple_indices], dtype=bool)
					tuples = np.concatenate((tuples, new_tuples))
					while len(tuples) >= batch_size:
						# augment tuples, refactor later
						# batch_train_easy = [
						# 	tuples[:batch_size,0,:],
						# 	tuples[:batch_size,1,:],
						# 	tuples[:batch_size,6,:]
						# ]
						# yield batch_train_easy
						# batch_train_medium = [
						# 	tuples[:batch_size,0,:],
						# 	tuples[:batch_size,1,:],
						# 	tuples[:batch_size,5,:]
						# ]
						# yield batch_train_medium
						# batch_train_semi_hard = [
						# 	tuples[:batch_size,0,:],
						# 	tuples[:batch_size,1,:],
						# 	tuples[:batch_size,4,:]
						# ]
						# yield batch_train_semi_hard
						batch_train_hard = [
							tuples[:batch_size,0,:],
							tuples[:batch_size,1,:],
							tuples[:batch_size,2,:]
						]
						yield batch_train_hard
						tuples = tuples[batch_size:,:,:]


def train_inputs_length(files, table_id_prefix):
	samples = 0
	for file in files:
		fname = correct_file_ending(file, 'h5')
		with h5py.File(fname, 'r') as hf:
			for key in hf.keys():
				if table_id_prefix in key:
					samples += len(hf[key])
	return samples


if __name__ == "__main__":

	#file_path = os.path.abspath('data/samples/lichess_db_standard_rated_2020-02-06-tuples-strong.h5')
	#test_arr = tuples_from_table(file_path,"tuples_0")
	#print(test_arr.shape)

	#test2 = tuples_from_file(file_path, table_id_prefix="tuples", tuple_indices=[0,1,2])
	#print(f"test2.shape={test2.shape}")

	file_paths = [
		os.path.abspath('data/samples/lichess_db_standard_rated_2020-02-06-tuples-strong.h5') #,
		#os.path.abspath('data/samples/lichess_db_standard_rated_2020-02-07-tuples-strong.h5')
	]
	# test3 = tuples_from_file_array(file_paths, table_id_prefix="tuples", tuple_indices=[0,1,2])
	# print(f"test3.shape={test3.shape}")
	# train, test = inputs_from_tuples(test3)
	# print(f"len(train): {len(train)}, train[0].shape: {train[0].shape}")
	# print(f"len(test): {len(test)}, test[0].shape: {test[0].shape}")

	print(train_inputs_length(file_paths, table_id_prefix="tuples"))

	generator = train_inputs_file_array_generator(file_paths, table_id_prefix="tuples",
					tuple_indices=[0,1,6], batch_size=128)
	for batch in generator:
		print(f"batch[0].shape: {batch[0].shape}")
