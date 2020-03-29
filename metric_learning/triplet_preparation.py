import os
import numpy as np
import h5py

def correct_file_ending(file, ending):
	len_ending = len(ending)
	out_file = ""
	if file[-len_ending:] == ending:
		out_file = file
	else:
		out_file = f"{file}.{ending}"
	return out_file

def tuples_from_table(file, table, tuple_indices=[0,1,6]):
	fname = correct_file_ending(file, 'h5')
	with h5py.File(fname, 'r') as hf:
		keys = hf.keys()
		print(keys)
		if table in keys:
			return hf[table][:,tuple_indices]

def tuples_from_file(file, table_id_prefix, tuple_indices=[0,1,6]):
	fname = correct_file_ending(file, 'h5')
	triplets = []
	with h5py.File(fname, 'r') as hf:
		print(hf.keys())
		for key in hf.keys():
			if table_id_prefix in key:
				triplets.extend(hf[key][:, tuple_indices])
	return np.asarray(triplets)

if __name__ == "__main__":

	file_path = os.path.abspath('data/samples/lichess_db_standard_rated_2020-02-06-tuples-strong.h5')
	#test_arr = tuples_from_table(file_path,"tuples_0")
	#print(test_arr.shape)

	test2 = tuples_from_file(file_path, table_id_prefix="tuples", tuple_indices=[0,1,2])
	print(f"test2.shape={test2.shape}")
	




# 	'''Test Dataset'''

# file_path = os.path.abspath('../data/samples/lichess_db_standard_rated_2013-01-tuples.h5')
# triplets = tuples_from_table(file_path, "tuples_0", tuple_indices=[0,1,2])

# train_triplets, test_triplets = train_test_split(triplets, test_size=0.2, random_state=42)
# #train_dummy_label = np.zeros_like( (train_triplets.shape[0]),) )
# #test_dummy_label = np.zeros_like( (test_triplets.shape[0],) )
# train_triplets.shape, test_triplets.shape #, train_dummy_label.shape, test_dummy_label.shape

# anc_train = train_triplets[:,0,:]
# pos_train = train_triplets[:,1,:]
# neg_train = train_triplets[:,2,:]

# anc_test = test_triplets[:,0,:]
# pos_test = test_triplets[:,1,:]
# neg_test = test_triplets[:,2,:]