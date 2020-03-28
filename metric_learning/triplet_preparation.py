import h5py
import os

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

if __name__ == "__main__":

	file_path = os.path.abspath('data/samples/lichess_db_standard_rated_2013-01-tuples.h5')
	test_arr = tuples_from_table(file_path,
								 "tuples_0")
	print(test_arr.shape)
	