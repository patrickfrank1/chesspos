from os import listdir
from os.path import isfile, join

def files_from_directory(directory):
	file_arr = [f for f in listdir(directory) if isfile(join(directory, f))]
	return file_arr

def correct_file_ending(file, ending):
	len_ending = len(ending)
	out_file = ""
	if file[-len_ending:] == ending:
		out_file = file
	else:
		out_file = f"{file}.{ending}"
	return out_file
