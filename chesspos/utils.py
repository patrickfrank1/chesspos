from os import listdir
from os.path import isfile, join, abspath

def files_from_directory(directory):
	absdir = abspath(directory)
	file_arr = [join(absdir, f) for f in listdir(absdir) if isfile(join(absdir, f))]
	return file_arr

def correct_file_ending(file, ending):
	len_ending = len(ending)
	out_file = ""
	if file[-len_ending:] == ending:
		out_file = file
	else:
		out_file = f"{file}.{ending}"
	return out_file

if __name__ == "__main__":
	print(files_from_directory("data/bitboards/testdir"))