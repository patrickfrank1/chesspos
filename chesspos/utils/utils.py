from os import listdir
from os.path import isfile, join, abspath

def files_from_directory(directory, file_type=None):
	absdir = abspath(directory)
	file_arr = None
	if file_type is None:
		file_arr = [join(absdir, f) for f in listdir(absdir) if isfile(join(absdir, f))]
	else:
		file_arr = [join(absdir, f) for f in listdir(absdir)
								if (isfile(join(absdir, f)) and f.endswith(file_type))]
	return file_arr

def correct_file_ending(file, ending):
	len_ending = len(ending)
	out_file = ""
	if file[-len_ending:] == ending:
		out_file = file
	else:
		out_file = f"{file}.{ending}"
	return out_file
