from os import listdir
from os.path import isfile, join

def files_from_directory(directory):
	file_arr = [f for f in listdir(directory) if isfile(join(directory, f))]
	return file_arr

