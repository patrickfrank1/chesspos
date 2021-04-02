import argparse
from os.path import join, abspath

import faiss

from chesspos.utils import files_from_directory, correct_file_ending
from chesspos.binary_index import index_load_bitboard_file_array, init_binary_index

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Generate a searchable faiss index from bitboards.')

	parser.add_argument('input_directory', type=str, action="store",
		help='directory that contains all bitboards (stored in h5 files), which should be stored in an index'
	)
	parser.add_argument('--table_key', type=str, action="store", default="position",
		help='substring, that is contained in all relevant h5 table names (default: position should work out of the box)'
	)
	parser.add_argument('--save_path', type=str, action='store', default="",
		help="full path- and filename to the output index, defaults to the input directory"
	)

	args = parser.parse_args()

	# print inputs for the user
	print(f"Input directory: {args.input_directory}")
	print(f"Table key for h5 tables: {args.table_key}")
	print(f"Index saved at: {args.save_path}\n\n")

	# prepare variables
	dimension = 0
	is_binary = True
	index_type = 'bitboard' # this might change in the future, eg to support non-binary indices
	if index_type == 'bitboard':
		dimension = 776
		is_binary = True

	save_path = None
	if args.save_path == "":
		save_path = join(abspath(args.input_directory), "bitboard_index")
	else:
		save_path = abspath(args.save_path)

	# execute script with inputs
	files = files_from_directory(args.input_directory, file_type='h5')
	print("Files to be added to index:")
	print(files)
	index = init_binary_index(dimension)
	index, file_ids = index_load_bitboard_file_array(
		files,
		args.table_key,
		index
	)
	print(f"The index contains {round(index.ntotal/1.e6,3)} million positions.")

	save = correct_file_ending(save_path, 'faiss')
	if is_binary:
		faiss.write_index_binary(index, save)

	print(f"Index successfully saved at {save_path}.")
