#!/usr/bin/env python3

import chess
import chess.pgn
import numpy as np
import h5py
from numpy.random import randint, shuffle
import argparse

def pgn2pos(file, ptype='bitboard', generate_tuples=False, save_file=None, tuple_file=None):

	game_list = []
	counter = 1
	fname = correct_file_ending(file, "pgn")

	with open(fname, 'r') as f:

		while True:
			game = chess.pgn.read_game(f)
			temp_game = None

			if game is None:
				break  # end of file
			else:
				if ptype == 'fen':
					temp_game = game_fen(game)
				elif ptype == 'bitboard':
					temp_game = game_bb(game)
				else:
					raise ValueError("ptype not implemented")

			game_list.append(temp_game)
			print(f" Games processed: {counter}", end="\r")
			counter += 1

	if save_file is not None:
		if ptype == 'fen':
			save_fen(game_list, save_file)
		elif ptype == 'bitboard':
			save_bb(game_list, save_file)

	if generate_tuples:
		tup = tuple_generator(game_list)

		if tuple_file is not None:
			save_tuples(tup, tuple_file)

	return 0

def game_fen(game):

	board = chess.Board()
	pos = []

	for move in game.mainline_moves():
		board.push(move)
		pos.append(board.fen())

	return pos

def game_bb(game):

	board = chess.Board()
	pos = []

	for move in game.mainline_moves():
		board.push(move)
		embedding = np.array([], dtype=bool)

		for color in [1, 0]:
			for i in range(1, 7): # P N B R Q K / white
				bmp = np.zeros(shape=(64,)).astype(bool)
				for j in list(board.pieces(i, color)):
					bmp[j] = True
				embedding = np.concatenate((embedding, bmp))

		additional = np.array([
			bool(board.turn),
			bool(board.castling_rights & chess.BB_A1),
			bool(board.castling_rights & chess.BB_H1),
			bool(board.castling_rights & chess.BB_A8),
			bool(board.castling_rights & chess.BB_H8)
		])
		embedding = np.concatenate((embedding, additional))
		pos.append(embedding)

	return pos

def correct_file_ending(file, ending):
	len_ending = len(ending)
	out_file = ""
	if file[-len_ending:] == ending:
		out_file = file
	else:
		out_file = f"{file}.{ending}"
	return out_file

def save_bb(game_list, file):
	fname = correct_file_ending(file, "h5py")
	position = []
	game_id = []

	for (i, game) in enumerate(game_list):
		for pos in game:
			position.append(pos)
			game_id.append(i)

	with h5py.File(fname, "w") as f:
		data1 = f.create_dataset("position", shape=(len(position), 773),
			dtype=bool, compression="gzip", compression_opts=9)
		data2 = f.create_dataset("game_id", shape=(len(position),),
			dtype=np.int, compression="gzip", compression_opts=9)

		data1[:] = position[:]
		data2[:] = game_id[:]

def save_fen(game_list, file):
	fname = correct_file_ending(file, "h5py")
	position = []
	game_id = []

	for (i, game) in enumerate(game_list):
		for pos in game:
			position.append(pos)
			game_id.append(i)

	with h5py.File(fname, "w") as f:
		data1 = f.create_dataset("position", shape=(len(position),),
			dtype=h5py.string_dtype(encoding='ascii'), compression="gzip", compression_opts=9)
		data2 = f.create_dataset("game_id", shape=(len(position),),
			dtype=np.int, compression="gzip", compression_opts=9)

		data1[:] = position[:]
		data2[:] = game_id[:]

def tuple_generator(game_list):
	tuples = []

	for (i, game) in enumerate(game_list):
		if len(game) <= 20 or len(game_list[(i+1)%len(game_list)]) <= 20:
			pass
		else:
			game_len = len(game)
			offset = 10
			sample_index = randint(offset, high=game_len-10, size=(2,)) # two samples per game
			next_game = np.copy(game_list[(i+1)%len(game_list)][10:])
			shuffle(next_game)
			tmp_tuple = np.array([
				game[sample_index[0]], game[1+sample_index[0]], # anchor + positive
				game[2+sample_index[0]], game[3+sample_index[0]], # 2 positive
				game[4+sample_index[0]], game[(14+sample_index[0])%game_len], #positive, distant
				*next_game[:9] # negative samples
			])
			tuples.append(tmp_tuple)
			print(f"Tuples generated: {i}", end="\r")	
	
	return tuples

def save_tuples(tuples, file):
	fname = correct_file_ending(file, "h5py")

	with h5py.File(fname, "w") as f:
		data1 = f.create_dataset("tuples", shape=(len(tuples),15, 773),
			dtype=bool, compression="gzip", compression_opts=9)

		data1[:] = tuples[:]

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Parse argumants to generate files')

	parser.add_argument('input', type=str, action="store", help='pgn file with input games')
	parser.add_argument('--format', type=str, default='bitboard', action="store", help='Encoding format for positions: fen, bitboard')
	parser.add_argument('--save_position', type=str, action="store", help='h5py file to store the encoded positions')
	parser.add_argument('--tuples', type=bool, default=False, action="store", help='h5py file to sore the encoded positions')
	parser.add_argument('--save_tuples', type=str, action="store", help='h5py file to store the encoded tuples')

	args = parser.parse_args()

	print(f"Input file at: {args.input}")
	print(f"Chess positions encoded as: {args.format}")
	print(f"Positions saved at: {args.save_position}")
	print(f"Tuples generated: {args.tuples}")
	print(f"Tuples saved at: {args.save_tuples}")

	test_file = args.input
	test_ptype = args.format
	test_save_file = args.save_position
	test_generate_tuples = args.tuples
	test_tuple_file = args.save_tuples

	pgn2pos(test_file, ptype=test_ptype, save_file=test_save_file,
			generate_tuples=test_generate_tuples, tuple_file=test_tuple_file)
