#!/usr/bin/env python3

import chess
import chess.pgn
import numpy as np
import h5py

def pgn2pos(file, ptype='bitboard', save_file=None):

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

	return game_list

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

if __name__ == "__main__":

	test_file = "data/test3.pgn"
	test_ptype = 'bitboard'
	test_save_file = "data/test3_bb.h5py"

	test_games = pgn2pos(test_file, ptype=test_ptype, save_file=test_save_file)