#!/usr/bin/env python3

import chess
import chess.pgn
import numpy as np
import h5py

def pgn2pos(file, ptype='bb'):

	game_list = []
	counter = 1

	with open(file, 'r') as rf:

		while True:
			game = chess.pgn.read_game(rf)

			if game is None:
				break  # end of file
			else:
				temp_game = game_pos(game=game, ptype=ptype)
				game_list.append(temp_game)
				print(counter, end="\r")
				counter += 1

	return game_list

def game_pos(game, ptype='bb'):

	board = chess.Board()
	pos = []

	for move in game.mainline_moves():
		board.push(move)
		embedding = np.array([], dtype=bool)

		if ptype == 'fen':
			pos.append(board.fen())
		elif ptype == 'bb':
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
		else:
			raise ValueError("This ptype is not implemented")

	return pos


if __name__ == "__main__":

	test_games = pgn2pos(file="data/lichess_db_standard_rated_2013-01.pgn", ptype='bb')
	position = []
	game_id = []

	for (num, g) in enumerate(test_games):
		for p in g:
			position.append(p)
			game_id.append(num)

	print(len(position), len(game_id))

	with h5py.File("data/lichess_db_standard_rated_2013-01.hdf5", "w") as f:
		data1 = f.create_dataset("positions", shape=(len(position), 773),
				compression="gzip", compression_opts=9)
		data2 = f.create_dataset("game_id", shape=(len(position),),
			compression="gzip", compression_opts=9)
		
		data1[:] = position[:]
		data2[:] = game_id[:]
