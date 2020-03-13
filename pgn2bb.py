#!/usr/bin/env python3

import chess
import chess.pgn
import numpy as np


def pgn2pos(file, ptype='bb'):

	game_list = []

	with open(file, 'r') as f:

		while True:
			game = chess.pgn.read_game(f)

			if game is None:
				break  # end of file
			else:
				temp_game = game_fen(game=game, ptype=ptype)
				game_list.append(temp_game)

	return game_list

def game_fen(game, ptype='bb'):

	board = chess.Board()
	pos = []

	for move in game.mainline_moves():
		board.push(move)
		s = []
		embedding = np.array([], dtype=bool)

		if ptype == 'fen':
			pos.append(board.fen())
		elif ptype == 'bb':
			for i in range(1,7): # P N B R Q K / white
				s.append(list(board.pieces(i,0)))
				bmp = np.zeros(shape=(64,))
				for j in range(64):
					if j in list(board.pieces(i,1)):
						bmp[j] = True
				print(bmp)
				embedding = np.concatenate((embedding,bmp))
			for i in range(1,7): # P N B R Q K / black
				s.append(list(board.pieces(i,0)))
				bmp = np.zeros(shape=(64,))
				for j in range(64):
					if j in list(board.pieces(i,0)):
						bmp[j] = True
				print(bmp)
				embedding = np.concatenate((embedding,bmp))
			embedding = np.concatenate((embedding, [bool(board.turn)])) # white=1, black=0
			print(s)
			embedding = np.concatenate((embedding, [bool(board.castling_rights & chess.BB_A1)]))
			embedding = np.concatenate((embedding, [bool(board.castling_rights & chess.BB_H1)]))
			embedding = np.concatenate((embedding, [bool(board.castling_rights & chess.BB_A8)]))
			embedding = np.concatenate((embedding, [bool(board.castling_rights & chess.BB_H8)]))
			
			print(embedding, len(embedding))
			

#	print(board.pieces(i,0))
#	print("\n vars",(i,0),"\n")
#	print(board.pieces(i,1))
#	print("\n vars",(i,1),"\n")
		else:
			raise ValueError("This ptype is not implemented")

	return pos


if __name__ == "__main__":

	test_games = pgn2pos(file="data/test2.pgn", ptype='bb')

	for tg in test_games:
		print(tg)
