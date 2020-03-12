#!/usr/bin/env python3

import chess
import chess.pgn

def pgn2pos(file, ptype='bb'):

	game_list = []

	with open(file, 'r') as f:

		while True:
			game = chess.pgn.read_game(f)

			if game is None:
				break  # end of file
			else:
				temp_game = gamepos(game=game, ptype=ptype)
				game_list.append(temp_game)

	return game_list

def gamepos(game, ptype='bb'):

	board = chess.Board()
	game_fen = []

	for move in game.mainline_moves():
		board.push(move)

		if ptype == 'fen':
			game_fen.append(board.fen())
		else:
			raise ValueError("This ptype is not implemented")

	return game_fen

if __name__ == "__main__":

	test_games = pgn2pos(file="data/test.pgn", ptype='fen')

	for tg in test_games:
		print(tg)

#for i in range(1,6):
#	print(board.pieces(i,0))
#	print("\n vars",(i,0),"\n")
#	print(board.pieces(i,1))
#	print("\n vars",(i,1),"\n")