import numpy as np
from numpy.random import randint, shuffle
import h5py
import chess
import chess.pgn

from chesspos.utils.utils import correct_file_ending
from chesspos.utils.board_bitboard_converter import board_to_bitboard


def pgn_to_bitboard(pgn_file, generate_tuples=False, save_file=None,
	tuple_file=None, chunksize=100000, game_filter=None):
	game_list = []
	game_id = []
	counter = 1
	game_index = -1
	save_number = 0
	pgn_name = correct_file_ending(pgn_file, "pgn")

	with open(pgn_name, 'r') as f:
		while True:
			next_game = chess.pgn.read_game(f)
			game_index += 1

			if next_game is not None and game_filter is not {} and \
				filter_out(next_game.headers, game_filter):
				continue

			if counter % chunksize == 0 or next_game is None:
				print("")
				if save_file is not None:
					save_bb(game_list, game_id, save_file, dset_num=save_number)
					print("Game positions saved.")
				else:
					raise ValueError("Save bitbaord file path not provided.")

				if generate_tuples:
					tup = tuple_generator(game_list)
					print("Tuples generated.")
					if tuple_file is not None:
						save_tuples(tup, tuple_file, dset_num=save_number)
						print("Tuples saved.")
					else:
						raise ValueError("Save tuple file path not provided.")

				print(f"\rChunk {save_number} processed.")
				save_number += 1
				counter = 1
				game_list = []
				if next_game is None:
					break # end of file, break the while True loop

			else:
				temp_game = game_bb(next_game, game_nr=counter)
				if len(temp_game) > 0:
					game_list.append(temp_game)
					game_id.append(game_index)
				print(f" Games parsed: {game_index} Games processed: {counter}", end="\r")

				counter += 1

	return 0

def filter_out(header, game_filter):
	#headers are often non-standard, try..except!
	out = False
	if 'elo_min' in game_filter.keys():
		try:
			if int(header.get("WhiteElo")) < int(game_filter["elo_min"]) \
			or int(header.get("BlackElo")) < int(game_filter["elo_min"]):
				out = True
		except Exception as e:
			#print(f"\n WhiteElo {header.get('WhiteElo')}, BlackElo {header.get('BlackElo')}")
			#print(e)
			out = True
	if 'time_min' in game_filter.keys():
		try:
			minute, second = header.get("TimeControl").split("+")
			if int(minute) + int(second) < int(game_filter["time_min"]):
				out = True
		except Exception as e:
			#print(f"\n{header.get('TimeControl').split('+')}")
			#print(e)
			pass
	return out

def game_bb(game, game_nr=0):

	board = chess.Board()
	pos = []
	for move in game.mainline_moves():
		try:
			board.push(move)
		except Exception as e:
			print(f"Exception occurred in game number {game_nr}")
			print(e)
			return pos
		else:
			embedding = board_to_bitboard(board)
			pos.append(embedding)
	return pos

def save_bb(game_list, game_id, file, dset_num=0):
	fname = correct_file_ending(file, "h5")
	position = []
	gid = []

	for (i, game) in enumerate(game_list):
		for pos in game:
			position.append(pos)
			gid.append(game_id[i])

	with h5py.File(fname, "a") as f:
		data1 = f.create_dataset(f"position_{dset_num}", shape=(len(position), 773),
			dtype=bool, compression="gzip", compression_opts=9)
		data2 = f.create_dataset(f"game_id_{dset_num}", shape=(len(position),),
			dtype=np.int, compression="gzip", compression_opts=9)

		data1[:] = position[:]
		data2[:] = gid[:]


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

	return tuples

def save_tuples(tuples, file, dset_num=0):
	fname = correct_file_ending(file, "h5")

	with h5py.File(fname, "a") as f:
		data1 = f.create_dataset(f"tuples_{dset_num}", shape=(len(tuples), 15, 773),
			dtype=bool, compression="gzip", compression_opts=9)

		data1[:] = tuples[:]
