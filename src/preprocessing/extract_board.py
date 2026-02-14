import random
import math
from typing import Generator

import chess
import chess.pgn as pgn

from src.utils.fileops import file_paths_from_directory

SAMPLE_POSITIONS = True
SAMPLE_ELO = True
SUBSAMPLE_POSITIONS = 0.33
THRESHOLD_ELO = 2000.0


def sample_position_by_ply(ply: int) -> float:
    return min((ply/30.0**2), 1.0)


def sample_elo(header: pgn.Headers) -> float:
    white_elo = 1500 if header["WhiteElo"] == "?" else int(header["WhiteElo"])
    black_elo = 1500 if header["BlackElo"] == "?" else int(header["BlackElo"])
    min_elo = min(white_elo, black_elo)
    return 1.0 / (1.0 + math.exp(-0.005*(min_elo - THRESHOLD_ELO)))


def extract_board(directory: str) -> Generator[chess.Board]:
    """
    This extracts chess positions from a .pgn file
    """
    pgn_files = file_paths_from_directory(directory, ".pgn")
    print(f"These pgn files will be processed: {pgn_files}")
    # iterate over all pgn files
    for file_path in pgn_files:
        print(f"Extraction from {file_path}")
        with open(file_path, 'r') as file:
            # iterate over all games in a file
            while True:
                header = pgn.read_headers(file)
                if header is None:
                    break
                # TODO: make statistics about discarded games
                # Discarded elo
                # Discarded moves
                # Error with header or body
                if random.random() < sample_elo(header):
                    game = pgn.read_game(file)
                    if game is None:
                        print("\nGame is None?!\n")
                        continue
                    board = chess.Board()
                    # iterate over all moves in a game
                    for i, move in enumerate(game.mainline_moves()):
                        try:
                            board.push(move)
                        except Exception as exc:
                            print(f"Invalid move at position {i}.")
                            print(exc)
                        if random.random() < sample_position_by_ply(i) * SUBSAMPLE_POSITIONS:
                            yield board
