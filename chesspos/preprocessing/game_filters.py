from typing import Tuple
import logging

import chess
import chess.pgn

import chesspos.custom_types as ct

def no_filter(header: chess.pgn.Headers) -> bool:
	return True

def elo_filter(white_elo_range: Tuple[int, int], black_elo_range: Tuple[int, int], debug=False) -> ct.GameFilter:
	def filter(header: chess.pgn.Headers) -> bool:
		logger = logging.getLogger(__name__)
		#headers are often non-standard, try..except!
		process_game = True
		try:
			if not white_elo_range[0] < header.get("WhiteElo") < white_elo_range[1]:
				process_game = False
			elif not black_elo_range[0] < header.get("BlackElo") < black_elo_range[1]:
				process_game = False
		except Exception as e:
			if debug:
				logger.error(f"Exception in filter_by_elo", exec_info=True)
			process_game = False
		finally:
			return process_game

def time_control_filter(time_range_minutes: Tuple[int, int], debug: bool = False) -> ct.GameFilter:
	def filter(header: chess.pgn.Headers) -> bool:
		logger = logging.getLogger(__name__)
		#headers are often non-standard, try..except!
		process_game = True
		try:
			minutes, increment_seconds = _get_time_and_increment_from_time_control_header(header.get("TimeControl"))
			total_time = minutes +_increment_to_total_time_equivalent(increment_seconds, 40)
			if not time_range_minutes[0] < total_time < time_range_minutes[1]:
				process_game = False
		except Exception as e:
			if debug:
				logger.error(f"Exception in filter_by_time_control", exec_info=True)
			process_game = False
		finally:
			return process_game
	return filter
		
def white_wins(header: chess.pgn.Headers) -> bool:
	"""Filter games where white wins"""
	result = header.get("Result")
	return result == "1-0"
	
def black_wins(header: chess.pgn.Headers) -> bool:
	"""Filter games where white wins"""
	result = header.get("Result")
	return result == "0-1"

def _get_time_and_increment_from_time_control_header(time_control: str) -> Tuple[int, int]:
	minute, second = time_control.split("+")
	return int(minute), int(second)

def _increment_to_total_time_equivalent(increment: int, moves: int) -> int:
	return increment * moves / 60
