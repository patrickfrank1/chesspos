import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
	level=logging.ERROR,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	filename="pgn_extract.log"
)

from dataclasses import dataclass

import chess
import chess.pgn
import numpy as np

import chesspos.custom_types as ct


@dataclass
class GameProcessor():
	is_process_position: ct.PositionFilter
	position_processor: ct.PositionProcessor
	position_aggregator: ct.PositionAggregator = lambda position_encodings: position_encodings
	_board: chess.Board = chess.Board()
	
	def __call__(self, game: chess.pgn.Game) -> np.ndarray:
		self._board = chess.Board()
		encodings = self.game_processor(game)
		aggregated_encodings = self.position_aggregator(encodings)
		return aggregated_encodings

	def _push_move(self, move: chess.Move, move_nr: int = -1) -> None:
		"""Push a move to the board and append the position to the list of positions"""
		try:
			self._board.push(move)
		except Exception as e:
			logger.error(f"Exception occurred in position number {move_nr}")
			raise Exception(e)

	def game_processor(self, game: chess.pgn.Game) -> np.ndarray:
		"""Process a game and return a numpy array of the processed positions"""
		encodings = self.get_sample_encoding()
		for i, move in enumerate(game.mainline_moves()):
			self._push_move(move, i)
			if self.is_process_position(self._board):
				encoding = self.position_processor(self._board).reshape(1, -1)
				encodings = np.append(encodings, encoding, axis=0)
		return encodings

	def get_sample_encoding(self) -> np.ndarray:
		"""Process a game and return a dummy encoding, to get its shape and dtype"""
		board: chess.Board = chess.Board()
		encoding = self.position_processor(board)
		return np.empty((0, *encoding.shape), dtype=encoding.dtype)

