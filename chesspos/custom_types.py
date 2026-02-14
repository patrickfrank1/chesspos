from typing import TypeVar, Callable

import numpy as np
from chess import Board
from chess.pgn import Game, Headers

GameFilter = TypeVar("GameFilter", bound=Callable[[Headers], bool])
GameProcessor = TypeVar("GameProcessor", bound=Callable[[Game], np.ndarray])
PositionFilter = TypeVar("PositionFilter", bound=Callable[[Board], bool])
PositionProcessor = TypeVar("PositionProcessor", bound=Callable[[Board, int], np.ndarray])
PositionAggregator = TypeVar("PositionAggregator", bound=Callable[[np.ndarray], np.ndarray])
