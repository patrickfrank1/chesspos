from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator

import chess
import chess.pgn

from src.dataset.config import SamplingFilters
from src.dataset.types import GameMetadata, GameRecord, PositionRecord
from src.utils.fileops import file_paths_from_directory


@dataclass
class PGNProcessor:
    sampling_filters: SamplingFilters = field(default_factory=SamplingFilters)

    def process_directory(self, directory: str) -> Generator[GameRecord, None, None]:
        pgn_files = file_paths_from_directory(directory, ".pgn")
        for file_path in pgn_files:
            yield from self.process_file(file_path)

    def process_file(self, file_path: str) -> Generator[GameRecord, None, None]:
        with open(file_path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                record = self.extract_game(game)
                if record is not None and len(record.positions) > 0:
                    yield record

    def extract_game(self, game: chess.pgn.Game) -> GameRecord | None:
        headers = game.headers
        metadata = self._extract_metadata(headers)

        if not self._passes_elo_filter(metadata):
            return None

        positions = list(self._extract_positions(game, metadata))
        if len(positions) == 0:
            return None

        return GameRecord(positions=positions, metadata=metadata)

    def _extract_metadata(self, headers: chess.pgn.Headers) -> GameMetadata:
        white_elo = self._parse_elo(headers.get("WhiteElo"))
        black_elo = self._parse_elo(headers.get("BlackElo"))
        return GameMetadata(
            white_elo=white_elo,
            black_elo=black_elo,
            result=headers.get("Result"),
            opening=headers.get("Opening"),
            event=headers.get("Event"),
            date=headers.get("Date"),
        )

    def _parse_elo(self, elo_str: str | None) -> int | None:
        if elo_str is None or elo_str == "?":
            return None
        try:
            return int(elo_str)
        except ValueError:
            return None

    def _passes_elo_filter(self, metadata: GameMetadata) -> bool:
        filters = self.sampling_filters
        min_elo = min(
            metadata.white_elo or 1500,
            metadata.black_elo or 1500,
        )
        prob = 1.0 / (1.0 + math.exp(-0.005 * (min_elo - filters.min_elo)))
        return random.random() < prob

    def _extract_positions(
        self,
        game: chess.pgn.Game,
        metadata: GameMetadata,
    ) -> Generator[PositionRecord, None, None]:
        filters = self.sampling_filters
        board = chess.Board()
        moves = list(game.mainline_moves())

        for ply, move in enumerate(moves):
            if filters.min_ply > 0 and ply < filters.min_ply:
                board.push(move)
                continue
            if filters.max_ply is not None and ply >= filters.max_ply:
                break

            board.push(move)

            if self._should_sample_position(ply):
                yield PositionRecord(
                    board=board.copy(),
                    ply=ply,
                    metadata=metadata,
                    move_sequence=list(moves[: ply + 1]),
                )

    def _should_sample_position(self, ply: int) -> bool:
        prob = min((ply / 30.0) ** 2, 1.0) * self.sampling_filters.subsample_rate
        return random.random() < prob

    def extract_temporal_windows(
        self,
        game: chess.pgn.Game,
        window_size: int = 10,
    ) -> Generator[list[PositionRecord], None, None]:
        metadata = self._extract_metadata(game.headers)
        if not self._passes_elo_filter(metadata):
            return

        board = chess.Board()
        moves = list(game.mainline_moves())
        window: list[PositionRecord] = []

        for ply, move in enumerate(moves):
            board.push(move)

            record = PositionRecord(
                board=board.copy(),
                ply=ply,
                metadata=metadata,
                move_sequence=list(moves[: ply + 1]),
            )
            window.append(record)

            if len(window) > window_size:
                window.pop(0)

            if len(window) == window_size:
                yield list(window)
