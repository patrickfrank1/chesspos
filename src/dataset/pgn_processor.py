from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Generator

import chess
import chess.pgn

from src.dataset.config import GameSubsampling, GameSubsampleTier, TimeControlFilter
from src.dataset.types import GameMetadata, GameRecord, PositionRecord
from src.utils.fileops import file_paths_from_directory


@dataclass
class PGNProcessor:
    subsampling: GameSubsampling = field(default_factory=GameSubsampling)
    time_control_filter: TimeControlFilter = field(default_factory=TimeControlFilter)

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
        metadata = self.extract_metadata(headers)

        if not self._keep_game(metadata):
            return None

        return self._build_record(game, metadata)

    def keep_game(self, headers: chess.pgn.Headers) -> bool:
        """Filter decision from headers only, without parsing movetext."""
        return self._keep_game(self.extract_metadata(headers))

    def _build_record(
        self, game: chess.pgn.Game, metadata: GameMetadata
    ) -> GameRecord | None:
        positions = list(self._extract_positions(game, metadata))
        if len(positions) == 0:
            return None

        return GameRecord(positions=positions, metadata=metadata)

    def extract_metadata(self, headers: chess.pgn.Headers) -> GameMetadata:
        white_elo = self._parse_elo(headers.get("WhiteElo"))
        black_elo = self._parse_elo(headers.get("BlackElo"))
        return GameMetadata(
            white_elo=white_elo,
            black_elo=black_elo,
            result=headers.get("Result"),
            opening=headers.get("Opening"),
            event=headers.get("Event"),
            date=headers.get("Date"),
            time_control=headers.get("TimeControl"),
        )

    def _parse_elo(self, elo_str: str | None) -> int | None:
        if elo_str is None or elo_str == "?":
            return None
        try:
            return int(elo_str)
        except ValueError:
            return None

    def _parse_time_control(self, tc: str | None) -> int | None:
        """Return the base time in seconds, or None when unknown/unparseable.

        Supported PGN formats: "sec" (e.g. "300"), "sec+inc" (e.g. "180+2"),
        and "moves/sec" (e.g. "40/900"). "-", "?", "*" and malformed values
        yield None.
        """
        if tc is None:
            return None
        tc = tc.strip()
        if tc in {"-", "?", "*", ""}:
            return None
        try:
            if "/" in tc:
                return int(tc.split("/")[-1])
            return int(tc.split("+")[0])
        except ValueError:
            return None

    def _keep_game(self, metadata: GameMetadata) -> bool:
        if not self._passes_time_control_filter(metadata):
            return False
        tier = self._match_tier(metadata)
        if tier is None:
            return False
        return random.random() < tier.rate

    def _passes_time_control_filter(self, metadata: GameMetadata) -> bool:
        min_seconds = self.time_control_filter.min_seconds
        if min_seconds is None:
            return True
        base_seconds = self._parse_time_control(metadata.time_control)
        if base_seconds is None:
            return True
        return base_seconds >= min_seconds

    def _match_tier(self, metadata: GameMetadata) -> GameSubsampleTier | None:
        strength = min(
            metadata.white_elo or 0,
            metadata.black_elo or 0,
        )
        matching = [tier for tier in self.subsampling.tiers if strength >= tier.min_elo]
        if not matching:
            return None
        return max(matching, key=lambda tier: tier.min_elo)

    def _extract_positions(
        self,
        game: chess.pgn.Game,
        metadata: GameMetadata,
    ) -> Generator[PositionRecord, None, None]:
        board = chess.Board()
        moves = list(game.mainline_moves())

        for ply, move in enumerate(moves):
            board.push(move)
            yield PositionRecord(
                board=board.copy(),
                ply=ply,
                metadata=metadata,
                move_sequence=list(moves[: ply + 1]),
            )

    def extract_temporal_windows(
        self,
        game: chess.pgn.Game,
        window_size: int = 10,
    ) -> Generator[list[PositionRecord], None, None]:
        metadata = self.extract_metadata(game.headers)
        if not self._keep_game(metadata):
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
