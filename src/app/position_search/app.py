"""FastAPI app that serves the position-search web UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Sequence

import chess
import chess.svg
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.dataset.token_stream import TokenStreamEncoder
from src.search.codec import decode_board
from src.search.query import GameHit, search_games
from src.app.position_search.examples import EXAMPLES, Example

PAGE_PATH = Path(__file__).with_name("static") / "index.html"
QUERY_BOARD_SIZE = 320
RESULT_BOARD_SIZE = 200


class SearchRequest(BaseModel):
    """JSON body accepted by ``POST /api/search``."""

    fen: str = Field(min_length=1, description="Position to search for")
    n_games: int = Field(default=10, ge=1, le=100)
    k: int = Field(default=500, ge=1, le=5000)


def _board_svg(board: chess.Board, size: int) -> str:
    return chess.svg.board(board, size=size)


def _game_payload(game: GameHit) -> dict:
    payload = game.to_dict()
    payload["svg"] = _board_svg(decode_board(game.best_position), RESULT_BOARD_SIZE)
    return payload


def create_app(
    *,
    store,
    embedder,
    examples: Sequence[Example] = EXAMPLES,
) -> FastAPI:
    """Build the app around an open position store and a checkpoint embedder."""
    examples = tuple(examples)
    encoder = TokenStreamEncoder()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        store.close()

    app = FastAPI(title="chesspos position search", lifespan=lifespan)
    app.state.store = store
    app.state.embedder = embedder
    app.state.examples = examples

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(
            PAGE_PATH.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/examples")
    def list_examples(response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return {
            "examples": [
                {"label": example.label, "fen": example.fen} for example in examples
            ]
        }

    @app.get("/api/info")
    def info(response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return {"rows": int(store.count()), "dim": int(embedder.dim)}

    @app.post("/api/search")
    def search(request: SearchRequest) -> dict:
        try:
            board = chess.Board(request.fen)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid FEN: {exc}") from exc
        if not board.is_valid():
            raise HTTPException(
                status_code=400,
                detail="invalid FEN: the position is not reachable",
            )
        segment = encoder.encode(board)
        games = search_games(
            store=store,
            embedder=embedder,
            segment=segment,
            n_games=request.n_games,
            candidate_limit=request.k,
        )
        return {
            "query": {
                "fen": board.fen(),
                "svg": _board_svg(board, QUERY_BOARD_SIZE),
            },
            "games": [_game_payload(game) for game in games],
        }

    return app
