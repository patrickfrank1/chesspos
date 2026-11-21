from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.chess_eval.backend import (
    DEFAULT_MODEL_ROOTS,
    catalog_by_id,
    discover_models,
    resolve_device,
    state_response,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class PositionRequest(BaseModel):
    start_fen: str = chess.STARTING_FEN
    moves: list[str] = Field(default_factory=list)
    model_ids: list[str] = Field(default_factory=list)


class MoveRequest(PositionRequest):
    source: str
    target: str
    promotion: str | None = None


def create_app(
    model_roots: Sequence[str | Path] | None = None,
    device: str | None = None,
) -> FastAPI:
    roots = [Path(root) for root in (model_roots or DEFAULT_MODEL_ROOTS)]
    catalog = catalog_by_id(discover_models(roots))
    resolved_device = resolve_device(device)

    app = FastAPI(title="chesspos position evaluator")
    app.state.catalog = catalog
    app.state.device = resolved_device
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "device": str(resolved_device),
            "models": len(catalog),
        }

    @app.get("/api/models")
    def models() -> dict:
        return {
            "models": [info.to_dict() for info in catalog.values()],
            "device": str(resolved_device),
        }

    @app.post("/api/position")
    def position(request: PositionRequest) -> dict:
        try:
            return state_response(
                request.start_fen,
                request.moves,
                request.model_ids,
                catalog,
                resolved_device,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/move")
    def move(request: MoveRequest) -> dict:
        promotion = request.promotion if request.promotion else ""
        uci = f"{request.source}{request.target}{promotion}"
        try:
            return state_response(
                request.start_fen,
                [*request.moves, uci],
                request.model_ids,
                catalog,
                resolved_device,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
