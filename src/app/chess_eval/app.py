from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Sequence

import chess
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.app.chess_eval.attribution import (
    METHODS,
    OCCLUSION_MODES,
    attribute_position,
)
from src.app.chess_eval.backend import (
    DEFAULT_MODEL_ROOTS,
    catalog_by_id,
    discover_models,
    load_model,
    replay,
    resolve_device,
    state_response,
)
from src.app.chess_eval.puzzles import background_boards
from src.app.chess_eval.samples import DEFAULT_PUZZLE_DIR, load_puzzle_samples
from src.app.chess_eval.stockfish import ensure_nnue, nnue_info

STATIC_DIR = Path(__file__).resolve().parent / "static"
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")


class PositionRequest(BaseModel):
    start_fen: str = chess.STARTING_FEN
    moves: list[str] = Field(default_factory=list)
    model_ids: list[str] = Field(default_factory=list)


class MoveRequest(PositionRequest):
    source: str
    target: str
    promotion: str | None = None


class AttributionRequest(PositionRequest):
    model_id: str
    method: str = "occlusion"
    occlusion: str = "mask"
    samples: int = Field(default=128, ge=2, le=4096)
    seed: int = 0


def create_app(
    model_roots: Sequence[str | Path] | None = None,
    device: str | None = None,
    nnue_cache_dir: str | Path | None = None,
    puzzle_data_dir: str | Path | None = None,
) -> FastAPI:
    roots = [Path(root) for root in (model_roots or DEFAULT_MODEL_ROOTS)]
    catalog = catalog_by_id(discover_models(roots))
    resolved_device = resolve_device(device)
    resolved_nnue_dir = Path(nnue_cache_dir) if nnue_cache_dir else None
    resolved_puzzle_dir = (
        Path(puzzle_data_dir) if puzzle_data_dir is not None else DEFAULT_PUZZLE_DIR
    )

    app = FastAPI(title="chesspos position evaluator")
    app.state.catalog = catalog
    app.state.device = resolved_device
    app.state.nnue_cache_dir = resolved_nnue_dir

    @app.middleware("http")
    async def isolation_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    @app.get("/health")
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

    @app.get("/api/samples")
    def samples() -> dict:
        try:
            return {
                "samples": [
                    sample.to_dict()
                    for sample in load_puzzle_samples(resolved_puzzle_dir)
                ]
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/stockfish/info")
    def stockfish_info() -> dict:
        return nnue_info(resolved_nnue_dir)

    @app.get("/api/stockfish/nnue")
    def stockfish_nnue() -> FileResponse:
        try:
            path = ensure_nnue(resolved_nnue_dir)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=path.name,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.post("/api/attribution")
    def attribution(request: AttributionRequest) -> dict:
        info = catalog.get(request.model_id)
        if info is None:
            raise HTTPException(
                status_code=404, detail=f"unknown model id {request.model_id!r}"
            )
        if request.method not in METHODS:
            raise HTTPException(
                status_code=400,
                detail=f"method must be one of {METHODS}",
            )
        if request.occlusion not in OCCLUSION_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"occlusion must be one of {OCCLUSION_MODES}",
            )
        try:
            boards, _ = replay(request.start_fen, request.moves)
            handle = load_model(info, resolved_device)
            background = (
                background_boards(resolved_puzzle_dir, count=4)
                if request.method == "shap"
                else ()
            )
            result = attribute_position(
                handle,
                boards,
                device=resolved_device,
                method=request.method,
                occlusion=request.occlusion,
                samples=request.samples,
                seed=request.seed,
                background=background,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"fen": boards[-1].fen(), "attribution": result.to_dict()}

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
