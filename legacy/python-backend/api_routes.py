from __future__ import annotations

from fastapi import APIRouter, HTTPException

from game_service import BattleSessionManager, GameSessionManager, ServiceError
from models_service import discover_models
from schemas import (
    BattleState,
    CreateBattleRequest,
    CreateGameRequest,
    GameState,
    ModelInfo,
    MoveRequest,
    NNResponse,
)


def create_api_router(
    game_manager: GameSessionManager,
    battle_manager: BattleSessionManager,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/models", response_model=list[ModelInfo])
    async def list_models() -> list[dict[str, object]]:
        return discover_models()

    @router.post("/game/new", response_model=GameState)
    async def create_game(request: CreateGameRequest) -> dict[str, object]:
        return await _with_service_error(game_manager.create_session(request.model_path, request.simulations))

    @router.get("/game/{session_id}", response_model=GameState)
    async def get_game(session_id: str) -> dict[str, object]:
        return await _with_service_error(game_manager.get_state(session_id))

    @router.post("/game/{session_id}/move", response_model=GameState)
    async def make_move(session_id: str, request: MoveRequest) -> dict[str, object]:
        return await _with_service_error(game_manager.make_human_move(session_id, request.row, request.col))

    @router.post("/game/{session_id}/pass", response_model=GameState)
    async def pass_move(session_id: str) -> dict[str, object]:
        return await _with_service_error(game_manager.pass_move(session_id))

    @router.post("/game/{session_id}/resign", response_model=GameState)
    async def resign(session_id: str) -> dict[str, object]:
        return await _with_service_error(game_manager.resign(session_id))

    @router.post("/game/{session_id}/ai-move", response_model=GameState)
    async def ai_move(session_id: str) -> dict[str, object]:
        return await _with_service_error(game_manager.ai_move(session_id))

    @router.post("/game/{session_id}/undo", response_model=GameState)
    async def undo(session_id: str) -> dict[str, object]:
        return await _with_service_error(game_manager.undo(session_id))

    @router.post("/game/{session_id}/redo", response_model=GameState)
    async def redo(session_id: str) -> dict[str, object]:
        return await _with_service_error(game_manager.redo(session_id))

    @router.get("/game/{session_id}/nn", response_model=NNResponse)
    async def get_nn(session_id: str) -> dict[str, object]:
        return await _with_service_error(game_manager.nn_predictions(session_id))

    @router.post("/battle/new", response_model=BattleState)
    async def create_battle(request: CreateBattleRequest) -> dict[str, object]:
        return await _with_service_error(
            battle_manager.create_session(
                request.black_model_path,
                request.white_model_path,
                request.simulations,
            )
        )

    @router.get("/battle/{session_id}", response_model=BattleState)
    async def get_battle(session_id: str) -> dict[str, object]:
        return await _with_service_error(battle_manager.get_state(session_id))

    return router


async def _with_service_error(awaitable):
    try:
        return await awaitable
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
