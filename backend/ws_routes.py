from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from game_service import BattleSessionManager, GameSessionManager, ServiceError


def create_ws_router(
    game_manager: GameSessionManager,
    battle_manager: BattleSessionManager,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/game/{session_id}/research")
    async def research_socket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            generation, expected_revision = await game_manager.activate_stream(session_id, "research")
        except ServiceError:
            await websocket.close(code=4404)
            return

        try:
            while True:
                frame, expected_revision = await game_manager.research_step(session_id, generation, expected_revision)
                await websocket.send_json(frame)
                if frame["done"]:
                    break
                await asyncio.sleep(0.02)
        except WebSocketDisconnect:
            pass
        finally:
            await game_manager.release_stream(session_id, generation)

    @router.websocket("/ws/game/{session_id}/autoplay")
    async def autoplay_socket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            generation, expected_revision = await game_manager.activate_stream(session_id, "autoplay")
        except ServiceError:
            await websocket.close(code=4404)
            return

        try:
            while True:
                frame, expected_revision = await game_manager.autoplay_step(session_id, generation, expected_revision)
                await websocket.send_json(frame)
                if frame["done"]:
                    break
                await asyncio.sleep(0.15)
        except WebSocketDisconnect:
            pass
        finally:
            await game_manager.release_stream(session_id, generation)

    @router.websocket("/ws/game/{session_id}/ai-move")
    async def ai_move_socket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            generation, expected_revision = await game_manager.activate_stream(session_id, "ai_move")
        except ServiceError:
            await websocket.close(code=4404)
            return

        try:
            while True:
                frame, expected_revision = await game_manager.ai_move_step(session_id, generation, expected_revision)
                await websocket.send_json(frame)
                if frame["done"]:
                    break
                await asyncio.sleep(0.02)
        except WebSocketDisconnect:
            pass
        finally:
            await game_manager.release_stream(session_id, generation)

    @router.websocket("/ws/battle/{session_id}")
    async def battle_socket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            generation, expected_revision = await battle_manager.activate_stream(session_id)
        except ServiceError:
            await websocket.close(code=4404)
            return

        try:
            while True:
                frame, expected_revision = await battle_manager.battle_step(session_id, generation, expected_revision)
                await websocket.send_json(frame)
                if frame["done"]:
                    break
                await asyncio.sleep(0.18)
        except WebSocketDisconnect:
            pass

    return router
