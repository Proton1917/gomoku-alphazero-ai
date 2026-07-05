from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from db import Database
from go_rules import (
    BOARD_SIZE,
    PASS_MOVE,
    RESIGN_MOVE,
    apply_go_move,
    board_key,
    board_full,
    empty_board,
    is_pass,
    is_resign,
    score_area,
)
from katago_runtime import ModelSearchRuntime
from models_service import get_default_battle_model_paths, normalize_model_path


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def score_for_state(status: str, board: list[list[int]], last_move: list[int] | None) -> dict[str, float | int] | None:
    if status != "finished" or (last_move is not None and is_resign(last_move)):
        return None
    return score_area(board).to_dict()


class ServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class GameSession:
    id: str
    board: list[list[int]]
    current_player: int
    move_history: list[dict[str, Any]]
    history_index: int
    status: str
    winner: int | None
    model_path: str
    simulations: int
    last_move: list[int] | None
    revision: int
    created_at: str
    updated_at: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    stream_generation: int = 0
    active_stream: str | None = None

    def __post_init__(self) -> None:
        self.search_runtime = ModelSearchRuntime(self.model_path, max_visits=self.simulations)

    @classmethod
    def new(cls, model_path: str, simulations: int) -> "GameSession":
        now = iso_now()
        return cls(
            id=str(uuid.uuid4()),
            board=empty_board(),
            current_player=1,
            move_history=[],
            history_index=-1,
            status="active",
            winner=None,
            model_path=model_path,
            simulations=simulations,
            last_move=None,
            revision=0,
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "GameSession":
        board = record["board"]
        if len(board) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in board):
            board = empty_board()
            move_history: list[dict[str, Any]] = []
            history_index = -1
            current_player = 1
            status = "active"
            winner = None
            last_move = None
        else:
            move_history = record["move_history"]
            history_index = record["history_index"]
            current_player = record["current_player"]
            status = record["status"]
            winner = record["winner"]
            last_move = record["last_move"]

        return cls(
            id=record["id"],
            board=board,
            current_player=current_player,
            move_history=move_history,
            history_index=history_index,
            status=status,
            winner=winner,
            model_path=record["model_path"],
            simulations=record["simulations"],
            last_move=last_move,
            revision=record["revision"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "board": self.board,
            "current_player": self.current_player,
            "move_history": self.move_history,
            "history_index": self.history_index,
            "status": self.status,
            "winner": self.winner,
            "model_path": self.model_path,
            "simulations": self.simulations,
            "last_move": self.last_move,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "board": self.board,
            "current_player": self.current_player,
            "move_history": self.move_history,
            "history_index": self.history_index,
            "status": self.status,
            "winner": self.winner,
            "model_path": self.model_path,
            "simulations": self.simulations,
            "last_move": self.last_move,
            "search_visits": int(getattr(self.search_runtime, "last_search_visits", 0)),
            "can_undo": self.history_index >= 0,
            "can_redo": self.history_index < len(self.move_history) - 1,
            "score": score_for_state(self.status, self.board, self.last_move),
        }


@dataclass
class BattleSession:
    id: str
    board: list[list[int]]
    current_player: int
    move_history: list[dict[str, Any]]
    move_count: int
    status: str
    winner: int | None
    black_model_path: str
    white_model_path: str
    simulations: int
    last_move: list[int] | None
    revision: int
    created_at: str
    updated_at: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    stream_generation: int = 0

    def __post_init__(self) -> None:
        self.black_search = ModelSearchRuntime(self.black_model_path, max_visits=self.simulations)
        self.white_search = ModelSearchRuntime(self.white_model_path, max_visits=self.simulations)

    @classmethod
    def new(cls, black_model_path: str, white_model_path: str, simulations: int) -> "BattleSession":
        now = iso_now()
        return cls(
            id=str(uuid.uuid4()),
            board=empty_board(),
            current_player=1,
            move_history=[],
            move_count=0,
            status="active",
            winner=None,
            black_model_path=black_model_path,
            white_model_path=white_model_path,
            simulations=simulations,
            last_move=None,
            revision=0,
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BattleSession":
        board = record["board"]
        if len(board) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in board):
            board = empty_board()
            move_history: list[dict[str, Any]] = []
            move_count = 0
            current_player = 1
            status = "active"
            winner = None
            last_move = None
        else:
            move_history = record["move_history"]
            move_count = record["move_count"]
            current_player = record["current_player"]
            status = record["status"]
            winner = record["winner"]
            last_move = record["last_move"]

        return cls(
            id=record["id"],
            board=board,
            current_player=current_player,
            move_history=move_history,
            move_count=move_count,
            status=status,
            winner=winner,
            black_model_path=record["black_model_path"],
            white_model_path=record["white_model_path"],
            simulations=record["simulations"],
            last_move=last_move,
            revision=record["revision"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "board": self.board,
            "current_player": self.current_player,
            "move_history": self.move_history,
            "move_count": self.move_count,
            "status": self.status,
            "winner": self.winner,
            "black_model_path": self.black_model_path,
            "white_model_path": self.white_model_path,
            "simulations": self.simulations,
            "last_move": self.last_move,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "board": self.board,
            "current_player": self.current_player,
            "move_history": self.move_history,
            "move_count": self.move_count,
            "status": self.status,
            "winner": self.winner,
            "black_model_path": self.black_model_path,
            "white_model_path": self.white_model_path,
            "simulations": self.simulations,
            "last_move": self.last_move,
            "score": score_for_state(self.status, self.board, self.last_move),
        }


class GameSessionManager:
    def __init__(self, db: Database):
        self.db = db
        self.sessions: dict[str, GameSession] = {}

    async def create_session(self, model_path: str | None, simulations: int) -> dict[str, Any]:
        try:
            normalized_model = normalize_model_path(model_path)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc

        session = GameSession.new(normalized_model, simulations)
        self.sessions[session.id] = session
        self.db.save_game(session.to_record())
        return session.to_state()

    async def get_state(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        return session.to_state()

    async def make_human_move(self, session_id: str, row: int, col: int) -> dict[str, Any]:
        session = self._get_session(session_id)
        async with session.lock:
            self._ensure_active(session)
            self._apply_new_move(session, row, col, session.current_player)
            self._persist_game(session)
            return session.to_state()

    async def pass_move(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        async with session.lock:
            self._ensure_active(session)
            self._apply_new_move(session, PASS_MOVE[0], PASS_MOVE[1], session.current_player)
            self._persist_game(session)
            return session.to_state()

    async def resign(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        async with session.lock:
            self._ensure_active(session)
            self._apply_new_move(session, RESIGN_MOVE[0], RESIGN_MOVE[1], session.current_player)
            self._persist_game(session)
            return session.to_state()

    async def ai_move(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        async with session.lock:
            self._ensure_active(session)
            move = await self._choose_engine_move(session, session.search_runtime)
            self._apply_new_move(session, move[0], move[1], session.current_player)
            self._persist_game(session)
            return session.to_state()

    async def undo(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        async with session.lock:
            if session.history_index < 0:
                raise ServiceError(400, "当前没有可悔棋的步数")
            session.history_index -= 1
            self._rebuild_position(session)
            session.revision += 1
            self._persist_game(session)
            return session.to_state()

    async def redo(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        async with session.lock:
            if session.history_index >= len(session.move_history) - 1:
                raise ServiceError(400, "当前没有可前进的步数")
            session.history_index += 1
            self._rebuild_position(session)
            session.revision += 1
            self._persist_game(session)
            return session.to_state()

    async def nn_predictions(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        async with session.lock:
            policy_matrix, value_matrix = await asyncio.to_thread(
                session.search_runtime.nn_overlay,
                session.board,
                session.current_player,
            )
            return {
                "policy_matrix": np.asarray(policy_matrix).tolist(),
                "value_matrix": np.asarray(value_matrix).tolist(),
                "current_player": session.current_player,
            }

    async def activate_stream(self, session_id: str, stream_name: str) -> tuple[int, int]:
        session = self._get_session(session_id)
        async with session.lock:
            session.stream_generation += 1
            session.active_stream = stream_name
            return session.stream_generation, session.revision

    async def release_stream(self, session_id: str, generation: int) -> None:
        session = self._get_session(session_id)
        async with session.lock:
            if session.stream_generation == generation:
                session.active_stream = None

    async def research_step(self, session_id: str, generation: int, expected_revision: int) -> tuple[dict[str, Any], int]:
        session = self._get_session(session_id)
        async with session.lock:
            reason = self._stream_stop_reason(session, generation, expected_revision)
            if reason is not None:
                return self._research_frame(session, True, reason), session.revision
            return self._research_frame(session, True, "katago_direct_engine"), session.revision

    async def autoplay_step(self, session_id: str, generation: int, expected_revision: int) -> tuple[dict[str, Any], int]:
        session = self._get_session(session_id)
        async with session.lock:
            reason = self._stream_stop_reason(session, generation, expected_revision)
            if reason is not None:
                return self._autoplay_frame(session, True, reason), session.revision
            if session.status != "active":
                return self._autoplay_frame(session, True, "game_finished"), session.revision
            move = await self._choose_engine_move(session, session.search_runtime)
            self._apply_new_move(session, move[0], move[1], session.current_player)
            self._persist_game(session)
            done = session.status != "active"
            return self._autoplay_frame(session, done, "game_finished" if done else None), session.revision

    async def ai_move_step(self, session_id: str, generation: int, expected_revision: int) -> tuple[dict[str, Any], int]:
        session = self._get_session(session_id)
        async with session.lock:
            reason = self._stream_stop_reason(session, generation, expected_revision)
            if reason is not None:
                return self._ai_move_frame(session, True, reason), session.revision
            if session.status != "active":
                return self._ai_move_frame(session, True, "game_finished"), session.revision

            move = await self._choose_engine_move(session, session.search_runtime)
            self._apply_new_move(session, move[0], move[1], session.current_player)
            self._persist_game(session)
            return self._ai_move_frame(session, True, "move_applied"), session.revision

    def _get_session(self, session_id: str) -> GameSession:
        session = self.sessions.get(session_id)
        if session:
            return session
        record = self.db.get_game(session_id)
        if record is None:
            raise ServiceError(404, "游戏会话不存在")
        try:
            session = GameSession.from_record(record)
        except Exception as exc:
            raise ServiceError(400, f"恢复游戏会话失败: {exc}") from exc
        self.sessions[session_id] = session
        return session

    def _ensure_active(self, session: GameSession) -> None:
        if session.status != "active":
            raise ServiceError(400, "对局已结束，无法继续操作")

    def _persist_game(self, session: GameSession) -> None:
        session.updated_at = iso_now()
        self.db.save_game(session.to_record())

    async def _choose_engine_move(self, session: GameSession, search_runtime: ModelSearchRuntime) -> tuple[int, int]:
        try:
            move, _dt = await asyncio.to_thread(
                search_runtime.choose_move,
                session.board,
                session.current_player,
                session.move_history[: session.history_index + 1],
            )
        except Exception as exc:
            raise ServiceError(500, f"KataGo 落子失败: {exc}") from exc
        return move

    def _apply_new_move(self, session: GameSession, row: int, col: int, player: int) -> None:
        move = [row, col]
        if is_resign(move):
            pass
        elif is_pass(move):
            pass
        else:
            try:
                result = apply_go_move(session.board, row, col, player)
            except ValueError as exc:
                raise ServiceError(400, str(exc)) from exc
            if self._repeats_previous_position(session, result.board):
                raise ServiceError(400, "打劫/全局同形：该手会重复旧局面")

        session.move_history = session.move_history[: session.history_index + 1]
        session.move_history.append({"move": move, "player": player})
        session.history_index += 1
        self._rebuild_position(session)
        session.revision += 1

    def _repeats_previous_position(self, session: GameSession, candidate_board: list[list[int]]) -> bool:
        return board_key(candidate_board) in self._position_keys(session.move_history[: session.history_index + 1])

    def _position_keys(self, move_history: list[dict[str, Any]]) -> set[str]:
        board = empty_board()
        keys = {board_key(board)}
        for record in move_history:
            move = list(record["move"])
            if is_pass(move) or is_resign(move):
                continue
            result = apply_go_move(board, int(move[0]), int(move[1]), int(record["player"]))
            board = result.board
            keys.add(board_key(board))
        return keys

    def _rebuild_position(self, session: GameSession) -> None:
        board = empty_board()
        current_player = 1
        winner: int | None = None
        status = "active"
        last_move: list[int] | None = None
        consecutive_passes = 0

        for record in session.move_history[: session.history_index + 1]:
            player = int(record["player"])
            move = list(record["move"])
            last_move = move

            if is_resign(move):
                status = "finished"
                winner = -player
                current_player = player
                break

            if is_pass(move):
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    score = score_area(board)
                    status = "finished"
                    winner = score.winner
                    current_player = player
                    break
                current_player = -player
                continue

            consecutive_passes = 0
            result = apply_go_move(board, int(move[0]), int(move[1]), player)
            board = result.board

            if board_full(board):
                score = score_area(board)
                status = "finished"
                winner = score.winner
                current_player = player
                break

            current_player = -player

        session.board = board
        session.current_player = current_player
        session.status = status
        session.winner = winner
        session.last_move = last_move

    def _research_frame(self, session: GameSession, done: bool, reason: str | None) -> dict[str, Any]:
        return {
            "type": "research_update",
            "game": session.to_state(),
            "visit_count": int(getattr(session.search_runtime, "last_search_visits", 0)),
            "value": 0.0,
            "visit_matrix": np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
            "value_matrix": np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
            "done": done,
            "reason": reason,
        }

    def _autoplay_frame(self, session: GameSession, done: bool, reason: str | None) -> dict[str, Any]:
        return {
            "type": "autoplay_update",
            "game": session.to_state(),
            "done": done,
            "reason": reason,
        }

    def _ai_move_frame(self, session: GameSession, done: bool, reason: str | None) -> dict[str, Any]:
        return {
            "type": "ai_move_update",
            "game": session.to_state(),
            "visit_count": int(getattr(session.search_runtime, "last_search_visits", 0)),
            "value": 0.0,
            "visit_matrix": np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
            "value_matrix": np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
            "done": done,
            "reason": reason,
        }

    def _stream_stop_reason(
        self,
        session: GameSession,
        generation: int,
        expected_revision: int,
    ) -> str | None:
        if session.stream_generation != generation:
            if session.active_stream is None:
                return "stream_closed"
            return "stream_superseded"
        if session.revision != expected_revision:
            return "state_changed"
        return None


class BattleSessionManager:
    def __init__(self, db: Database):
        self.db = db
        self.sessions: dict[str, BattleSession] = {}

    async def create_session(
        self,
        black_model_path: str | None,
        white_model_path: str | None,
        simulations: int,
    ) -> dict[str, Any]:
        default_black, default_white = get_default_battle_model_paths()
        try:
            normalized_black = normalize_model_path(black_model_path or default_black)
            normalized_white = normalize_model_path(white_model_path or default_white)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc

        session = BattleSession.new(normalized_black, normalized_white, simulations)
        self.sessions[session.id] = session
        self.db.save_battle(session.to_record())
        return session.to_state()

    async def get_state(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        return session.to_state()

    async def activate_stream(self, session_id: str) -> tuple[int, int]:
        session = self._get_session(session_id)
        async with session.lock:
            session.stream_generation += 1
            return session.stream_generation, session.revision

    async def battle_step(self, session_id: str, generation: int, expected_revision: int) -> tuple[dict[str, Any], int]:
        session = self._get_session(session_id)
        async with session.lock:
            if session.stream_generation != generation:
                return self._battle_frame(session, True, "stream_superseded"), session.revision
            if session.revision != expected_revision:
                return self._battle_frame(session, True, "state_changed"), session.revision
            if session.status != "active":
                return self._battle_frame(session, True, "game_finished"), session.revision
            move = await self._compute_ai_move(session)
            self._apply_move(session, move[0], move[1], session.current_player)
            self._persist_battle(session)
            done = session.status != "active"
            return self._battle_frame(session, done, "game_finished" if done else None), session.revision

    def _get_session(self, session_id: str) -> BattleSession:
        session = self.sessions.get(session_id)
        if session:
            return session
        record = self.db.get_battle(session_id)
        if record is None:
            raise ServiceError(404, "对战会话不存在")
        try:
            session = BattleSession.from_record(record)
        except Exception as exc:
            raise ServiceError(400, f"恢复对战会话失败: {exc}") from exc
        self.sessions[session_id] = session
        return session

    def _persist_battle(self, session: BattleSession) -> None:
        session.updated_at = iso_now()
        self.db.save_battle(session.to_record())

    async def _compute_ai_move(self, session: BattleSession) -> tuple[int, int]:
        search_runtime = session.black_search if session.current_player == 1 else session.white_search
        try:
            move, _dt = await asyncio.to_thread(
                search_runtime.choose_move,
                session.board,
                session.current_player,
                session.move_history,
            )
        except Exception as exc:
            raise ServiceError(500, f"KataGo 对战落子失败: {exc}") from exc
        return move

    def _apply_move(self, session: BattleSession, row: int, col: int, player: int) -> None:
        move = [row, col]
        if is_resign(move):
            pass
        elif is_pass(move):
            pass
        else:
            try:
                result = apply_go_move(session.board, row, col, player)
            except ValueError as exc:
                raise ServiceError(400, str(exc)) from exc
            if self._repeats_previous_battle_position(session, result.board):
                raise ServiceError(400, "打劫/全局同形：该手会重复旧局面")

        session.move_history.append({"move": move, "player": player})
        session.move_count += 1
        session.revision += 1
        self._rebuild_battle_position(session)

    def _repeats_previous_battle_position(self, session: BattleSession, candidate_board: list[list[int]]) -> bool:
        return board_key(candidate_board) in self._battle_position_keys(session.move_history)

    def _battle_position_keys(self, move_history: list[dict[str, Any]]) -> set[str]:
        board = empty_board()
        keys = {board_key(board)}
        for record in move_history:
            move = list(record["move"])
            if is_pass(move) or is_resign(move):
                continue
            result = apply_go_move(board, int(move[0]), int(move[1]), int(record["player"]))
            board = result.board
            keys.add(board_key(board))
        return keys

    def _rebuild_battle_position(self, session: BattleSession) -> None:
        board = empty_board()
        current_player = 1
        winner: int | None = None
        status = "active"
        last_move: list[int] | None = None
        consecutive_passes = 0

        for record in session.move_history:
            player = int(record["player"])
            move = list(record["move"])
            last_move = move

            if is_resign(move):
                status = "finished"
                winner = -player
                current_player = player
                break
            if is_pass(move):
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    score = score_area(board)
                    status = "finished"
                    winner = score.winner
                    current_player = player
                    break
                current_player = -player
                continue

            consecutive_passes = 0
            result = apply_go_move(board, int(move[0]), int(move[1]), player)
            board = result.board
            if board_full(board):
                score = score_area(board)
                status = "finished"
                winner = score.winner
                current_player = player
                break
            current_player = -player

        session.board = board
        session.current_player = current_player
        session.status = status
        session.winner = winner
        session.last_move = last_move

    def _battle_frame(self, session: BattleSession, done: bool, reason: str | None) -> dict[str, Any]:
        return {
            "type": "battle_update",
            "battle": session.to_state(),
            "done": done,
            "reason": reason,
        }
