from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS game_sessions (
                    id TEXT PRIMARY KEY,
                    board TEXT NOT NULL,
                    current_player INTEGER NOT NULL,
                    move_history TEXT NOT NULL,
                    history_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    winner INTEGER,
                    model_path TEXT NOT NULL,
                    simulations INTEGER NOT NULL,
                    last_move TEXT,
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS battle_sessions (
                    id TEXT PRIMARY KEY,
                    board TEXT NOT NULL,
                    current_player INTEGER NOT NULL,
                    move_history TEXT NOT NULL,
                    move_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    winner INTEGER,
                    black_model_path TEXT NOT NULL,
                    white_model_path TEXT NOT NULL,
                    simulations INTEGER NOT NULL,
                    last_move TEXT,
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def save_game(self, record: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO game_sessions (
                    id, board, current_player, move_history, history_index, status,
                    winner, model_path, simulations, last_move, revision, created_at, updated_at
                ) VALUES (
                    :id, :board, :current_player, :move_history, :history_index, :status,
                    :winner, :model_path, :simulations, :last_move, :revision, :created_at, :updated_at
                )
                ON CONFLICT(id) DO UPDATE SET
                    board = excluded.board,
                    current_player = excluded.current_player,
                    move_history = excluded.move_history,
                    history_index = excluded.history_index,
                    status = excluded.status,
                    winner = excluded.winner,
                    model_path = excluded.model_path,
                    simulations = excluded.simulations,
                    last_move = excluded.last_move,
                    revision = excluded.revision,
                    updated_at = excluded.updated_at
                """,
                self._encode_record(record),
            )

    def save_battle(self, record: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO battle_sessions (
                    id, board, current_player, move_history, move_count, status,
                    winner, black_model_path, white_model_path, simulations,
                    last_move, revision, created_at, updated_at
                ) VALUES (
                    :id, :board, :current_player, :move_history, :move_count, :status,
                    :winner, :black_model_path, :white_model_path, :simulations,
                    :last_move, :revision, :created_at, :updated_at
                )
                ON CONFLICT(id) DO UPDATE SET
                    board = excluded.board,
                    current_player = excluded.current_player,
                    move_history = excluded.move_history,
                    move_count = excluded.move_count,
                    status = excluded.status,
                    winner = excluded.winner,
                    black_model_path = excluded.black_model_path,
                    white_model_path = excluded.white_model_path,
                    simulations = excluded.simulations,
                    last_move = excluded.last_move,
                    revision = excluded.revision,
                    updated_at = excluded.updated_at
                """,
                self._encode_record(record),
            )

    def get_game(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM game_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return self._decode_row(row) if row else None

    def get_battle(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM battle_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return self._decode_row(row) if row else None

    def _encode_record(self, record: dict[str, Any]) -> dict[str, Any]:
        encoded = dict(record)
        for key in ("board", "move_history", "last_move"):
            encoded[key] = json.dumps(encoded[key]) if encoded.get(key) is not None else None
        return encoded

    def _decode_row(self, row: sqlite3.Row) -> dict[str, Any]:
        decoded = dict(row)
        for key in ("board", "move_history", "last_move"):
            decoded[key] = json.loads(decoded[key]) if decoded.get(key) is not None else None
        return decoded
