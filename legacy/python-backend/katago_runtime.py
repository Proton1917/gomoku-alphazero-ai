from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from go_rules import BOARD_SIZE, PASS_MOVE, RESIGN_MOVE
from models_service import KATAGO_CONFIG_PATH, KATAGO_EXECUTABLE


GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"


class KataGoRuntimeError(RuntimeError):
    pass


class KataGoProcess:
    def __init__(self, model_path: str, max_visits: int):
        self.model_path = str(Path(model_path).resolve())
        self.max_visits = max(1, min(int(max_visits), 2000))
        self.proc: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] | None = None
        self.lock = threading.Lock()
        self.last_search_time = 0.0
        self.last_search_visits = 0

    def choose_move(
        self,
        board: list[list[int]],
        current_player: int,
        move_history: list[dict[str, Any]] | None = None,
    ) -> tuple[tuple[int, int], float]:
        del board
        with self.lock:
            started = time.time()
            self._ensure_running()
            self._load_position(move_history or [])
            color = "B" if current_player == 1 else "W"
            response = self._command(f"genmove {color}", timeout=120.0)
            move = self._parse_gtp_move(response.strip())
            self.last_search_time = time.time() - started
            self.last_search_visits = self.max_visits
            return move, self.last_search_time

    def nn_overlay(self) -> tuple[np.ndarray, np.ndarray]:
        policy = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        value = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        return policy, value

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    def _ensure_running(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return

        if not KATAGO_EXECUTABLE.exists():
            raise KataGoRuntimeError(f"未找到 KataGo 可执行文件: {KATAGO_EXECUTABLE}")
        if not KATAGO_CONFIG_PATH.exists():
            raise KataGoRuntimeError(f"未找到 KataGo GTP 配置: {KATAGO_CONFIG_PATH}")
        if not Path(self.model_path).exists():
            raise KataGoRuntimeError(f"未找到 KataGo 模型: {self.model_path}")

        self.proc = subprocess.Popen(
            [
                str(KATAGO_EXECUTABLE),
                "gtp",
                "-model",
                self.model_path,
                "-config",
                str(KATAGO_CONFIG_PATH),
                "-override-config",
                f"maxVisits={self.max_visits},logAllGTPCommunication=false",
            ],
            cwd=str(KATAGO_EXECUTABLE.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.output_queue = queue.Queue()
        threading.Thread(target=self._read_stdout, args=(self.proc, self.output_queue), daemon=True).start()
        self._command("boardsize 19", timeout=60.0)
        self._command("komi 6.5", timeout=10.0)

    def _read_stdout(self, proc: subprocess.Popen[str], output_queue: queue.Queue[str]) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            output_queue.put(line.rstrip("\n").rstrip("\r"))

    def _load_position(self, move_history: list[dict[str, Any]]) -> None:
        self._command("clear_board", timeout=10.0)
        for record in move_history:
            move = list(record.get("move", []))
            player = int(record.get("player", 1))
            color = "B" if player == 1 else "W"
            if move == PASS_MOVE:
                coord = "pass"
            elif move == RESIGN_MOVE:
                break
            elif len(move) == 2:
                coord = board_to_gtp(int(move[0]), int(move[1]))
            else:
                continue
            self._command(f"play {color} {coord}", timeout=10.0)

    def _command(self, command: str, timeout: float) -> str:
        proc = self.proc
        output_queue = self.output_queue
        if proc is None or proc.stdin is None or output_queue is None:
            raise KataGoRuntimeError("KataGo 进程未启动")

        proc.stdin.write(command + "\n")
        proc.stdin.flush()

        deadline = time.time() + timeout
        started = False
        status = ""
        payload: list[str] = []
        noise_tail: list[str] = []

        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            try:
                line = output_queue.get(timeout=min(0.2, remaining))
            except queue.Empty:
                if proc.poll() is not None:
                    raise KataGoRuntimeError(f"KataGo 已退出，命令失败: {command}")
                continue
            clean = line.strip()

            if clean.startswith("=") or clean.startswith("?"):
                started = True
                status = clean[0]
                first_payload = clean[1:].strip()
                if first_payload:
                    payload.append(first_payload)
                continue

            if started:
                if clean == "":
                    if status == "?":
                        raise KataGoRuntimeError(f"KataGo 拒绝命令 {command}: {' '.join(payload)}")
                    return "\n".join(payload)
                payload.append(clean)
                continue

            if clean:
                noise_tail.append(clean)
                noise_tail = noise_tail[-12:]

        self.close()
        raise KataGoRuntimeError(f"KataGo 响应超时: {command}; tail={noise_tail}")

    def _parse_gtp_move(self, raw: str) -> tuple[int, int]:
        normalized = raw.strip().lower()
        if normalized == "pass":
            return PASS_MOVE[0], PASS_MOVE[1]
        if normalized == "resign":
            return RESIGN_MOVE[0], RESIGN_MOVE[1]
        return gtp_to_board(raw)


class ModelSearchRuntime:
    def __init__(self, model_path: str, max_visits: int = 128):
        self.model_path = model_path
        self.is_direct_engine = True
        self.direct_engine = KataGoProcess(model_path, max_visits=max_visits)

    @property
    def last_search_visits(self) -> int:
        return self.direct_engine.last_search_visits

    def choose_move(
        self,
        board: list[list[int]],
        current_player: int,
        move_history: list[dict[str, Any]] | None = None,
    ) -> tuple[tuple[int, int], float]:
        return self.direct_engine.choose_move(board, current_player, move_history)

    def nn_overlay(self, board: list[list[int]], current_player: int) -> tuple[np.ndarray, np.ndarray]:
        del board, current_player
        return self.direct_engine.nn_overlay()

    def show_data(self, root: object | None = None) -> tuple[float, np.ndarray, np.ndarray]:
        del root
        return (
            0.0,
            np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32),
            np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        )

    def close(self) -> None:
        self.direct_engine.close()


def board_to_gtp(row: int, col: int) -> str:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"坐标超出 19 路棋盘: {row}, {col}")
    return f"{GTP_COLUMNS[col]}{BOARD_SIZE - row}"


def gtp_to_board(move: str) -> tuple[int, int]:
    normalized = move.strip().upper()
    if len(normalized) < 2:
        raise KataGoRuntimeError(f"KataGo 返回了无法解析的坐标: {move}")
    col = GTP_COLUMNS.find(normalized[0])
    if col < 0:
        raise KataGoRuntimeError(f"KataGo 返回了无法解析的列坐标: {move}")
    try:
        gtp_row = int(normalized[1:])
    except ValueError as exc:
        raise KataGoRuntimeError(f"KataGo 返回了无法解析的行坐标: {move}") from exc
    row = BOARD_SIZE - gtp_row
    if not (0 <= row < BOARD_SIZE):
        raise KataGoRuntimeError(f"KataGo 返回了越界坐标: {move}")
    return row, col
