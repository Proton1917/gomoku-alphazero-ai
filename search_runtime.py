from __future__ import annotations

import copy
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from AlphaGomoku.pytorch_graph_model import AlphaGomokuGraphModel
from train import board_size, board_to_tensor, build_model_from_state_dict, evaluation_func, get_runtime_device, show_nn


def normalize_board(board: list[list[int]], current_player: int) -> list[list[int]]:
    normalized = copy.deepcopy(board)
    if current_player == -1:
        for row_index in range(board_size):
            for col_index in range(board_size):
                normalized[row_index][col_index] *= -1
    return normalized


def encode_alpha_gomoku_input(board: list[list[int]], black_to_move: bool) -> torch.Tensor:
    board_array = np.asarray(board, dtype=np.int8)
    own = (board_array == 1).astype(np.float32)
    opp = (board_array == -1).astype(np.float32)
    legal = (board_array == 0).astype(np.float32)
    ones = np.ones_like(legal, dtype=np.float32)
    black = np.full_like(legal, 1.0 if black_to_move else 0.0, dtype=np.float32)
    white = np.full_like(legal, 0.0 if black_to_move else 1.0, dtype=np.float32)
    forbidden = np.zeros_like(legal, dtype=np.float32)
    zeros = np.zeros_like(legal, dtype=np.float32)
    stacked = np.stack([legal, own, opp, ones, black, white, forbidden, zeros], axis=0)
    return torch.from_numpy(stacked)


class CurrentTorchEvaluator:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.device = get_runtime_device()
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model = build_model_from_state_dict(state_dict)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, board: list[list[int]], black_to_move: bool) -> tuple[float, np.ndarray]:
        del black_to_move
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, probs = self.model.calc(x)
        return float(value.item()), probs.squeeze(0).detach().cpu().numpy()

    def nn_overlay(self, board: list[list[int]], black_to_move: bool) -> tuple[np.ndarray, np.ndarray]:
        del black_to_move
        policy_matrix, value_matrix = show_nn(self.model, board)
        return np.asarray(policy_matrix, dtype=np.float32), np.asarray(value_matrix, dtype=np.float32)


class AlphaGomokuEvaluator:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.device = get_runtime_device()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.graph_spec = checkpoint["graph_spec"]
        self.rows = int(self.graph_spec["config"]["rows"])
        self.cols = int(self.graph_spec["config"]["cols"])
        self.model = AlphaGomokuGraphModel(self.graph_spec)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def _forward(
        self,
        board: list[list[int]],
        black_to_move: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(board) != self.rows or len(board[0]) != self.cols:
            raise ValueError(f"AlphaGomoku checkpoint expects {self.rows}x{self.cols}, got {len(board)}x{len(board[0])}")
        x = encode_alpha_gomoku_input(board, black_to_move).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value, action_values, moves_left = self.model(x)
        return (
            policy[0].detach().cpu().numpy(),
            value[0].detach().cpu().numpy(),
            action_values[0].detach().cpu().numpy(),
            moves_left[0].detach().cpu().numpy(),
        )

    def predict(self, board: list[list[int]], black_to_move: bool) -> tuple[float, np.ndarray]:
        policy, value, _action_values, _moves_left = self._forward(board, black_to_move)
        scalar = float(value[0] - value[2])
        return scalar, policy[0]

    def nn_overlay(self, board: list[list[int]], black_to_move: bool) -> tuple[np.ndarray, np.ndarray]:
        policy, _value, action_values, _moves_left = self._forward(board, black_to_move)
        policy_matrix = np.asarray(policy[0], dtype=np.float32).copy()
        value_matrix = np.asarray(action_values[0] - action_values[2], dtype=np.float32).copy()

        board_array = np.asarray(board)
        policy_matrix[board_array != 0] = 0.0
        value_matrix[board_array != 0] = 0.0
        return policy_matrix, value_matrix


def _load_checkpoint(path: str) -> Any:
    return torch.load(path, map_location="cpu")


def is_rapfi_engine_path(model_path: str) -> bool:
    return Path(model_path).name.startswith("pbrain-rapfi")


def create_evaluator(model_path: str):
    checkpoint = _load_checkpoint(model_path)
    if isinstance(checkpoint, dict) and "graph_spec" in checkpoint and "state_dict" in checkpoint:
        return AlphaGomokuEvaluator(model_path)
    return CurrentTorchEvaluator(model_path)


class RapfiProcess:
    def __init__(self, exe_path: str, timeout_ms: int = 6000):
        self.exe_path = Path(exe_path).resolve()
        self.cwd = str(self.exe_path.parent)
        self.timeout_ms = timeout_ms
        self.proc = subprocess.Popen(
            [str(self.exe_path)],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._cmd(f"START {board_size}")
        self._cmd(f"INFO timeout_turn {timeout_ms}")
        self._cmd("INFO timeout_match 0")
        self._cmd("INFO time_left 0")
        self._drain_until(lambda line: line == "OK")
        self.started = False

    def _cmd(self, s: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def _drain_until(self, pred, timeout: float = 50.0) -> list[str]:
        assert self.proc.stdout is not None
        deadline = time.time() + timeout
        lines: list[str] = []
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n").rstrip("\r")
            lines.append(line)
            if pred(line):
                return lines
        raise RuntimeError(f"Rapfi timeout waiting for output, tail={lines[-20:]}")

    def restart(self) -> None:
        self._cmd("RESTART")
        self._drain_until(lambda line: line == "OK")
        self.started = False

    def _board_sync_move(self, board: list[list[int]], move: tuple[int, int]) -> tuple[int, int]:
        if board[move[0]][move[1]] != 0:
            raise RuntimeError(f"Rapfi produced illegal move {move}")
        return move

    def choose_move(self, board: list[list[int]]) -> tuple[tuple[int, int], float]:
        t0 = time.time()
        if not self.started and all(cell == 0 for row in board for cell in row):
            self._cmd("BEGIN")
            lines = self._drain_until(lambda line: re.fullmatch(r"\d+,\d+", line) is not None)
            self.started = True
        else:
            if not self.started:
                self._cmd("BOARD")
                for r in range(board_size):
                    for c in range(board_size):
                        if board[r][c] == 1:
                            self._cmd(f"{c},{r},1")
                        elif board[r][c] == -1:
                            self._cmd(f"{c},{r},2")
                self._cmd("DONE")
            else:
                last_move = None
                for r in range(board_size):
                    for c in range(board_size):
                        if board[r][c] != 0:
                            last_move = (r, c)
                if last_move is None:
                    raise RuntimeError("Rapfi persistent state is inconsistent")
                self._cmd(f"TURN {last_move[1]},{last_move[0]}")
            lines = self._drain_until(lambda line: re.fullmatch(r"\d+,\d+", line) is not None)
            self.started = True

        c, r = map(int, lines[-1].split(","))
        return (r, c), time.time() - t0

    def close(self) -> None:
        try:
            self._cmd("END")
        except Exception:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class RapfiDirectRuntime:
    def __init__(self, exe_path: str, timeout_ms: int = 6000):
        self.process = RapfiProcess(exe_path, timeout_ms=timeout_ms)
        self.last_search_time = 0.0
        self.last_board_after_engine_move: list[list[int]] | None = None

    def _restart_and_sync(self, board: list[list[int]]) -> None:
        self.process.restart()
        if all(cell == 0 for row in board for cell in row):
            self.last_board_after_engine_move = None
            return
        self.process._cmd("BOARD")
        for r in range(board_size):
            for c in range(board_size):
                if board[r][c] == 1:
                    self.process._cmd(f"{c},{r},1")
                elif board[r][c] == -1:
                    self.process._cmd(f"{c},{r},2")
        self.process._cmd("DONE")

    def _find_incremental_opponent_move(self, board: list[list[int]]) -> tuple[int, int] | None:
        if self.last_board_after_engine_move is None:
            return None
        diffs: list[tuple[int, int]] = []
        for r in range(board_size):
            for c in range(board_size):
                prev = self.last_board_after_engine_move[r][c]
                curr = board[r][c]
                if prev != curr:
                    if prev == 0 and curr != 0:
                        diffs.append((r, c))
                    else:
                        return None
        if len(diffs) != 1:
            return None
        return diffs[0]

    def choose_move(self, board: list[list[int]], current_player: int) -> tuple[tuple[int, int], float]:
        incremental_move = self._find_incremental_opponent_move(board)
        if self.last_board_after_engine_move is None and all(cell == 0 for row in board for cell in row):
            t0 = time.time()
            self.process._cmd("BEGIN")
            lines = self.process._drain_until(lambda line: re.fullmatch(r"\d+,\d+", line) is not None)
            c, r = map(int, lines[-1].split(","))
            move = (r, c)
            dt = time.time() - t0
        elif incremental_move is not None:
            t0 = time.time()
            self.process._cmd(f"TURN {incremental_move[1]},{incremental_move[0]}")
            lines = self.process._drain_until(lambda line: re.fullmatch(r"\d+,\d+", line) is not None)
            c, r = map(int, lines[-1].split(","))
            move = (r, c)
            dt = time.time() - t0
        else:
            t0 = time.time()
            self._restart_and_sync(board)
            lines = self.process._drain_until(lambda line: re.fullmatch(r"\d+,\d+", line) is not None)
            c, r = map(int, lines[-1].split(","))
            move = (r, c)
            dt = time.time() - t0
        self.last_search_time = dt
        next_board = copy.deepcopy(board)
        next_board[move[0]][move[1]] = 1 if current_player == 1 else -1
        self.last_board_after_engine_move = next_board
        return move, dt

    def nn_overlay(self) -> tuple[np.ndarray, np.ndarray]:
        policy = np.zeros((board_size, board_size), dtype=np.float32)
        value = np.zeros((board_size, board_size), dtype=np.float32)
        return policy, value

    def close(self) -> None:
        self.process.close()


@dataclass
class SearchNode:
    board: list[list[int]]
    black_to_move: bool
    parent: SearchNode | None = None
    move: tuple[int, int] | None = None
    children: dict[tuple[int, int], tuple[SearchNode | None, float]] | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    value: float | None = None
    val: float = 0.0

    def update_value(self) -> None:
        if not self.children:
            self.val = float(self.value if self.value is not None else 0.0)
        else:
            self.val = self.value_sum / max(self.visit_count, 1)


def root_matches_board(root: SearchNode | None, board: list[list[int]], black_to_move: bool) -> bool:
    if root is None:
        return False
    return root.board == board and root.black_to_move == black_to_move


def advance_root_for_move(root: SearchNode | None, row: int, col: int) -> SearchNode | None:
    if root is None or not root.children:
        return None
    child_info = root.children.get((row, col))
    if child_info is None:
        return None
    child, _prior = child_info
    if child is None:
        return None
    child.parent = None
    return child


def best_move_from_root(root: SearchNode | None) -> list[int] | None:
    if root is None or not root.children:
        return None
    best_move: tuple[int, int] | None = None
    best_visits = -1
    for move, (child, _prior) in root.children.items():
        if child is None:
            continue
        if child.visit_count > best_visits:
            best_visits = child.visit_count
            best_move = move
    return list(best_move) if best_move else None


class GenericMCTS:
    def __init__(self, evaluator, c_puct: float = 0.8, puct2: float = 0.02):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.puct2 = puct2

    def no_child(self, board: list[list[int]]) -> bool:
        return all(cell != 0 for row in board for cell in row)

    def is_terminal(self, board: list[list[int]]) -> bool:
        return self.no_child(board) or evaluation_func(board) != 0

    def evaluate_node(self, node: SearchNode) -> float:
        if self.no_child(node.board):
            return 0.0
        eval_value = evaluation_func(node.board)
        if eval_value != 0:
            return float(eval_value)
        if node.value is None:
            raise RuntimeError("Node value is missing")
        return float(node.value)

    def expand_node(self, node: SearchNode) -> None:
        value, policy = self.evaluator.predict(node.board, node.black_to_move)
        node.value = float(value)
        node.children = {}
        total = 0.0
        for i in range(board_size):
            for j in range(board_size):
                if node.board[i][j] == 0:
                    total += float(policy[i][j])
        if total <= 0.0:
            total = 1e-10
        for i in range(board_size):
            for j in range(board_size):
                if node.board[i][j] == 0:
                    node.children[(i, j)] = (None, float(policy[i][j]) / total)

    def select_child(self, node: SearchNode) -> SearchNode:
        assert node.children
        total_visits = sum((child.visit_count if child is not None else 0) for child, _ in node.children.values())
        explore_buff = np.sqrt(total_visits + 1.0)
        log_total = np.log(total_visits + 1.0)

        exp1 = 0.0
        exp2 = 0.0
        for child, _ in node.children.values():
            if child is not None:
                exp1 += child.val * child.visit_count
                exp2 += child.visit_count
        ave = exp1 / (exp2 + 1e-5)

        best_score = -1e18
        best_move: tuple[int, int] | None = None
        for move, (child, prior) in node.children.items():
            explore = self.c_puct * prior * explore_buff
            exploit = ave
            if child is not None and child.visit_count != 0:
                exploit = child.val
                explore /= (child.visit_count + 1)
            explore += self.puct2 * np.sqrt(log_total / ((child.visit_count if child else 0) + 1))
            score = explore - exploit
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            raise RuntimeError("Failed to select a child")

        child, prior = node.children[best_move]
        if child is None:
            i, j = best_move
            new_board = copy.deepcopy(node.board)
            new_board[i][j] = 1
            for x in range(board_size):
                for y in range(board_size):
                    new_board[x][y] *= -1
            child = SearchNode(
                board=new_board,
                black_to_move=not node.black_to_move,
                parent=node,
                move=best_move,
            )
            node.children[best_move] = (child, prior)
        return child

    def _simulate_once(self, root: SearchNode) -> None:
        node = root
        search_path = [node]

        while node.children:
            node = self.select_child(node)
            search_path.append(node)

        if not self.is_terminal(node.board):
            self.expand_node(node)
        else:
            node.value = self.evaluate_node(node)

        value = self.evaluate_node(node)
        for visited in reversed(search_path):
            visited.visit_count += 1
            visited.value_sum += value
            visited.update_value()
            value = -value

    def _root_to_probs(self, root: SearchNode) -> np.ndarray:
        probs = np.zeros((board_size, board_size), dtype=np.float32)
        total_visits = sum((child.visit_count if child is not None else 0) for child, _ in (root.children or {}).values())
        if total_visits > 0 and root.children:
            for move, (child, _prior) in root.children.items():
                if child is not None:
                    probs[move[0], move[1]] = child.visit_count / total_visits
        return probs

    def _top_moves(self, probs: np.ndarray) -> list[tuple[tuple[int, int], float]]:
        moves: list[tuple[tuple[int, int], float]] = []
        for i in range(board_size):
            for j in range(board_size):
                if probs[i][j] > 0:
                    moves.append(((i, j), float(probs[i][j])))
        moves.sort(key=lambda item: item[1], reverse=True)
        return moves

    def _get_root(self, board: list[list[int]], black_to_move: bool, current_root: SearchNode | None) -> SearchNode:
        if root_matches_board(current_root, board, black_to_move):
            return current_root  # type: ignore[return-value]
        return SearchNode(board=copy.deepcopy(board), black_to_move=black_to_move)

    def run_batch(
        self,
        board: list[list[int]],
        black_to_move: bool,
        simulations: int,
        current_root: SearchNode | None = None,
    ) -> tuple[float, np.ndarray, SearchNode]:
        root = self._get_root(board, black_to_move, current_root)
        for _ in range(simulations):
            self._simulate_once(root)
        probs = self._root_to_probs(root)
        return root.value_sum / max(root.visit_count, 1), probs, root

    def run_until_stable(
        self,
        board: list[list[int]],
        black_to_move: bool,
        min_simulations: int,
        max_simulations: int,
        current_root: SearchNode | None = None,
        batch_size: int = 64,
        stable_batches: int = 4,
        min_top_prob: float = 0.55,
        min_margin: float = 0.08,
    ) -> tuple[float, np.ndarray, SearchNode, dict[str, object]]:
        root = self._get_root(board, black_to_move, current_root)
        stable_count = 0
        last_best_move: tuple[int, int] | None = None
        last_best_prob = 0.0
        last_margin = 0.0
        initial_visits = root.visit_count

        while root.visit_count < max_simulations:
            current_batch = min(batch_size, max_simulations - root.visit_count)
            for _ in range(current_batch):
                self._simulate_once(root)

            probs = self._root_to_probs(root)
            top_moves = self._top_moves(probs)
            if not top_moves:
                break

            best_move, best_prob = top_moves[0]
            second_prob = top_moves[1][1] if len(top_moves) > 1 else 0.0
            margin = best_prob - second_prob

            if best_move == last_best_move:
                stable_count += 1
            else:
                stable_count = 1
                last_best_move = best_move

            last_best_prob = best_prob
            last_margin = margin

            if (
                root.visit_count >= min_simulations
                and stable_count >= stable_batches
                and best_prob >= min_top_prob
                and margin >= min_margin
            ):
                break

        probs = self._root_to_probs(root)
        debug = {
            "used_simulations": root.visit_count,
            "added_simulations": root.visit_count - initial_visits,
            "best_move": last_best_move,
            "best_prob": round(last_best_prob, 6),
            "margin": round(last_margin, 6),
            "stable_batches": stable_count,
        }
        return root.value_sum / max(root.visit_count, 1), probs, root, debug

    def show_data(self, root: SearchNode | None) -> tuple[float, np.ndarray, np.ndarray]:
        if root is None or not root.children:
            return 0.0, np.zeros((board_size, board_size), dtype=np.int32), np.zeros((board_size, board_size), dtype=np.float32)

        visit_matrix = np.zeros((board_size, board_size), dtype=np.int32)
        value_matrix = np.zeros((board_size, board_size), dtype=np.float32)
        for move, (child, _prior) in root.children.items():
            if child is None:
                continue
            visit_matrix[move[0], move[1]] = child.visit_count
            value_matrix[move[0], move[1]] = -child.val
        return float(root.val), visit_matrix, value_matrix


class ModelSearchRuntime:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_direct_engine = is_rapfi_engine_path(model_path)
        if self.is_direct_engine:
            self.direct_engine = RapfiDirectRuntime(model_path, timeout_ms=1000)
            self.evaluator = None
            self.mcts = None
        else:
            self.direct_engine = None
            self.evaluator = create_evaluator(model_path)
            self.mcts = GenericMCTS(self.evaluator)

    def choose_move(self, board: list[list[int]], current_player: int) -> tuple[tuple[int, int], float]:
        if self.direct_engine is None:
            raise RuntimeError("choose_move() is only available for direct engine runtimes")
        return self.direct_engine.choose_move(board, current_player)

    def run_batch(
        self,
        board: list[list[int]],
        current_player: int,
        simulations: int,
        current_root: SearchNode | None = None,
    ) -> tuple[float, np.ndarray, SearchNode]:
        if self.mcts is None:
            raise RuntimeError("run_batch() is not supported for direct engine runtimes")
        normalized = normalize_board(board, current_player)
        black_to_move = current_player == 1
        return self.mcts.run_batch(normalized, black_to_move, simulations, current_root)

    def run_until_stable(
        self,
        board: list[list[int]],
        current_player: int,
        min_simulations: int,
        max_simulations: int,
        current_root: SearchNode | None = None,
        batch_size: int = 64,
        stable_batches: int = 4,
        min_top_prob: float = 0.55,
        min_margin: float = 0.08,
    ) -> tuple[float, np.ndarray, SearchNode, dict[str, object]]:
        if self.mcts is None:
            raise RuntimeError("run_until_stable() is not supported for direct engine runtimes")
        normalized = normalize_board(board, current_player)
        black_to_move = current_player == 1
        return self.mcts.run_until_stable(
            normalized,
            black_to_move,
            min_simulations=min_simulations,
            max_simulations=max_simulations,
            current_root=current_root,
            batch_size=batch_size,
            stable_batches=stable_batches,
            min_top_prob=min_top_prob,
            min_margin=min_margin,
        )

    def nn_overlay(self, board: list[list[int]], current_player: int) -> tuple[np.ndarray, np.ndarray]:
        if self.direct_engine is not None:
            return self.direct_engine.nn_overlay()
        normalized = normalize_board(board, current_player)
        black_to_move = current_player == 1
        return self.evaluator.nn_overlay(normalized, black_to_move)

    def show_data(self, root: SearchNode | None) -> tuple[float, np.ndarray, np.ndarray]:
        if self.mcts is None:
            return 0.0, np.zeros((board_size, board_size), dtype=np.int32), np.zeros((board_size, board_size), dtype=np.float32)
        return self.mcts.show_data(root)
