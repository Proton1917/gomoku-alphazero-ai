from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import itertools
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from train import board_size, evaluation_func


@dataclasses.dataclass(frozen=True)
class MatchTask:
    round_index: int
    black_budget_sec: int
    white_budget_sec: int


@dataclasses.dataclass
class MatchResult:
    round_index: int
    black_budget_sec: int
    white_budget_sec: int
    winner: int
    move_count: int
    black_avg_s: float
    white_avg_s: float
    wall_time_s: float


class RapfiEngine:
    def __init__(self, exe_path: str, cwd: str, timeout_ms: int):
        self.proc = subprocess.Popen(
            [exe_path],
            cwd=cwd,
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
        self._drain_until(lambda line: line == "OK", timeout=30.0)

    def _cmd(self, s: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def _drain_until(self, pred, timeout: float) -> list[str]:
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
        raise RuntimeError(f"Rapfi timeout waiting for output. tail={lines[-20:]}")

    def restart_game(self) -> None:
        self._cmd("RESTART")
        self._drain_until(lambda line: line == "OK", timeout=30.0)

    def begin(self) -> tuple[tuple[int, int], float]:
        t0 = time.time()
        self._cmd("BEGIN")
        lines = self._drain_until(lambda line: re.fullmatch(r"\d+,\d+", line) is not None, timeout=120.0)
        col, row = map(int, lines[-1].split(","))
        return (row, col), time.time() - t0

    def turn(self, move: tuple[int, int]) -> tuple[tuple[int, int], float]:
        t0 = time.time()
        row, col = move
        self._cmd(f"TURN {col},{row}")
        lines = self._drain_until(lambda line: re.fullmatch(r"\d+,\d+", line) is not None, timeout=120.0)
        col2, row2 = map(int, lines[-1].split(","))
        return (row2, col2), time.time() - t0

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


def build_round_robin_tasks(budgets: list[int]) -> list[MatchTask]:
    players: list[int | None] = budgets[:]
    if len(players) % 2 == 1:
        players.append(None)

    n = len(players)
    rounds = n - 1
    tasks: list[MatchTask] = []

    for round_index in range(rounds):
        for pair_index in range(n // 2):
            left = players[pair_index]
            right = players[n - 1 - pair_index]
            if left is None or right is None:
                continue
            if round_index % 2 == 0:
                black, white = left, right
            else:
                black, white = right, left
            tasks.append(
                MatchTask(
                    round_index=round_index + 1,
                    black_budget_sec=int(black),
                    white_budget_sec=int(white),
                )
            )
        players = [players[0]] + [players[-1]] + players[1:-1]

    return tasks


def run_single_match(task: MatchTask, exe_path: str, cwd: str) -> MatchResult:
    black = RapfiEngine(exe_path, cwd, timeout_ms=task.black_budget_sec * 1000)
    white = RapfiEngine(exe_path, cwd, timeout_ms=task.white_budget_sec * 1000)

    board = [[0] * board_size for _ in range(board_size)]
    move_count = 0
    current_player = 1
    black_times: list[float] = []
    white_times: list[float] = []
    wall_start = time.time()

    try:
        move, dt = black.begin()
        black_times.append(dt)
        board[move[0]][move[1]] = 1
        move_count += 1
        last_move = move
        terminal = evaluation_func(board)
        if terminal != 0:
            return MatchResult(
                round_index=task.round_index,
                black_budget_sec=task.black_budget_sec,
                white_budget_sec=task.white_budget_sec,
                winner=1,
                move_count=move_count,
                black_avg_s=sum(black_times) / len(black_times),
                white_avg_s=0.0,
                wall_time_s=time.time() - wall_start,
            )

        current_player = -1
        while move_count < board_size * board_size:
            if current_player == -1:
                move, dt = white.turn(last_move)
                white_times.append(dt)
            else:
                move, dt = black.turn(last_move)
                black_times.append(dt)

            if board[move[0]][move[1]] != 0:
                raise RuntimeError(f"Illegal move {move} at ply {move_count + 1}")
            board[move[0]][move[1]] = current_player
            move_count += 1
            last_move = move
            terminal = evaluation_func(board)
            if terminal != 0:
                return MatchResult(
                    round_index=task.round_index,
                    black_budget_sec=task.black_budget_sec,
                    white_budget_sec=task.white_budget_sec,
                    winner=current_player,
                    move_count=move_count,
                    black_avg_s=sum(black_times) / max(len(black_times), 1),
                    white_avg_s=sum(white_times) / max(len(white_times), 1),
                    wall_time_s=time.time() - wall_start,
                )
            current_player = -current_player

        return MatchResult(
            round_index=task.round_index,
            black_budget_sec=task.black_budget_sec,
            white_budget_sec=task.white_budget_sec,
            winner=0,
            move_count=move_count,
            black_avg_s=sum(black_times) / max(len(black_times), 1),
            white_avg_s=sum(white_times) / max(len(white_times), 1),
            wall_time_s=time.time() - wall_start,
        )
    finally:
        black.close()
        white.close()


def analyze_results(results: list[MatchResult]) -> dict[str, object]:
    white_wins = [r for r in results if r.winner == -1]
    if white_wins:
        white_win_counts: dict[int, int] = defaultdict(int)
        white_win_moves: dict[int, list[int]] = defaultdict(list)
        for result in white_wins:
            white_win_counts[result.white_budget_sec] += 1
            white_win_moves[result.white_budget_sec].append(result.move_count)
        strongest_white = min(
            white_win_counts,
            key=lambda budget: (-white_win_counts[budget], sum(white_win_moves[budget]) / len(white_win_moves[budget]), budget),
        )
        white_basis = {
            "mode": "white_wins_exist",
            "white_win_counts": dict(sorted(white_win_counts.items())),
            "strongest_white_budget": strongest_white,
        }
    else:
        white_losses: dict[int, list[int]] = defaultdict(list)
        for result in results:
            white_losses[result.white_budget_sec].append(result.move_count)
        strongest_white = max(
            white_losses,
            key=lambda budget: (sum(white_losses[budget]) / len(white_losses[budget]), -budget),
        )
        white_basis = {
            "mode": "all_black_win",
            "white_survival_avg_moves": {
                budget: round(sum(moves) / len(moves), 3) for budget, moves in sorted(white_losses.items())
            },
            "strongest_white_budget": strongest_white,
        }

    black_wins: dict[int, list[int]] = defaultdict(list)
    for result in results:
        if result.winner == 1:
            black_wins[result.black_budget_sec].append(result.move_count)
    fastest_black = min(
        black_wins,
        key=lambda budget: (sum(black_wins[budget]) / len(black_wins[budget]), budget),
    )

    return {
        "white_basis": white_basis,
        "fastest_black_budget": fastest_black,
        "black_avg_win_moves": {budget: round(sum(moves) / len(moves), 3) for budget, moves in sorted(black_wins.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rapfi 1-7s single round-robin with 6-way concurrency.")
    parser.add_argument("--exe", type=str, default=str(Path("Rapfi-engine/pbrain-rapfi-macos-apple-silicon").resolve()))
    parser.add_argument("--cwd", type=str, default=str(Path("Rapfi-engine").resolve()))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budgets", type=str, default="1,2,3,4,5,6,7")
    args = parser.parse_args()

    budgets = [int(item) for item in args.budgets.split(",") if item.strip()]
    tasks = build_round_robin_tasks(budgets)
    print({"budgets": budgets, "num_matches": len(tasks), "workers": args.workers}, flush=True)

    results: list[MatchResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(run_single_match, task, args.exe, args.cwd): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            print(dataclasses.asdict(result), flush=True)

    results.sort(key=lambda item: (item.round_index, item.black_budget_sec, item.white_budget_sec))
    analysis = analyze_results(results)
    print({"analysis": analysis}, flush=True)

    strongest_white = analysis["white_basis"]["strongest_white_budget"]
    fastest_black = analysis["fastest_black_budget"]
    if strongest_white != fastest_black:
        final_task = MatchTask(round_index=999, black_budget_sec=int(fastest_black), white_budget_sec=int(strongest_white))
        final_result = run_single_match(final_task, args.exe, args.cwd)
        print({"final_match": dataclasses.asdict(final_result)}, flush=True)


if __name__ == "__main__":
    main()
