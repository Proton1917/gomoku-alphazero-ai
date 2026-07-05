from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from backend.models_service import discover_models
from train import Model, board_size, evaluation_func


@dataclass
class MatchStats:
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0
    total_moves: int = 0

    @property
    def games(self) -> int:
        return self.black_wins + self.white_wins + self.draws

    @property
    def average_moves(self) -> float:
        return self.total_moves / self.games if self.games else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Gomoku models with round-robin matches.")
    parser.add_argument("--rounds", type=str, default="", help="Comma-separated explicit rounds to include.")
    parser.add_argument("--min-round", type=int, default=45, help="Minimum round number to include.")
    parser.add_argument("--max-round", type=int, default=50, help="Maximum round number to include.")
    parser.add_argument("--recent", type=int, default=0, help="Keep only the newest N rounds after range filtering.")
    parser.add_argument("--simulations", type=int, default=16, help="MCTS simulations per move.")
    parser.add_argument("--games-per-side", type=int, default=1, help="Games for each color assignment.")
    parser.add_argument("--max-models", type=int, default=0, help="Cap number of models after filtering.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save raw results as JSON.")
    return parser.parse_args()


def select_models(args: argparse.Namespace) -> list[dict[str, object]]:
    requested_rounds = {
        int(part.strip())
        for part in args.rounds.split(",")
        if part.strip()
    }

    if requested_rounds:
        models = [
            model
            for model in discover_models()
            if int(model["round"]) in requested_rounds
        ]
    else:
        models = [
            model
            for model in discover_models()
            if args.min_round <= int(model["round"]) <= args.max_round
        ]
    models.sort(key=lambda item: int(item["round"]))

    if args.recent > 0:
        models = models[-args.recent :]

    if args.max_models > 0:
        models = models[-args.max_models :]

    if len(models) < 2:
        raise SystemExit("Need at least two models after filtering.")

    return models


def play_single_game(black: Model, white: Model, simulations: int) -> tuple[int, int]:
    board = [[0] * board_size for _ in range(board_size)]

    for move_index in range(board_size * board_size):
        current_model = black if move_index % 2 == 0 else white
        row, col = current_model.call(board, simulations=simulations)
        board[row][col] = 1

        if evaluation_func(board) != 0:
            return (1 if move_index % 2 == 0 else -1), move_index + 1

        for board_row in range(board_size):
            for board_col in range(board_size):
                board[board_row][board_col] *= -1

    return 0, board_size * board_size


def add_score(scoreboard: dict[int, dict[str, float]], round_num: int, points: float, key: str) -> None:
    scoreboard[round_num]["points"] += points
    scoreboard[round_num][key] += 1


def run_benchmark(models: list[dict[str, object]], simulations: int, games_per_side: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    loaded_models = {
        int(model["round"]): Model(str(model["path"]), use_rand=0, simulations=simulations)
        for model in models
    }

    scoreboard: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "points": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "draws": 0.0,
            "games": 0.0,
        }
    )
    matches: list[dict[str, object]] = []

    for left_model, right_model in itertools.combinations(models, 2):
        left_round = int(left_model["round"])
        right_round = int(right_model["round"])
        stats = MatchStats()

        for _ in range(games_per_side):
            winner, move_count = play_single_game(loaded_models[left_round], loaded_models[right_round], simulations)
            stats.total_moves += move_count
            if winner == 1:
                stats.black_wins += 1
                add_score(scoreboard, left_round, 1.0, "wins")
                add_score(scoreboard, right_round, 0.0, "losses")
            elif winner == -1:
                stats.white_wins += 1
                add_score(scoreboard, left_round, 0.0, "losses")
                add_score(scoreboard, right_round, 1.0, "wins")
            else:
                stats.draws += 1
                add_score(scoreboard, left_round, 0.5, "draws")
                add_score(scoreboard, right_round, 0.5, "draws")
            scoreboard[left_round]["games"] += 1
            scoreboard[right_round]["games"] += 1

        for _ in range(games_per_side):
            winner, move_count = play_single_game(loaded_models[right_round], loaded_models[left_round], simulations)
            stats.total_moves += move_count
            if winner == 1:
                stats.black_wins += 1
                add_score(scoreboard, right_round, 1.0, "wins")
                add_score(scoreboard, left_round, 0.0, "losses")
            elif winner == -1:
                stats.white_wins += 1
                add_score(scoreboard, right_round, 0.0, "losses")
                add_score(scoreboard, left_round, 1.0, "wins")
            else:
                stats.draws += 1
                add_score(scoreboard, right_round, 0.5, "draws")
                add_score(scoreboard, left_round, 0.5, "draws")
            scoreboard[left_round]["games"] += 1
            scoreboard[right_round]["games"] += 1

        matches.append(
            {
                "left_round": left_round,
                "right_round": right_round,
                "games": stats.games,
                "black_wins": stats.black_wins,
                "white_wins": stats.white_wins,
                "draws": stats.draws,
                "average_moves": round(stats.average_moves, 2),
            }
        )

    ranking = []
    for model in models:
        round_num = int(model["round"])
        stats = scoreboard[round_num]
        ranking.append(
            {
                "round": round_num,
                "points": stats["points"],
                "wins": int(stats["wins"]),
                "losses": int(stats["losses"]),
                "draws": int(stats["draws"]),
                "games": int(stats["games"]),
                "win_rate": stats["points"] / stats["games"] if stats["games"] else 0.0,
                "path": str(model["path"]),
            }
        )

    ranking.sort(key=lambda item: (item["points"], item["win_rate"], item["round"]), reverse=True)
    return ranking, matches


def print_ranking(ranking: list[dict[str, object]], matches: list[dict[str, object]], simulations: int, games_per_side: int) -> None:
    print("=" * 88)
    print(f"Benchmark summary | simulations={simulations} | games_per_side={games_per_side}")
    print("=" * 88)
    print(f"{'rank':>4} {'round':>6} {'points':>8} {'wins':>6} {'losses':>8} {'draws':>6} {'games':>6} {'score':>8}")
    for index, item in enumerate(ranking, start=1):
        print(
            f"{index:>4} {int(item['round']):>6} {float(item['points']):>8.1f} "
            f"{int(item['wins']):>6} {int(item['losses']):>8} {int(item['draws']):>6} "
            f"{int(item['games']):>6} {float(item['win_rate']):>8.3f}"
        )

    print("\nPairwise results")
    for match in matches:
        print(
            f"R{match['left_round']} vs R{match['right_round']} | games={match['games']} "
            f"| black_wins={match['black_wins']} | white_wins={match['white_wins']} "
            f"| draws={match['draws']} | avg_moves={match['average_moves']}"
        )


def main() -> None:
    args = parse_args()
    models = select_models(args)
    ranking, matches = run_benchmark(models, args.simulations, args.games_per_side)
    print_ranking(ranking, matches, args.simulations, args.games_per_side)

    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(
                {
                    "models": models,
                    "ranking": ranking,
                    "matches": matches,
                    "simulations": args.simulations,
                    "games_per_side": args.games_per_side,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved JSON results to {args.output_json}")


if __name__ == "__main__":
    main()
