from __future__ import annotations

import argparse
import random
from pathlib import Path

from backend.models_service import discover_models
from online_refine_vs_baseline import OnlineCandidate, MatchStats, generate_opening, play_single_game
from train import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online self-improvement against a pool of opponents.")
    parser.add_argument("--candidate", type=str, required=True, help="Candidate model checkpoint path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save online-updated checkpoints.")
    parser.add_argument("--games", type=int, default=12, help="Total games to play.")
    parser.add_argument("--simulations", type=int, default=128, help="MCTS simulations per move for all sides.")
    parser.add_argument("--opening-plies", type=int, default=8, help="Random opening plies before model play starts.")
    parser.add_argument("--replay-size", type=int, default=4096, help="Max samples kept in memory.")
    parser.add_argument("--batch-size", type=int, default=64, help="Online update batch size.")
    parser.add_argument("--update-steps", type=int, default=4, help="Gradient steps after each game.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Online update learning rate.")
    parser.add_argument("--seed", type=int, default=20260404, help="Random seed.")
    parser.add_argument("--save-every", type=int, default=2, help="Save checkpoint every N games.")
    parser.add_argument("--include-legacy", action="store_true", help="Include legacy 4090 baseline in the opponent pool.")
    return parser.parse_args()


def resolve_opponent_pool(include_legacy: bool) -> list[dict[str, object]]:
    pool = []
    for model in discover_models():
        round_num = int(model["round"])
        model_type = str(model["type"])
        if model_type == "legacy":
            if include_legacy:
                pool.append(model)
            continue
        if round_num >= 46:
            pool.append(model)
    return pool


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opponent_pool = resolve_opponent_pool(args.include_legacy)
    if not opponent_pool:
        raise SystemExit("No opponent pool found.")

    candidate = OnlineCandidate(
        checkpoint_path=args.candidate,
        simulations=args.simulations,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        update_steps=args.update_steps,
        learning_rate=args.learning_rate,
    )
    stats = MatchStats()
    opponent_counts: dict[str, int] = {}

    print(
        {
            "phase": "start",
            "candidate": args.candidate,
            "device": str(candidate.device),
            "games": args.games,
            "simulations": args.simulations,
            "opponent_pool": [{"name": model["name"], "path": model["path"], "round": model["round"], "type": model["type"]} for model in opponent_pool],
        },
        flush=True,
    )

    for game_index in range(args.games):
        opponent_meta = opponent_pool[game_index % len(opponent_pool)]
        opening_moves = generate_opening(rng, args.opening_plies)
        opponent_key = str(opponent_meta["path"])
        seen = opponent_counts.get(opponent_key, 0)
        candidate_is_black = seen % 2 == 0
        opponent_counts[opponent_key] = seen + 1
        opponent = Model(str(opponent_meta["path"]), use_rand=0, simulations=args.simulations)

        winner, moves = play_single_game(candidate, opponent, candidate_is_black, opening_moves)
        stats.total_moves += moves

        candidate_color = 1 if candidate_is_black else -1
        if winner == candidate_color:
            stats.candidate_wins += 1
            candidate_result = 1
        elif winner == 0:
            stats.draws += 1
            candidate_result = 0
        else:
            stats.baseline_wins += 1
            candidate_result = -1

        candidate.finalize_episode(candidate_result)
        update_info = candidate.online_update()

        if (game_index + 1) % args.save_every == 0 or game_index + 1 == args.games:
            candidate.save(str(output_dir / f"{game_index + 1}.pth"))

        print(
            {
                "game": game_index + 1,
                "candidate_color": "black" if candidate_is_black else "white",
                "opponent_name": opponent_meta["name"],
                "opponent_round": opponent_meta["round"],
                "opponent_type": opponent_meta["type"],
                "winner": "black" if winner == 1 else "white" if winner == -1 else "draw",
                "moves": moves,
                "candidate_wins": stats.candidate_wins,
                "baseline_wins": stats.baseline_wins,
                "draws": stats.draws,
                "update_loss": round(update_info["loss"], 4),
                "buffer_size": int(update_info["buffer_size"]),
            },
            flush=True,
        )

    candidate.save(str(output_dir / "latest.pth"))
    print(
        {
            "phase": "done",
            "candidate_wins": stats.candidate_wins,
            "baseline_wins": stats.baseline_wins,
            "draws": stats.draws,
            "candidate_score": stats.candidate_wins + 0.5 * stats.draws,
            "games": stats.games,
            "avg_moves": round(stats.total_moves / max(stats.games, 1), 2),
            "latest_checkpoint": str(output_dir / "latest.pth"),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
