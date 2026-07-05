from __future__ import annotations

import argparse
import copy
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from train import (
    MCTS,
    Model,
    board_size,
    board_to_tensor,
    build_model_from_state_dict,
    calc_next_move,
    evaluation_func,
    get_runtime_device,
)


@dataclass
class ReplaySample:
    board_tensor: torch.Tensor
    policy_tensor: torch.Tensor
    value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online self-improvement against a fixed baseline.")
    parser.add_argument("--candidate", type=str, required=True, help="Candidate model checkpoint path.")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline model checkpoint path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save online-updated checkpoints.")
    parser.add_argument("--games", type=int, default=4, help="Total games to play.")
    parser.add_argument("--simulations", type=int, default=128, help="MCTS simulations per move for both sides.")
    parser.add_argument("--opening-plies", type=int, default=8, help="Random opening plies before model play starts.")
    parser.add_argument("--replay-size", type=int, default=2048, help="Max samples kept in memory.")
    parser.add_argument("--batch-size", type=int, default=64, help="Online update batch size.")
    parser.add_argument("--update-steps", type=int, default=4, help="Gradient steps after each game.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Online update learning rate.")
    parser.add_argument("--seed", type=int, default=20260404, help="Random seed.")
    parser.add_argument("--save-every", type=int, default=2, help="Save checkpoint every N games.")
    return parser.parse_args()


def flip_board(board: list[list[int]]) -> list[list[int]]:
    return [[-cell for cell in row] for row in board]


def generate_opening(rng: random.Random, opening_plies: int) -> list[tuple[int, int]]:
    while True:
        board = [[0] * board_size for _ in range(board_size)]
        moves: list[tuple[int, int]] = []
        ok = True
        for _ in range(opening_plies):
            legal = [(i, j) for i in range(board_size) for j in range(board_size) if board[i][j] == 0]
            move = rng.choice(legal)
            board[move[0]][move[1]] = 1
            moves.append(move)
            if evaluation_func(board) != 0:
                ok = False
                break
            board = flip_board(board)
        if ok:
            return moves


def _advance_root(root, move: tuple[int, int] | None):
    if root is None or move is None or not root.children:
        return None
    child_info = root.children.get(move)
    if child_info is None:
        return None
    child, _prior = child_info
    if child is None:
        return None
    child.parent = None
    return child


class OnlineCandidate:
    def __init__(self, checkpoint_path: str, simulations: int, replay_size: int, batch_size: int, update_steps: int, learning_rate: float):
        self.checkpoint_path = checkpoint_path
        self.simulations = simulations
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.device = get_runtime_device()

        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model = build_model_from_state_dict(state_dict)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_buffer: deque[ReplaySample] = deque(maxlen=replay_size)
        self.root = None
        self.episode_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._reset_search()

    def _reset_search(self) -> None:
        self.mcts = MCTS(self.model, use_rand=0)
        self.root = None

    def _sync_root(self, board: list[list[int]]) -> None:
        if self.root is not None and self.root.board != board:
            self.root = None

    def choose_move(self, board: list[list[int]]) -> tuple[int, int]:
        self._sync_root(board)
        (_value, action_probs), self.root = self.mcts.run(copy.deepcopy(board), self.simulations, 0, self.root, 1)
        move = calc_next_move(board, action_probs, temperature=0)
        self.episode_samples.append(
            (
                board_to_tensor(copy.deepcopy(board)),
                torch.FloatTensor(action_probs).view(-1),
            )
        )
        return move

    def observe_move(self, move: tuple[int, int]) -> None:
        self.root = _advance_root(self.root, move)

    def finalize_episode(self, result: int) -> None:
        for board_tensor, policy_tensor in self.episode_samples:
            self.replay_buffer.append(ReplaySample(board_tensor=board_tensor, policy_tensor=policy_tensor, value=float(result)))
        self.episode_samples.clear()
        self._reset_search()

    def online_update(self) -> dict[str, float]:
        if not self.replay_buffer:
            return {"steps": 0, "loss": 0.0, "buffer_size": 0}

        self.model.train()
        losses: list[float] = []
        for _ in range(self.update_steps):
            batch = random.sample(list(self.replay_buffer), k=min(len(self.replay_buffer), self.batch_size))
            boards = torch.stack([sample.board_tensor for sample in batch]).to(self.device)
            policies = torch.stack([sample.policy_tensor for sample in batch]).to(self.device)
            values = torch.tensor([sample.value for sample in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)
            pred_values, pred_policies = self.model(boards)
            value_loss = F.mse_loss(pred_values, values)
            policy_loss = -(policies * F.log_softmax(pred_policies, dim=1)).sum(dim=1).mean()
            loss = 2 * value_loss + policy_loss
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))

        self._reset_search()
        return {
            "steps": float(self.update_steps),
            "loss": sum(losses) / max(len(losses), 1),
            "buffer_size": float(len(self.replay_buffer)),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


@dataclass
class MatchStats:
    candidate_wins: int = 0
    baseline_wins: int = 0
    draws: int = 0
    total_moves: int = 0

    @property
    def games(self) -> int:
        return self.candidate_wins + self.baseline_wins + self.draws


def play_single_game(candidate: OnlineCandidate, baseline: Model, candidate_is_black: bool, opening_moves: list[tuple[int, int]]) -> tuple[int, int]:
    board = [[0] * board_size for _ in range(board_size)]
    current_color = 1
    move_count = 0
    candidate_color = 1 if candidate_is_black else -1

    for move in opening_moves:
        board[move[0]][move[1]] = 1
        candidate.observe_move(move)
        move_count += 1
        if evaluation_func(board) != 0:
            return current_color, move_count
        board = flip_board(board)
        current_color = -current_color

    while move_count < board_size * board_size:
        if current_color == candidate_color:
            move = candidate.choose_move(board)
        else:
            move = baseline.call(board, simulations=baseline.simulations)

        board[move[0]][move[1]] = 1
        candidate.observe_move(move)
        move_count += 1
        if evaluation_func(board) != 0:
            return current_color, move_count
        board = flip_board(board)
        current_color = -current_color

    return 0, move_count


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = OnlineCandidate(
        checkpoint_path=args.candidate,
        simulations=args.simulations,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        update_steps=args.update_steps,
        learning_rate=args.learning_rate,
    )
    baseline = Model(args.baseline, use_rand=0, simulations=args.simulations)
    stats = MatchStats()

    print(
        {
            "phase": "start",
            "candidate": args.candidate,
            "baseline": args.baseline,
            "device": str(candidate.device),
            "games": args.games,
            "simulations": args.simulations,
            "update_steps": args.update_steps,
            "batch_size": args.batch_size,
            "replay_size": args.replay_size,
        },
        flush=True,
    )

    for game_index in range(args.games):
        opening_moves = generate_opening(rng, args.opening_plies)
        candidate_is_black = game_index % 2 == 0
        winner, moves = play_single_game(candidate, baseline, candidate_is_black, opening_moves)
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
