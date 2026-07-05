from __future__ import annotations

from dataclasses import dataclass


BOARD_SIZE = 19
KOMI = 6.5
PASS_MOVE = [-1, -1]
RESIGN_MOVE = [-2, -2]


@dataclass(frozen=True)
class MoveResult:
    board: list[list[int]]
    captured: int


@dataclass(frozen=True)
class ScoreResult:
    black_area: int
    white_area: int
    komi: float
    white_score: float
    margin: float
    winner: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "black_area": self.black_area,
            "white_area": self.white_area,
            "komi": self.komi,
            "white_score": self.white_score,
            "margin": self.margin,
            "winner": self.winner,
        }


def empty_board() -> list[list[int]]:
    return [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def is_on_board(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def is_pass(move: list[int] | tuple[int, int]) -> bool:
    return list(move) == PASS_MOVE


def is_resign(move: list[int] | tuple[int, int]) -> bool:
    return list(move) == RESIGN_MOVE


def board_full(board: list[list[int]]) -> bool:
    return all(cell != 0 for row in board for cell in row)


def copy_board(board: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in board]


def board_key(board: list[list[int]]) -> str:
    return "".join(str(cell + 1) for row in board for cell in row)


def apply_go_move(board: list[list[int]], row: int, col: int, player: int) -> MoveResult:
    if not is_on_board(row, col):
        raise ValueError("落子位置超出 19 路棋盘范围")
    if board[row][col] != 0:
        raise ValueError("目标位置已有棋子")

    next_board = copy_board(board)
    next_board[row][col] = player
    opponent = -player
    captured = 0

    for nr, nc in neighbors(row, col):
        if next_board[nr][nc] != opponent:
            continue
        group, liberties = collect_group(next_board, nr, nc)
        if liberties == 0:
            captured += len(group)
            for gr, gc in group:
                next_board[gr][gc] = 0

    own_group, own_liberties = collect_group(next_board, row, col)
    if own_liberties == 0:
        raise ValueError("禁入点：该手为自杀手")

    return MoveResult(board=next_board, captured=captured)


def score_area(board: list[list[int]], komi: float = KOMI) -> ScoreResult:
    black_area = 0
    white_area = 0
    visited: set[tuple[int, int]] = set()

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            value = board[row][col]
            if value == 1:
                black_area += 1
                continue
            if value == -1:
                white_area += 1
                continue
            if (row, col) in visited:
                continue

            region, borders = collect_empty_region(board, row, col)
            visited.update(region)
            if borders == {1}:
                black_area += len(region)
            elif borders == {-1}:
                white_area += len(region)

    white_score = white_area + komi
    margin = black_area - white_score
    if margin > 0:
        winner = 1
    elif margin < 0:
        winner = -1
    else:
        winner = 0
    return ScoreResult(
        black_area=black_area,
        white_area=white_area,
        komi=komi,
        white_score=white_score,
        margin=margin,
        winner=winner,
    )


def neighbors(row: int, col: int):
    if row > 0:
        yield row - 1, col
    if row < BOARD_SIZE - 1:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col < BOARD_SIZE - 1:
        yield row, col + 1


def collect_group(board: list[list[int]], row: int, col: int) -> tuple[set[tuple[int, int]], int]:
    color = board[row][col]
    if color == 0:
        return set(), 0

    group: set[tuple[int, int]] = set()
    liberties: set[tuple[int, int]] = set()
    stack = [(row, col)]

    while stack:
        current = stack.pop()
        if current in group:
            continue
        group.add(current)
        for nr, nc in neighbors(*current):
            value = board[nr][nc]
            if value == 0:
                liberties.add((nr, nc))
            elif value == color and (nr, nc) not in group:
                stack.append((nr, nc))

    return group, len(liberties)


def collect_empty_region(board: list[list[int]], row: int, col: int) -> tuple[set[tuple[int, int]], set[int]]:
    if board[row][col] != 0:
        return set(), set()

    region: set[tuple[int, int]] = set()
    borders: set[int] = set()
    stack = [(row, col)]

    while stack:
        current = stack.pop()
        if current in region:
            continue
        region.add(current)
        for nr, nc in neighbors(*current):
            value = board[nr][nc]
            if value == 0 and (nr, nc) not in region:
                stack.append((nr, nc))
            elif value != 0:
                borders.add(value)

    return region, borders
