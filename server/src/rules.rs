//! 围棋规则：落子合法性（提子、自杀禁手）、区域数子、全局同形键。

use serde::Serialize;

pub const BOARD_SIZE: usize = 19;
pub const KOMI: f64 = 6.5;
pub const PASS_MOVE: [i32; 2] = [-1, -1];
pub const RESIGN_MOVE: [i32; 2] = [-2, -2];

pub type Board = Vec<Vec<i32>>;

pub fn empty_board() -> Board {
    vec![vec![0; BOARD_SIZE]; BOARD_SIZE]
}

pub fn is_on_board(row: i32, col: i32) -> bool {
    (0..BOARD_SIZE as i32).contains(&row) && (0..BOARD_SIZE as i32).contains(&col)
}

pub fn is_pass(mv: &[i32; 2]) -> bool {
    *mv == PASS_MOVE
}

pub fn is_resign(mv: &[i32; 2]) -> bool {
    *mv == RESIGN_MOVE
}

pub fn board_full(board: &Board) -> bool {
    board.iter().all(|row| row.iter().all(|&cell| cell != 0))
}

pub fn board_key(board: &Board) -> String {
    let mut key = String::with_capacity(BOARD_SIZE * BOARD_SIZE);
    for row in board {
        for &cell in row {
            key.push(char::from_digit((cell + 1) as u32, 10).unwrap());
        }
    }
    key
}

pub struct MoveResult {
    pub board: Board,
    #[allow(dead_code)]
    pub captured: usize,
}

#[derive(Serialize, Clone, Copy)]
pub struct ScoreResult {
    pub black_area: i32,
    pub white_area: i32,
    pub komi: f64,
    pub white_score: f64,
    pub margin: f64,
    pub winner: i32,
}

fn neighbors(row: usize, col: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut out = Vec::with_capacity(4);
    if row > 0 {
        out.push((row - 1, col));
    }
    if row < BOARD_SIZE - 1 {
        out.push((row + 1, col));
    }
    if col > 0 {
        out.push((row, col - 1));
    }
    if col < BOARD_SIZE - 1 {
        out.push((row, col + 1));
    }
    out.into_iter()
}

fn collect_group(board: &Board, row: usize, col: usize) -> (Vec<(usize, usize)>, usize) {
    let color = board[row][col];
    if color == 0 {
        return (Vec::new(), 0);
    }
    let mut in_group = [[false; BOARD_SIZE]; BOARD_SIZE];
    let mut liberty = [[false; BOARD_SIZE]; BOARD_SIZE];
    let mut group = Vec::new();
    let mut liberties = 0usize;
    let mut stack = vec![(row, col)];

    while let Some((r, c)) = stack.pop() {
        if in_group[r][c] {
            continue;
        }
        in_group[r][c] = true;
        group.push((r, c));
        for (nr, nc) in neighbors(r, c) {
            let value = board[nr][nc];
            if value == 0 {
                if !liberty[nr][nc] {
                    liberty[nr][nc] = true;
                    liberties += 1;
                }
            } else if value == color && !in_group[nr][nc] {
                stack.push((nr, nc));
            }
        }
    }
    (group, liberties)
}

fn collect_empty_region(board: &Board, row: usize, col: usize) -> (Vec<(usize, usize)>, Vec<i32>) {
    if board[row][col] != 0 {
        return (Vec::new(), Vec::new());
    }
    let mut in_region = [[false; BOARD_SIZE]; BOARD_SIZE];
    let mut region = Vec::new();
    let mut borders: Vec<i32> = Vec::new();
    let mut stack = vec![(row, col)];

    while let Some((r, c)) = stack.pop() {
        if in_region[r][c] {
            continue;
        }
        in_region[r][c] = true;
        region.push((r, c));
        for (nr, nc) in neighbors(r, c) {
            let value = board[nr][nc];
            if value == 0 {
                if !in_region[nr][nc] {
                    stack.push((nr, nc));
                }
            } else if !borders.contains(&value) {
                borders.push(value);
            }
        }
    }
    (region, borders)
}

pub fn apply_go_move(board: &Board, row: i32, col: i32, player: i32) -> Result<MoveResult, String> {
    if !is_on_board(row, col) {
        return Err("落子位置超出 19 路棋盘范围".to_string());
    }
    let (row, col) = (row as usize, col as usize);
    if board[row][col] != 0 {
        return Err("目标位置已有棋子".to_string());
    }

    let mut next_board = board.clone();
    next_board[row][col] = player;
    let opponent = -player;
    let mut captured = 0usize;

    for (nr, nc) in neighbors(row, col) {
        if next_board[nr][nc] != opponent {
            continue;
        }
        let (group, liberties) = collect_group(&next_board, nr, nc);
        if liberties == 0 {
            captured += group.len();
            for (gr, gc) in group {
                next_board[gr][gc] = 0;
            }
        }
    }

    let (_, own_liberties) = collect_group(&next_board, row, col);
    if own_liberties == 0 {
        return Err("禁入点：该手为自杀手".to_string());
    }

    Ok(MoveResult {
        board: next_board,
        captured,
    })
}

pub fn score_area(board: &Board) -> ScoreResult {
    let mut black_area = 0i32;
    let mut white_area = 0i32;
    let mut visited = [[false; BOARD_SIZE]; BOARD_SIZE];

    for row in 0..BOARD_SIZE {
        for col in 0..BOARD_SIZE {
            let value = board[row][col];
            if value == 1 {
                black_area += 1;
                continue;
            }
            if value == -1 {
                white_area += 1;
                continue;
            }
            if visited[row][col] {
                continue;
            }
            let (region, borders) = collect_empty_region(board, row, col);
            for &(r, c) in &region {
                visited[r][c] = true;
            }
            if borders == [1] {
                black_area += region.len() as i32;
            } else if borders == [-1] {
                white_area += region.len() as i32;
            }
        }
    }

    let white_score = white_area as f64 + KOMI;
    let margin = black_area as f64 - white_score;
    let winner = if margin > 0.0 {
        1
    } else if margin < 0.0 {
        -1
    } else {
        0
    };
    ScoreResult {
        black_area,
        white_area,
        komi: KOMI,
        white_score,
        margin,
        winner,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_single_stone() {
        let mut board = empty_board();
        // 白子在 (0,0)，黑子围住 (0,1) 与 (1,0)
        board[0][0] = -1;
        board[0][1] = 1;
        let result = apply_go_move(&board, 1, 0, 1).unwrap();
        assert_eq!(result.captured, 1);
        assert_eq!(result.board[0][0], 0);
    }

    #[test]
    fn suicide_rejected() {
        let mut board = empty_board();
        board[0][1] = -1;
        board[1][0] = -1;
        assert!(apply_go_move(&board, 0, 0, 1).is_err());
    }

    #[test]
    fn occupied_rejected() {
        let mut board = empty_board();
        board[3][3] = 1;
        assert!(apply_go_move(&board, 3, 3, -1).is_err());
    }

    #[test]
    fn empty_board_score_is_draw_area() {
        let score = score_area(&empty_board());
        assert_eq!(score.black_area, 0);
        assert_eq!(score.white_area, 0);
        assert_eq!(score.winner, -1); // komi 使白胜
    }

    #[test]
    fn board_key_roundtrip() {
        let mut board = empty_board();
        board[0][0] = 1;
        board[18][18] = -1;
        let key = board_key(&board);
        assert_eq!(key.len(), BOARD_SIZE * BOARD_SIZE);
        assert!(key.starts_with('2'));
        assert!(key.ends_with('0'));
    }
}
