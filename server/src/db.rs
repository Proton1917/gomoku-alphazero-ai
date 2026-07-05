//! SQLite 持久化，沿用原 Python 后端的表结构与 JSON 编码方式。

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection, OptionalExtension, Row};
use serde_json::Value;

pub struct Database {
    conn: Mutex<Connection>,
}

#[derive(Clone)]
pub struct GameRecord {
    pub id: String,
    pub board: Value,
    pub current_player: i32,
    pub move_history: Value,
    pub history_index: i64,
    pub status: String,
    pub winner: Option<i32>,
    pub model_path: String,
    pub simulations: i64,
    pub last_move: Option<Value>,
    pub revision: i64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone)]
pub struct BattleRecord {
    pub id: String,
    pub board: Value,
    pub current_player: i32,
    pub move_history: Value,
    pub move_count: i64,
    pub status: String,
    pub winner: Option<i32>,
    pub black_model_path: String,
    pub white_model_path: String,
    pub simulations: i64,
    pub last_move: Option<Value>,
    pub revision: i64,
    pub created_at: String,
    pub updated_at: String,
}

impl Database {
    pub fn open(db_path: &Path) -> Result<Self, String> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("创建数据目录失败: {e}"))?;
        }
        let conn = Connection::open(db_path).map_err(|e| format!("打开数据库失败: {e}"))?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| format!("设置 WAL 失败: {e}"))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS game_sessions (
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
            );
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
            );",
        )
        .map_err(|e| format!("初始化表结构失败: {e}"))?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn save_game(&self, record: &GameRecord) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO game_sessions (
                id, board, current_player, move_history, history_index, status,
                winner, model_path, simulations, last_move, revision, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
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
                updated_at = excluded.updated_at",
            params![
                record.id,
                record.board.to_string(),
                record.current_player,
                record.move_history.to_string(),
                record.history_index,
                record.status,
                record.winner,
                record.model_path,
                record.simulations,
                record.last_move.as_ref().map(|v| v.to_string()),
                record.revision,
                record.created_at,
                record.updated_at,
            ],
        )
        .map_err(|e| format!("保存游戏会话失败: {e}"))?;
        Ok(())
    }

    pub fn save_battle(&self, record: &BattleRecord) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO battle_sessions (
                id, board, current_player, move_history, move_count, status,
                winner, black_model_path, white_model_path, simulations,
                last_move, revision, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
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
                updated_at = excluded.updated_at",
            params![
                record.id,
                record.board.to_string(),
                record.current_player,
                record.move_history.to_string(),
                record.move_count,
                record.status,
                record.winner,
                record.black_model_path,
                record.white_model_path,
                record.simulations,
                record.last_move.as_ref().map(|v| v.to_string()),
                record.revision,
                record.created_at,
                record.updated_at,
            ],
        )
        .map_err(|e| format!("保存对战会话失败: {e}"))?;
        Ok(())
    }

    pub fn get_game(&self, session_id: &str) -> Result<Option<GameRecord>, String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, board, current_player, move_history, history_index, status,
                    winner, model_path, simulations, last_move, revision, created_at, updated_at
             FROM game_sessions WHERE id = ?1",
            params![session_id],
            |row| {
                Ok(GameRecord {
                    id: row.get(0)?,
                    board: parse_json_column(row, 1)?,
                    current_player: row.get(2)?,
                    move_history: parse_json_column(row, 3)?,
                    history_index: row.get(4)?,
                    status: row.get(5)?,
                    winner: row.get(6)?,
                    model_path: row.get(7)?,
                    simulations: row.get(8)?,
                    last_move: parse_optional_json_column(row, 9)?,
                    revision: row.get(10)?,
                    created_at: row.get(11)?,
                    updated_at: row.get(12)?,
                })
            },
        )
        .optional()
        .map_err(|e| format!("查询游戏会话失败: {e}"))
    }

    pub fn get_battle(&self, session_id: &str) -> Result<Option<BattleRecord>, String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, board, current_player, move_history, move_count, status,
                    winner, black_model_path, white_model_path, simulations,
                    last_move, revision, created_at, updated_at
             FROM battle_sessions WHERE id = ?1",
            params![session_id],
            |row| {
                Ok(BattleRecord {
                    id: row.get(0)?,
                    board: parse_json_column(row, 1)?,
                    current_player: row.get(2)?,
                    move_history: parse_json_column(row, 3)?,
                    move_count: row.get(4)?,
                    status: row.get(5)?,
                    winner: row.get(6)?,
                    black_model_path: row.get(7)?,
                    white_model_path: row.get(8)?,
                    simulations: row.get(9)?,
                    last_move: parse_optional_json_column(row, 10)?,
                    revision: row.get(11)?,
                    created_at: row.get(12)?,
                    updated_at: row.get(13)?,
                })
            },
        )
        .optional()
        .map_err(|e| format!("查询对战会话失败: {e}"))
    }
}

fn parse_json_column(row: &Row<'_>, idx: usize) -> rusqlite::Result<Value> {
    let raw: String = row.get(idx)?;
    Ok(serde_json::from_str(&raw).unwrap_or(Value::Null))
}

fn parse_optional_json_column(row: &Row<'_>, idx: usize) -> rusqlite::Result<Option<Value>> {
    let raw: Option<String> = row.get(idx)?;
    Ok(raw.map(|s| serde_json::from_str(&s).unwrap_or(Value::Null)))
}
