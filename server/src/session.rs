//! 会话服务：人机对局与模型对战的内存状态、规则校验与持久化。
//! 语义与原 Python `game_service.py` 保持一致（含 JSON 输出结构）。

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config::{get_default_model_path, normalize_model_path};
use crate::db::{BattleRecord, Database, GameRecord};
use crate::katago::KataGoProcess;
use crate::rules::{
    apply_go_move, board_full, board_key, empty_board, is_pass, is_resign, score_area, Board,
    ScoreResult, BOARD_SIZE, PASS_MOVE, RESIGN_MOVE,
};

pub fn iso_now() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_default()
}

#[derive(Debug)]
pub struct ServiceError {
    pub status_code: u16,
    pub detail: String,
}

impl ServiceError {
    pub fn new(status_code: u16, detail: impl Into<String>) -> Self {
        Self {
            status_code,
            detail: detail.into(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MoveRecord {
    #[serde(rename = "move")]
    pub mv: [i32; 2],
    pub player: i32,
}

fn score_for_state(status: &str, board: &Board, last_move: &Option<[i32; 2]>) -> Option<ScoreResult> {
    if status != "finished" {
        return None;
    }
    if let Some(mv) = last_move {
        if is_resign(mv) {
            return None;
        }
    }
    Some(score_area(board))
}

/// 重放手顺得到的局面快照
struct Position {
    board: Board,
    current_player: i32,
    status: String,
    winner: Option<i32>,
    last_move: Option<[i32; 2]>,
}

fn replay_moves(moves: &[MoveRecord]) -> Position {
    let mut board = empty_board();
    let mut current_player = 1;
    let mut winner: Option<i32> = None;
    let mut status = "active".to_string();
    let mut last_move: Option<[i32; 2]> = None;
    let mut consecutive_passes = 0;

    for record in moves {
        let player = record.player;
        let mv = record.mv;
        last_move = Some(mv);

        if is_resign(&mv) {
            status = "finished".to_string();
            winner = Some(-player);
            current_player = player;
            break;
        }
        if is_pass(&mv) {
            consecutive_passes += 1;
            if consecutive_passes >= 2 {
                let score = score_area(&board);
                status = "finished".to_string();
                winner = Some(score.winner);
                current_player = player;
                break;
            }
            current_player = -player;
            continue;
        }

        consecutive_passes = 0;
        if let Ok(result) = apply_go_move(&board, mv[0], mv[1], player) {
            board = result.board;
        }
        if board_full(&board) {
            let score = score_area(&board);
            status = "finished".to_string();
            winner = Some(score.winner);
            current_player = player;
            break;
        }
        current_player = -player;
    }

    Position {
        board,
        current_player,
        status,
        winner,
        last_move,
    }
}

fn position_keys(moves: &[MoveRecord]) -> HashSet<String> {
    let mut board = empty_board();
    let mut keys = HashSet::new();
    keys.insert(board_key(&board));
    for record in moves {
        if is_pass(&record.mv) || is_resign(&record.mv) {
            continue;
        }
        if let Ok(result) = apply_go_move(&board, record.mv[0], record.mv[1], record.player) {
            board = result.board;
            keys.insert(board_key(&board));
        }
    }
    keys
}

/// 真实搜索算力（visits/s），保留一位小数；无搜索记录时为 0
fn visits_per_second(visits: u32, millis: u64) -> f64 {
    if millis == 0 || visits == 0 {
        return 0.0;
    }
    let vps = visits as f64 / (millis as f64 / 1000.0);
    (vps * 10.0).round() / 10.0
}

fn zero_matrix() -> Value {
    json!(vec![vec![0.0f32; BOARD_SIZE]; BOARD_SIZE])
}

fn board_from_value(value: &Value) -> Option<Board> {
    let board: Board = serde_json::from_value(value.clone()).ok()?;
    if board.len() != BOARD_SIZE || board.iter().any(|row| row.len() != BOARD_SIZE) {
        return None;
    }
    Some(board)
}

fn moves_from_value(value: &Value) -> Vec<MoveRecord> {
    serde_json::from_value(value.clone()).unwrap_or_default()
}

fn last_move_from_value(value: &Option<Value>) -> Option<[i32; 2]> {
    value
        .as_ref()
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

// ---------------------------------------------------------------------------
// 人机对局会话
// ---------------------------------------------------------------------------

pub struct GameSession {
    pub id: String,
    pub board: Board,
    pub current_player: i32,
    pub move_history: Vec<MoveRecord>,
    pub history_index: i64,
    pub status: String,
    pub winner: Option<i32>,
    pub model_path: String,
    pub simulations: i64,
    pub last_move: Option<[i32; 2]>,
    pub revision: i64,
    pub created_at: String,
    pub updated_at: String,
    pub stream_generation: i64,
    pub active_stream: Option<String>,
    pub engine: Arc<StdMutex<KataGoProcess>>,
}

impl GameSession {
    fn new(model_path: String, simulations: i64) -> Self {
        let now = iso_now();
        let engine = Arc::new(StdMutex::new(KataGoProcess::new(&model_path, simulations)));
        Self {
            id: Uuid::new_v4().to_string(),
            board: empty_board(),
            current_player: 1,
            move_history: Vec::new(),
            history_index: -1,
            status: "active".to_string(),
            winner: None,
            model_path,
            simulations,
            last_move: None,
            revision: 0,
            created_at: now.clone(),
            updated_at: now,
            stream_generation: 0,
            active_stream: None,
            engine,
        }
    }

    fn from_record(record: GameRecord) -> Self {
        let engine = Arc::new(StdMutex::new(KataGoProcess::new(
            &record.model_path,
            record.simulations,
        )));
        match board_from_value(&record.board) {
            Some(board) => Self {
                id: record.id,
                board,
                current_player: record.current_player,
                move_history: moves_from_value(&record.move_history),
                history_index: record.history_index,
                status: record.status,
                winner: record.winner,
                model_path: record.model_path,
                simulations: record.simulations,
                last_move: last_move_from_value(&record.last_move),
                revision: record.revision,
                created_at: record.created_at,
                updated_at: record.updated_at,
                stream_generation: 0,
                active_stream: None,
                engine,
            },
            None => Self {
                id: record.id,
                board: empty_board(),
                current_player: 1,
                move_history: Vec::new(),
                history_index: -1,
                status: "active".to_string(),
                winner: None,
                model_path: record.model_path,
                simulations: record.simulations,
                last_move: None,
                revision: record.revision,
                created_at: record.created_at,
                updated_at: record.updated_at,
                stream_generation: 0,
                active_stream: None,
                engine,
            },
        }
    }

    fn to_record(&self) -> GameRecord {
        GameRecord {
            id: self.id.clone(),
            board: json!(self.board),
            current_player: self.current_player,
            move_history: json!(self.move_history),
            history_index: self.history_index,
            status: self.status.clone(),
            winner: self.winner,
            model_path: self.model_path.clone(),
            simulations: self.simulations,
            last_move: self.last_move.map(|mv| json!(mv)),
            revision: self.revision,
            created_at: self.created_at.clone(),
            updated_at: self.updated_at.clone(),
        }
    }

    pub fn to_state(&self) -> Value {
        let (search_visits, search_millis) = self
            .engine
            .lock()
            .map(|engine| (engine.last_search_visits, engine.last_search_millis))
            .unwrap_or((0, 0));
        json!({
            "id": self.id,
            "board": self.board,
            "current_player": self.current_player,
            "move_history": self.move_history,
            "history_index": self.history_index,
            "status": self.status,
            "winner": self.winner,
            "model_path": self.model_path,
            "simulations": self.simulations,
            "last_move": self.last_move,
            "search_visits": search_visits,
            "search_millis": search_millis,
            "visits_per_second": visits_per_second(search_visits, search_millis),
            "can_undo": self.history_index >= 0,
            "can_redo": self.history_index < self.move_history.len() as i64 - 1,
            "score": score_for_state(&self.status, &self.board, &self.last_move),
        })
    }

    fn active_moves(&self) -> &[MoveRecord] {
        let end = (self.history_index + 1).max(0) as usize;
        &self.move_history[..end.min(self.move_history.len())]
    }

    fn rebuild_position(&mut self) {
        let position = replay_moves(self.active_moves());
        self.board = position.board;
        self.current_player = position.current_player;
        self.status = position.status;
        self.winner = position.winner;
        self.last_move = position.last_move;
    }

    fn ensure_active(&self) -> Result<(), ServiceError> {
        if self.status != "active" {
            return Err(ServiceError::new(400, "对局已结束，无法继续操作"));
        }
        Ok(())
    }

    fn apply_new_move(&mut self, row: i32, col: i32, player: i32) -> Result<(), ServiceError> {
        let mv = [row, col];
        if !is_resign(&mv) && !is_pass(&mv) {
            let result = apply_go_move(&self.board, row, col, player)
                .map_err(|e| ServiceError::new(400, e))?;
            if position_keys(self.active_moves()).contains(&board_key(&result.board)) {
                return Err(ServiceError::new(400, "打劫/全局同形：该手会重复旧局面"));
            }
        }

        self.move_history.truncate((self.history_index + 1).max(0) as usize);
        self.move_history.push(MoveRecord { mv, player });
        self.history_index += 1;
        self.rebuild_position();
        self.revision += 1;
        Ok(())
    }
}

pub struct GameSessionManager {
    db: Arc<Database>,
    sessions: StdMutex<HashMap<String, Arc<Mutex<GameSession>>>>,
}

impl GameSessionManager {
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            db,
            sessions: StdMutex::new(HashMap::new()),
        }
    }

    pub async fn create_session(
        &self,
        model_path: Option<&str>,
        simulations: i64,
    ) -> Result<Value, ServiceError> {
        let normalized_model =
            normalize_model_path(model_path).map_err(|e| ServiceError::new(400, e))?;
        let session = GameSession::new(normalized_model, simulations);
        let state = session.to_state();
        self.db
            .save_game(&session.to_record())
            .map_err(|e| ServiceError::new(500, e))?;
        self.sessions
            .lock()
            .unwrap()
            .insert(session.id.clone(), Arc::new(Mutex::new(session)));
        Ok(state)
    }

    fn get_session(&self, session_id: &str) -> Result<Arc<Mutex<GameSession>>, ServiceError> {
        if let Some(session) = self.sessions.lock().unwrap().get(session_id) {
            return Ok(session.clone());
        }
        let record = self
            .db
            .get_game(session_id)
            .map_err(|e| ServiceError::new(500, e))?
            .ok_or_else(|| ServiceError::new(404, "游戏会话不存在"))?;
        let session = Arc::new(Mutex::new(GameSession::from_record(record)));
        self.sessions
            .lock()
            .unwrap()
            .insert(session_id.to_string(), session.clone());
        Ok(session)
    }

    fn persist(&self, session: &mut GameSession) -> Result<(), ServiceError> {
        session.updated_at = iso_now();
        self.db
            .save_game(&session.to_record())
            .map_err(|e| ServiceError::new(500, e))
    }

    pub async fn get_state(&self, session_id: &str) -> Result<Value, ServiceError> {
        let session = self.get_session(session_id)?;
        let session = session.lock().await;
        Ok(session.to_state())
    }

    pub async fn make_human_move(
        &self,
        session_id: &str,
        row: i32,
        col: i32,
    ) -> Result<Value, ServiceError> {
        self.apply_move_locked(session_id, Some([row, col])).await
    }

    pub async fn pass_move(&self, session_id: &str) -> Result<Value, ServiceError> {
        self.apply_move_locked(session_id, Some(PASS_MOVE)).await
    }

    pub async fn resign(&self, session_id: &str) -> Result<Value, ServiceError> {
        self.apply_move_locked(session_id, Some(RESIGN_MOVE)).await
    }

    pub async fn ai_move(&self, session_id: &str) -> Result<Value, ServiceError> {
        self.apply_move_locked(session_id, None).await
    }

    /// mv 为 None 时表示让引擎选点
    async fn apply_move_locked(
        &self,
        session_id: &str,
        mv: Option<[i32; 2]>,
    ) -> Result<Value, ServiceError> {
        let session = self.get_session(session_id)?;
        let mut session = session.lock().await;
        session.ensure_active()?;
        let mv = match mv {
            Some(mv) => mv,
            None => {
                let (row, col) = choose_engine_move(
                    session.engine.clone(),
                    session.current_player,
                    session.active_moves().to_vec(),
                )
                .await?;
                [row, col]
            }
        };
        let player = session.current_player;
        session.apply_new_move(mv[0], mv[1], player)?;
        self.persist(&mut session)?;
        Ok(session.to_state())
    }

    pub async fn undo(&self, session_id: &str) -> Result<Value, ServiceError> {
        let session = self.get_session(session_id)?;
        let mut session = session.lock().await;
        if session.history_index < 0 {
            return Err(ServiceError::new(400, "当前没有可悔棋的步数"));
        }
        session.history_index -= 1;
        session.rebuild_position();
        session.revision += 1;
        self.persist(&mut session)?;
        Ok(session.to_state())
    }

    pub async fn redo(&self, session_id: &str) -> Result<Value, ServiceError> {
        let session = self.get_session(session_id)?;
        let mut session = session.lock().await;
        if session.history_index >= session.move_history.len() as i64 - 1 {
            return Err(ServiceError::new(400, "当前没有可前进的步数"));
        }
        session.history_index += 1;
        session.rebuild_position();
        session.revision += 1;
        self.persist(&mut session)?;
        Ok(session.to_state())
    }

    pub async fn nn_predictions(&self, session_id: &str) -> Result<Value, ServiceError> {
        let session = self.get_session(session_id)?;
        let session = session.lock().await;
        // KataGo 直连引擎不提供逐点 NN 概览，与 Python 版一致地返回零矩阵
        Ok(json!({
            "policy_matrix": zero_matrix(),
            "value_matrix": zero_matrix(),
            "current_player": session.current_player,
        }))
    }

    pub async fn activate_stream(
        &self,
        session_id: &str,
        stream_name: &str,
    ) -> Result<(i64, i64), ServiceError> {
        let session = self.get_session(session_id)?;
        let mut session = session.lock().await;
        session.stream_generation += 1;
        session.active_stream = Some(stream_name.to_string());
        Ok((session.stream_generation, session.revision))
    }

    pub async fn release_stream(&self, session_id: &str, generation: i64) {
        if let Ok(session) = self.get_session(session_id) {
            let mut session = session.lock().await;
            if session.stream_generation == generation {
                session.active_stream = None;
            }
        }
    }

    pub async fn research_step(
        &self,
        session_id: &str,
        generation: i64,
        expected_revision: i64,
    ) -> Result<(Value, i64), ServiceError> {
        let session = self.get_session(session_id)?;
        let session = session.lock().await;
        let reason = stream_stop_reason(&session, generation, expected_revision)
            .unwrap_or_else(|| "katago_direct_engine".to_string());
        Ok((research_frame(&session, true, Some(&reason)), session.revision))
    }

    pub async fn autoplay_step(
        &self,
        session_id: &str,
        generation: i64,
        expected_revision: i64,
    ) -> Result<(Value, i64), ServiceError> {
        let session = self.get_session(session_id)?;
        let mut session = session.lock().await;
        if let Some(reason) = stream_stop_reason(&session, generation, expected_revision) {
            return Ok((autoplay_frame(&session, true, Some(&reason)), session.revision));
        }
        if session.status != "active" {
            return Ok((
                autoplay_frame(&session, true, Some("game_finished")),
                session.revision,
            ));
        }
        let (row, col) = choose_engine_move(
            session.engine.clone(),
            session.current_player,
            session.active_moves().to_vec(),
        )
        .await?;
        let player = session.current_player;
        session.apply_new_move(row, col, player)?;
        self.persist(&mut session)?;
        let done = session.status != "active";
        let reason = if done { Some("game_finished") } else { None };
        Ok((autoplay_frame(&session, done, reason), session.revision))
    }

    pub async fn ai_move_step(
        &self,
        session_id: &str,
        generation: i64,
        expected_revision: i64,
    ) -> Result<(Value, i64), ServiceError> {
        let session = self.get_session(session_id)?;
        let mut session = session.lock().await;
        if let Some(reason) = stream_stop_reason(&session, generation, expected_revision) {
            return Ok((ai_move_frame(&session, true, Some(&reason)), session.revision));
        }
        if session.status != "active" {
            return Ok((
                ai_move_frame(&session, true, Some("game_finished")),
                session.revision,
            ));
        }
        let (row, col) = choose_engine_move(
            session.engine.clone(),
            session.current_player,
            session.active_moves().to_vec(),
        )
        .await?;
        let player = session.current_player;
        session.apply_new_move(row, col, player)?;
        self.persist(&mut session)?;
        Ok((
            ai_move_frame(&session, true, Some("move_applied")),
            session.revision,
        ))
    }
}

async fn choose_engine_move(
    engine: Arc<StdMutex<KataGoProcess>>,
    current_player: i32,
    move_history: Vec<MoveRecord>,
) -> Result<(i32, i32), ServiceError> {
    tokio::task::spawn_blocking(move || {
        let mut engine = engine
            .lock()
            .map_err(|_| "KataGo 引擎锁已损坏".to_string())?;
        engine.choose_move(current_player, &move_history)
    })
    .await
    .map_err(|e| ServiceError::new(500, format!("KataGo 落子任务失败: {e}")))?
    .map_err(|e| ServiceError::new(500, format!("KataGo 落子失败: {e}")))
}

fn stream_stop_reason(
    session: &GameSession,
    generation: i64,
    expected_revision: i64,
) -> Option<String> {
    if session.stream_generation != generation {
        if session.active_stream.is_none() {
            return Some("stream_closed".to_string());
        }
        return Some("stream_superseded".to_string());
    }
    if session.revision != expected_revision {
        return Some("state_changed".to_string());
    }
    None
}

fn research_frame(session: &GameSession, done: bool, reason: Option<&str>) -> Value {
    let visits = session
        .engine
        .lock()
        .map(|engine| engine.last_search_visits)
        .unwrap_or(0);
    json!({
        "type": "research_update",
        "game": session.to_state(),
        "visit_count": visits,
        "value": 0.0,
        "visit_matrix": zero_matrix(),
        "value_matrix": zero_matrix(),
        "done": done,
        "reason": reason,
    })
}

fn autoplay_frame(session: &GameSession, done: bool, reason: Option<&str>) -> Value {
    json!({
        "type": "autoplay_update",
        "game": session.to_state(),
        "done": done,
        "reason": reason,
    })
}

fn ai_move_frame(session: &GameSession, done: bool, reason: Option<&str>) -> Value {
    let visits = session
        .engine
        .lock()
        .map(|engine| engine.last_search_visits)
        .unwrap_or(0);
    json!({
        "type": "ai_move_update",
        "game": session.to_state(),
        "visit_count": visits,
        "value": 0.0,
        "visit_matrix": zero_matrix(),
        "value_matrix": zero_matrix(),
        "done": done,
        "reason": reason,
    })
}

// ---------------------------------------------------------------------------
// 模型对战会话
// ---------------------------------------------------------------------------

pub struct BattleSession {
    pub id: String,
    pub board: Board,
    pub current_player: i32,
    pub move_history: Vec<MoveRecord>,
    pub move_count: i64,
    pub status: String,
    pub winner: Option<i32>,
    pub black_model_path: String,
    pub white_model_path: String,
    pub simulations: i64,
    pub last_move: Option<[i32; 2]>,
    pub revision: i64,
    pub created_at: String,
    pub updated_at: String,
    pub stream_generation: i64,
    pub black_engine: Arc<StdMutex<KataGoProcess>>,
    pub white_engine: Arc<StdMutex<KataGoProcess>>,
}

impl BattleSession {
    fn new(black_model_path: String, white_model_path: String, simulations: i64) -> Self {
        let now = iso_now();
        Self {
            id: Uuid::new_v4().to_string(),
            board: empty_board(),
            current_player: 1,
            move_history: Vec::new(),
            move_count: 0,
            status: "active".to_string(),
            winner: None,
            black_engine: Arc::new(StdMutex::new(KataGoProcess::new(
                &black_model_path,
                simulations,
            ))),
            white_engine: Arc::new(StdMutex::new(KataGoProcess::new(
                &white_model_path,
                simulations,
            ))),
            black_model_path,
            white_model_path,
            simulations,
            last_move: None,
            revision: 0,
            created_at: now.clone(),
            updated_at: now,
            stream_generation: 0,
        }
    }

    fn from_record(record: BattleRecord) -> Self {
        let black_engine = Arc::new(StdMutex::new(KataGoProcess::new(
            &record.black_model_path,
            record.simulations,
        )));
        let white_engine = Arc::new(StdMutex::new(KataGoProcess::new(
            &record.white_model_path,
            record.simulations,
        )));
        match board_from_value(&record.board) {
            Some(board) => Self {
                id: record.id,
                board,
                current_player: record.current_player,
                move_history: moves_from_value(&record.move_history),
                move_count: record.move_count,
                status: record.status,
                winner: record.winner,
                black_model_path: record.black_model_path,
                white_model_path: record.white_model_path,
                simulations: record.simulations,
                last_move: last_move_from_value(&record.last_move),
                revision: record.revision,
                created_at: record.created_at,
                updated_at: record.updated_at,
                stream_generation: 0,
                black_engine,
                white_engine,
            },
            None => Self {
                id: record.id,
                board: empty_board(),
                current_player: 1,
                move_history: Vec::new(),
                move_count: 0,
                status: "active".to_string(),
                winner: None,
                black_model_path: record.black_model_path,
                white_model_path: record.white_model_path,
                simulations: record.simulations,
                last_move: None,
                revision: record.revision,
                created_at: record.created_at,
                updated_at: record.updated_at,
                stream_generation: 0,
                black_engine,
                white_engine,
            },
        }
    }

    fn to_record(&self) -> BattleRecord {
        BattleRecord {
            id: self.id.clone(),
            board: json!(self.board),
            current_player: self.current_player,
            move_history: json!(self.move_history),
            move_count: self.move_count,
            status: self.status.clone(),
            winner: self.winner,
            black_model_path: self.black_model_path.clone(),
            white_model_path: self.white_model_path.clone(),
            simulations: self.simulations,
            last_move: self.last_move.map(|mv| json!(mv)),
            revision: self.revision,
            created_at: self.created_at.clone(),
            updated_at: self.updated_at.clone(),
        }
    }

    pub fn to_state(&self) -> Value {
        json!({
            "id": self.id,
            "board": self.board,
            "current_player": self.current_player,
            "move_history": self.move_history,
            "move_count": self.move_count,
            "status": self.status,
            "winner": self.winner,
            "black_model_path": self.black_model_path,
            "white_model_path": self.white_model_path,
            "simulations": self.simulations,
            "last_move": self.last_move,
            "score": score_for_state(&self.status, &self.board, &self.last_move),
        })
    }

    fn rebuild_position(&mut self) {
        let position = replay_moves(&self.move_history);
        self.board = position.board;
        self.current_player = position.current_player;
        self.status = position.status;
        self.winner = position.winner;
        self.last_move = position.last_move;
    }

    fn apply_move(&mut self, row: i32, col: i32, player: i32) -> Result<(), ServiceError> {
        let mv = [row, col];
        if !is_resign(&mv) && !is_pass(&mv) {
            let result = apply_go_move(&self.board, row, col, player)
                .map_err(|e| ServiceError::new(400, e))?;
            if position_keys(&self.move_history).contains(&board_key(&result.board)) {
                return Err(ServiceError::new(400, "打劫/全局同形：该手会重复旧局面"));
            }
        }
        self.move_history.push(MoveRecord { mv, player });
        self.move_count += 1;
        self.revision += 1;
        self.rebuild_position();
        Ok(())
    }
}

pub struct BattleSessionManager {
    db: Arc<Database>,
    sessions: StdMutex<HashMap<String, Arc<Mutex<BattleSession>>>>,
}

impl BattleSessionManager {
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            db,
            sessions: StdMutex::new(HashMap::new()),
        }
    }

    pub async fn create_session(
        &self,
        black_model_path: Option<&str>,
        white_model_path: Option<&str>,
        simulations: i64,
    ) -> Result<Value, ServiceError> {
        let default_model = get_default_model_path();
        let black_candidate = black_model_path
            .map(str::to_string)
            .or_else(|| default_model.clone());
        let white_candidate = white_model_path
            .map(str::to_string)
            .or_else(|| default_model.clone());
        let normalized_black = normalize_model_path(black_candidate.as_deref())
            .map_err(|e| ServiceError::new(400, e))?;
        let normalized_white = normalize_model_path(white_candidate.as_deref())
            .map_err(|e| ServiceError::new(400, e))?;

        let session = BattleSession::new(normalized_black, normalized_white, simulations);
        let state = session.to_state();
        self.db
            .save_battle(&session.to_record())
            .map_err(|e| ServiceError::new(500, e))?;
        self.sessions
            .lock()
            .unwrap()
            .insert(session.id.clone(), Arc::new(Mutex::new(session)));
        Ok(state)
    }

    fn get_session(&self, session_id: &str) -> Result<Arc<Mutex<BattleSession>>, ServiceError> {
        if let Some(session) = self.sessions.lock().unwrap().get(session_id) {
            return Ok(session.clone());
        }
        let record = self
            .db
            .get_battle(session_id)
            .map_err(|e| ServiceError::new(500, e))?
            .ok_or_else(|| ServiceError::new(404, "对战会话不存在"))?;
        let session = Arc::new(Mutex::new(BattleSession::from_record(record)));
        self.sessions
            .lock()
            .unwrap()
            .insert(session_id.to_string(), session.clone());
        Ok(session)
    }

    pub async fn get_state(&self, session_id: &str) -> Result<Value, ServiceError> {
        let session = self.get_session(session_id)?;
        let session = session.lock().await;
        Ok(session.to_state())
    }

    pub async fn activate_stream(&self, session_id: &str) -> Result<(i64, i64), ServiceError> {
        let session = self.get_session(session_id)?;
        let mut session = session.lock().await;
        session.stream_generation += 1;
        Ok((session.stream_generation, session.revision))
    }

    pub async fn battle_step(
        &self,
        session_id: &str,
        generation: i64,
        expected_revision: i64,
    ) -> Result<(Value, i64), ServiceError> {
        let session = self.get_session(session_id)?;
        let mut session = session.lock().await;
        if session.stream_generation != generation {
            return Ok((
                battle_frame(&session, true, Some("stream_superseded")),
                session.revision,
            ));
        }
        if session.revision != expected_revision {
            return Ok((
                battle_frame(&session, true, Some("state_changed")),
                session.revision,
            ));
        }
        if session.status != "active" {
            return Ok((
                battle_frame(&session, true, Some("game_finished")),
                session.revision,
            ));
        }
        let engine = if session.current_player == 1 {
            session.black_engine.clone()
        } else {
            session.white_engine.clone()
        };
        let (row, col) = choose_engine_move(engine, session.current_player, session.move_history.clone())
            .await
            .map_err(|e| ServiceError::new(500, format!("KataGo 对战落子失败: {}", e.detail)))?;
        let player = session.current_player;
        session.apply_move(row, col, player)?;
        session.updated_at = iso_now();
        self.db
            .save_battle(&session.to_record())
            .map_err(|e| ServiceError::new(500, e))?;
        let done = session.status != "active";
        let reason = if done { Some("game_finished") } else { None };
        Ok((battle_frame(&session, done, reason), session.revision))
    }
}

fn battle_frame(session: &BattleSession, done: bool, reason: Option<&str>) -> Value {
    json!({
        "type": "battle_update",
        "battle": session.to_state(),
        "done": done,
        "reason": reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(r: i32, c: i32, p: i32) -> MoveRecord {
        MoveRecord { mv: [r, c], player: p }
    }

    #[test]
    fn replay_handles_resign() {
        let position = replay_moves(&[mv(0, 0, 1), mv(-2, -2, -1)]);
        assert_eq!(position.status, "finished");
        assert_eq!(position.winner, Some(1));
    }

    #[test]
    fn replay_two_passes_scores_game() {
        let position = replay_moves(&[mv(0, 0, 1), mv(-1, -1, -1), mv(-1, -1, 1)]);
        assert_eq!(position.status, "finished");
        // 全盘只有一个黑子：黑 361 目 vs 白 6.5 目
        assert_eq!(position.winner, Some(1));
    }

    #[test]
    fn superko_detected_via_position_keys() {
        let moves = vec![mv(0, 0, 1)];
        let keys = position_keys(&moves);
        let mut board = empty_board();
        board[0][0] = 1;
        assert!(keys.contains(&board_key(&board)));
        assert!(keys.contains(&board_key(&empty_board())));
    }
}
