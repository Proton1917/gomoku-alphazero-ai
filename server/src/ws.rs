//! WebSocket 路由，与原 FastAPI `/ws/*` 行为一致：
//! 帧循环推送，`done` 帧后服务端关闭连接。

use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use axum::extract::{Path, State, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::any;
use axum::Router;
use serde_json::Value;

use crate::AppState;

const NOT_FOUND_CLOSE: u16 = 4404;

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/ws/game/{session_id}/research", any(research_socket))
        .route("/ws/game/{session_id}/autoplay", any(autoplay_socket))
        .route("/ws/game/{session_id}/ai-move", any(ai_move_socket))
        .route("/ws/battle/{session_id}", any(battle_socket))
}

async fn research_socket(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Response {
    ws.on_upgrade(move |socket| {
        run_game_stream(socket, state, session_id, "research", Duration::from_millis(20))
    })
}

async fn autoplay_socket(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Response {
    ws.on_upgrade(move |socket| {
        run_game_stream(socket, state, session_id, "autoplay", Duration::from_millis(150))
    })
}

async fn ai_move_socket(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Response {
    ws.on_upgrade(move |socket| {
        run_game_stream(socket, state, session_id, "ai_move", Duration::from_millis(20))
    })
}

async fn battle_socket(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Response {
    ws.on_upgrade(move |socket| run_battle_stream(socket, state, session_id))
}

async fn run_game_stream(
    mut socket: WebSocket,
    state: Arc<AppState>,
    session_id: String,
    stream_name: &'static str,
    interval: Duration,
) {
    let (generation, mut expected_revision) =
        match state.games.activate_stream(&session_id, stream_name).await {
            Ok(pair) => pair,
            Err(_) => {
                close_with_code(&mut socket, NOT_FOUND_CLOSE).await;
                return;
            }
        };

    loop {
        let step = match stream_name {
            "research" => {
                state
                    .games
                    .research_step(&session_id, generation, expected_revision)
                    .await
            }
            "autoplay" => {
                state
                    .games
                    .autoplay_step(&session_id, generation, expected_revision)
                    .await
            }
            _ => {
                state
                    .games
                    .ai_move_step(&session_id, generation, expected_revision)
                    .await
            }
        };
        let (frame, revision) = match step {
            Ok(pair) => pair,
            Err(_) => break,
        };
        expected_revision = revision;
        let done = frame_done(&frame);
        if send_frame(&mut socket, &frame).await.is_err() {
            break;
        }
        if done {
            break;
        }
        tokio::time::sleep(interval).await;
    }

    state.games.release_stream(&session_id, generation).await;
    let _ = socket.send(Message::Close(None)).await;
}

async fn run_battle_stream(mut socket: WebSocket, state: Arc<AppState>, session_id: String) {
    let (generation, mut expected_revision) = match state.battles.activate_stream(&session_id).await
    {
        Ok(pair) => pair,
        Err(_) => {
            close_with_code(&mut socket, NOT_FOUND_CLOSE).await;
            return;
        }
    };

    loop {
        let (frame, revision) = match state
            .battles
            .battle_step(&session_id, generation, expected_revision)
            .await
        {
            Ok(pair) => pair,
            Err(_) => break,
        };
        expected_revision = revision;
        let done = frame_done(&frame);
        if send_frame(&mut socket, &frame).await.is_err() {
            break;
        }
        if done {
            break;
        }
        tokio::time::sleep(Duration::from_millis(180)).await;
    }

    let _ = socket.send(Message::Close(None)).await;
}

fn frame_done(frame: &Value) -> bool {
    frame.get("done").and_then(Value::as_bool).unwrap_or(true)
}

async fn send_frame(socket: &mut WebSocket, frame: &Value) -> Result<(), axum::Error> {
    socket
        .send(Message::Text(frame.to_string().into()))
        .await
}

async fn close_with_code(socket: &mut WebSocket, code: u16) {
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code,
            reason: "session not found".into(),
        })))
        .await;
}
