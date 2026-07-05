//! REST 路由，与原 FastAPI `/api/*` 接口一一对应。

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use crate::config::discover_models;
use crate::session::ServiceError;
use crate::AppState;

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(json!({ "detail": self.detail }))).into_response()
    }
}

#[derive(Deserialize)]
pub struct CreateGameRequest {
    #[serde(default)]
    model_path: Option<String>,
    #[serde(default = "default_simulations")]
    simulations: i64,
}

#[derive(Deserialize)]
pub struct MoveRequest {
    row: i32,
    col: i32,
}

#[derive(Deserialize)]
pub struct CreateBattleRequest {
    #[serde(default)]
    black_model_path: Option<String>,
    #[serde(default)]
    white_model_path: Option<String>,
    #[serde(default = "default_simulations")]
    simulations: i64,
}

fn default_simulations() -> i64 {
    128
}

fn validate_simulations(simulations: i64) -> Result<(), ServiceError> {
    if !(1..=2000).contains(&simulations) {
        return Err(ServiceError::new(422, "simulations 必须在 1..=2000 范围内"));
    }
    Ok(())
}

fn validate_coordinate(row: i32, col: i32) -> Result<(), ServiceError> {
    if !crate::rules::is_on_board(row, col) {
        return Err(ServiceError::new(422, "row/col 必须在 0..19 范围内"));
    }
    Ok(())
}

type ApiResult = Result<Json<serde_json::Value>, ServiceError>;

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/models", get(list_models))
        .route("/api/game/new", post(create_game))
        .route("/api/game/{session_id}", get(get_game))
        .route("/api/game/{session_id}/move", post(make_move))
        .route("/api/game/{session_id}/pass", post(pass_move))
        .route("/api/game/{session_id}/resign", post(resign))
        .route("/api/game/{session_id}/ai-move", post(ai_move))
        .route("/api/game/{session_id}/undo", post(undo))
        .route("/api/game/{session_id}/redo", post(redo))
        .route("/api/game/{session_id}/nn", get(get_nn))
        .route("/api/battle/new", post(create_battle))
        .route("/api/battle/{session_id}", get(get_battle))
}

async fn list_models() -> Json<serde_json::Value> {
    Json(json!(discover_models()))
}

async fn create_game(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateGameRequest>,
) -> ApiResult {
    validate_simulations(request.simulations)?;
    let value = state
        .games
        .create_session(request.model_path.as_deref(), request.simulations)
        .await?;
    Ok(Json(value))
}

async fn get_game(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> ApiResult {
    Ok(Json(state.games.get_state(&session_id).await?))
}

async fn make_move(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Json(request): Json<MoveRequest>,
) -> ApiResult {
    validate_coordinate(request.row, request.col)?;
    Ok(Json(
        state
            .games
            .make_human_move(&session_id, request.row, request.col)
            .await?,
    ))
}

async fn pass_move(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> ApiResult {
    Ok(Json(state.games.pass_move(&session_id).await?))
}

async fn resign(State(state): State<Arc<AppState>>, Path(session_id): Path<String>) -> ApiResult {
    Ok(Json(state.games.resign(&session_id).await?))
}

async fn ai_move(State(state): State<Arc<AppState>>, Path(session_id): Path<String>) -> ApiResult {
    Ok(Json(state.games.ai_move(&session_id).await?))
}

async fn undo(State(state): State<Arc<AppState>>, Path(session_id): Path<String>) -> ApiResult {
    Ok(Json(state.games.undo(&session_id).await?))
}

async fn redo(State(state): State<Arc<AppState>>, Path(session_id): Path<String>) -> ApiResult {
    Ok(Json(state.games.redo(&session_id).await?))
}

async fn get_nn(State(state): State<Arc<AppState>>, Path(session_id): Path<String>) -> ApiResult {
    Ok(Json(state.games.nn_predictions(&session_id).await?))
}

async fn create_battle(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateBattleRequest>,
) -> ApiResult {
    validate_simulations(request.simulations)?;
    let value = state
        .battles
        .create_session(
            request.black_model_path.as_deref(),
            request.white_model_path.as_deref(),
            request.simulations,
        )
        .await?;
    Ok(Json(value))
}

async fn get_battle(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> ApiResult {
    Ok(Json(state.battles.get_state(&session_id).await?))
}
