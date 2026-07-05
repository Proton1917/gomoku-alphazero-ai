//! KataGo Web API 服务（Rust 版）。
//! 端口 8000，接口与原 FastAPI 后端完全兼容。

mod api;
mod config;
mod db;
mod katago;
mod rules;
mod session;
mod ws;

use std::sync::Arc;

use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, Method};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;
use tower_http::cors::CorsLayer;

use crate::db::Database;
use crate::session::{BattleSessionManager, GameSessionManager};

pub struct AppState {
    pub games: GameSessionManager,
    pub battles: BattleSessionManager,
}

#[tokio::main]
async fn main() {
    let cfg = config::config();
    let database =
        Arc::new(Database::open(&cfg.db_path).unwrap_or_else(|e| panic!("数据库初始化失败: {e}")));

    let state = Arc::new(AppState {
        games: GameSessionManager::new(database.clone()),
        battles: BattleSessionManager::new(database),
    });

    let allowed_origins = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ]
    .into_iter()
    .map(|origin| origin.parse::<HeaderValue>().unwrap())
    .collect::<Vec<_>>();

    let cors = CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_credentials(true)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([CONTENT_TYPE]);

    let app = Router::new()
        .route("/healthz", get(healthcheck))
        .merge(api::router())
        .merge(ws::router())
        .layer(cors)
        .with_state(state);

    let addr = std::env::var("GOMOKU_SERVER_ADDR").unwrap_or_else(|_| "127.0.0.1:8000".to_string());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("监听 {addr} 失败: {e}"));
    println!("KataGo Web API (Rust) listening on http://{addr}");
    axum::serve(listener, app)
        .await
        .expect("HTTP 服务运行失败");
}

async fn healthcheck() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}
