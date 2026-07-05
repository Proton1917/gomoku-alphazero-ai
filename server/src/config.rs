//! 路径配置与 KataGo 模型发现。
//!
//! 默认约定（可用环境变量覆盖）：
//! - `GOMOKU_REPO_ROOT`：仓库根目录，默认取进程当前工作目录
//! - `KATAGO_ROOT`：KataGo 源码根目录，默认 `<repo_root>/../KataGo`
//! - `GOMOKU_DB_PATH`：SQLite 路径，默认 `<repo_root>/data/gomoku_web.sqlite3`

use std::env;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Serialize;

pub struct Config {
    pub repo_root: PathBuf,
    pub katago_executable: PathBuf,
    pub katago_model_path: PathBuf,
    pub katago_config_path: PathBuf,
    pub db_path: PathBuf,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

pub fn config() -> &'static Config {
    CONFIG.get_or_init(|| {
        let repo_root = env::var("GOMOKU_REPO_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| env::current_dir().expect("无法获取当前工作目录"));
        let katago_root = env::var("KATAGO_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| repo_root.parent().unwrap_or(&repo_root).join("KataGo"));
        let db_path = env::var("GOMOKU_DB_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| repo_root.join("data").join("gomoku_web.sqlite3"));
        Config {
            repo_root,
            katago_executable: katago_root.join("cpp").join("build-metal").join("katago"),
            katago_model_path: katago_root
                .join("cpp")
                .join("tests")
                .join("models")
                .join("g170-b6c96-s175395328-d26788732.bin.gz"),
            katago_config_path: katago_root.join("cpp").join("configs").join("gtp_example.cfg"),
            db_path,
        }
    })
}

#[derive(Serialize, Clone)]
pub struct ModelInfo {
    pub path: String,
    pub name: String,
    pub round: i32,
    #[serde(rename = "type")]
    pub model_type: String,
    pub priority: i32,
}

pub fn discover_models() -> Vec<ModelInfo> {
    let cfg = config();
    if !(cfg.katago_executable.exists()
        && cfg.katago_model_path.exists()
        && cfg.katago_config_path.exists())
    {
        return Vec::new();
    }
    vec![ModelInfo {
        path: canonical_string(&cfg.katago_model_path),
        name: "KataGo 19路 Metal".to_string(),
        round: 0,
        model_type: "katago".to_string(),
        priority: 0,
    }]
}

pub fn get_default_model_path() -> Option<String> {
    discover_models().into_iter().next().map(|m| m.path)
}

pub fn normalize_model_path(candidate: Option<&str>) -> Result<String, String> {
    let models = discover_models();
    if models.is_empty() {
        return Err("未找到 KataGo 可用模型，请先确认 KataGo 已完成 Metal 构建".to_string());
    }
    let candidate = candidate.unwrap_or("");
    if candidate.is_empty() {
        return Ok(models[0].path.clone());
    }

    let mut candidate_path = PathBuf::from(candidate);
    if !candidate_path.is_absolute() {
        candidate_path = config().repo_root.join(candidate_path);
    }
    let candidate_path = canonical_string(&candidate_path);

    for model in &models {
        if model.path == candidate_path {
            return Ok(candidate_path);
        }
    }
    Err("模型不存在或不在允许列表中".to_string())
}

fn canonical_string(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .to_string()
}
