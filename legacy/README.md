# Legacy Python 归档

本目录是 Python 代码的备份归档（2026-07-05 后端迁移到 Rust 时归档），**不再维护**。
当前生效的后端在 `server/`（Rust / axum），训练代码在 `training/`。

## python-backend/

原 FastAPI 后端（`backend/`）的完整备份。功能已 1:1 移植到 `server/`：
REST `/api/*`、WebSocket `/ws/*`、围棋规则（提子/禁自杀/全局同形/数子）、
KataGo GTP 子进程管理、SQLite 持久化（表结构不变，数据库现位于 `data/gomoku_web.sqlite3`）。

## python-scripts/

原仓库根目录的非训练 Python 脚本：

| 文件 | 原用途 |
|------|--------|
| `game_gui.py` | pygame 人机对弈 GUI（15 路五子棋时代） |
| `model_battle.py` | pygame 模型对战 GUI |
| `ai_battle.py` | 模型对弈函数库（`training/training_monitor.py` 仍从这里导入） |
| `benchmark_models.py` | 保存模型的循环赛基准测试 |
| `search_runtime.py` | Rapfi/MCTS 搜索运行时封装 |
| `online_refine_vs_baseline.py` | 在线精炼实验（对基线） |
| `online_refine_vs_pool.py` | 在线精炼实验（对模型池） |
| `rapfi_budget_round_robin.py` | Rapfi 预算循环赛 |

注意：这些脚本内的相对路径（如 `gomoku_cnn_strong/`、`backend.models_service`）
基于旧目录结构，如需复活请自行调整为 `models/checkpoints/`、`legacy/python-backend/` 等新路径。
