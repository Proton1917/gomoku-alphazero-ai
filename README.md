# Gomoku / KataGo 对局台

本仓库包含两部分：

1. **KataGo Web GUI**（当前主用途）：Rust 后端 + React 前端，驱动本机 Metal 版 KataGo 下 19 路围棋。
2. **五子棋 AlphaZero 训练**（保留）：`training/` 下的 MCTS + ResNet CNN 训练代码（15 路五子棋）。

旧的 Python Web 后端与各类 Python 脚本已归档到 `legacy/`（见 `legacy/README.md`），后端已完整迁移到 Rust。

## 目录结构

```text
Gomoku/
├── server/            # Rust 后端（axum + rusqlite + KataGo GTP）
│   └── src/           # rules / katago / session / db / api / ws
├── frontend/          # React + TypeScript + Vite 前端
├── training/          # 五子棋 AlphaZero 训练（Python / PyTorch MPS）
│   ├── train.py
│   ├── training_monitor.py
│   └── requirements.txt
├── models/
│   ├── checkpoints/   # 正式模型池（46/48/49/50/55.pth）
│   └── experiments/   # 4090 基线与派生实验模型
├── engines/           # 第三方引擎（不入库）：alpha-gomoku / rapfi / rapfi-src
├── legacy/            # 已归档的 Python 代码（只读备份）
│   ├── python-backend/
│   └── python-scripts/
├── scripts/           # 启动脚本（.command）
├── data/              # SQLite 数据库与日志（不入库）
└── README.md / AGENTS.md / LICENSE
```

## 快速开始（KataGo GUI）

前置条件：`../KataGo` 已完成 Metal 构建（`cpp/build-metal/katago`），本机装有 Rust 工具链与 Node.js。

```bash
# 一键启动（编译后端如缺失、起前后端、自动开浏览器）
./scripts/start_project.command

# 手动分别启动
cd server && cargo build --release
cd .. && ./server/target/release/gomoku-server        # 后端 127.0.0.1:8000
cd frontend && npm install && npm run dev             # 前端 127.0.0.1:5173
```

后端环境变量（可选）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GOMOKU_REPO_ROOT` | 进程工作目录 | 仓库根目录 |
| `KATAGO_ROOT` | `<repo>/../KataGo` | KataGo 源码根目录 |
| `GOMOKU_DB_PATH` | `<repo>/data/gomoku_web.sqlite3` | SQLite 路径 |
| `GOMOKU_SERVER_ADDR` | `127.0.0.1:8000` | 监听地址 |

## API 概览（Rust 后端，与原 FastAPI 完全兼容）

REST：`GET /healthz`、`GET /api/models`、`POST /api/game/new`、`GET /api/game/{id}`、
`POST /api/game/{id}/move|pass|resign|ai-move|undo|redo`、`GET /api/game/{id}/nn`、
`POST /api/battle/new`、`GET /api/battle/{id}`

WebSocket：`/ws/game/{id}/research`、`/ws/game/{id}/autoplay`、`/ws/game/{id}/ai-move`、`/ws/battle/{id}`

规则实现：自动提子、禁自杀、全局同形（打劫）拒手、连续两停一手按区域数子终局（贴目 6.5）。

## 训练（五子棋 AlphaZero）

```bash
conda run -n base python -m pip install -r training/requirements.txt
conda run -n base python training/train.py              # 从最新检查点续训至 Config.total_rounds
conda run -n base python training/training_monitor.py   # 监控进度并测试棋力
```

- 检查点目录：`models/checkpoints/`（按轮次命名，如 `55.pth`），路径已锚定仓库根目录，任意工作目录均可运行。
- 4090 基线模型：`models/experiments/model_4090_trained.pth`。
- Apple Silicon 上 NN 前向走 MPS，树搜索在 CPU；自对弈单进程。

## 构建与验证

```bash
cd server && cargo build --release && cargo test   # Rust 后端 + 单元测试
cd frontend && npm run build                       # 前端 tsc + vite
conda run -n base python -m py_compile training/*.py
```

## 许可证

MIT License
