# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two co-existing subsystems:

1. **KataGo Web GUI**（当前主用途）— 19 路围棋对局台。Rust 后端（`server/`，axum）驱动本机 Metal 版 KataGo（GTP 子进程），React 前端（`frontend/`）Canvas 棋盘。
2. **Gomoku AlphaZero training**（保留）— `training/train.py`，15 路五子棋 MCTS + ResNet CNN。

## 2026-07-05 重构记录

- 后端由 Python FastAPI **完整迁移到 Rust**（`server/`），接口与 JSON 结构 1:1 兼容；原 Python 后端归档于 `legacy/python-backend/`。
- 非训练 Python 脚本全部归档到 `legacy/python-scripts/`（game_gui、model_battle、ai_battle、benchmark_models、search_runtime、online_refine_*、rapfi_budget_round_robin）。
- 目录重命名：`gomoku_cnn_strong/` → `models/checkpoints/`，`gomoku_cnn_4090_test/` → `models/experiments/`，`AlphaGomoku/` → `engines/alpha-gomoku/`，`Rapfi-engine/` → `engines/rapfi/`，`rapfi-250615/` → `engines/rapfi-src/`。
- SQLite 移到 `data/gomoku_web.sqlite3`（表结构不变，旧数据已迁移）。
- 前端修复：WebSocket 句柄身份校验（防旧连接迟到事件污染新连接）、AI 自动落子失败退避（连续 3 次失败暂停）、AI 模式悔棋连撤到人类回合、模型/步长选择改为草稿态不被服务器响应覆写、会话恢复以 `getGame` 成功为准；删除死代码（useResearch、BattlePage、NN 热图、colors.ts）。

## Commands

```bash
# 一键启动前后端（缺后端二进制时自动 cargo build --release）
./scripts/start_project.command

# Rust 后端
cd server && cargo build --release && cargo test
GOMOKU_REPO_ROOT=$(pwd) ./server/target/release/gomoku-server   # 从仓库根运行

# 前端
cd frontend && npm install && npm run dev    # 开发 127.0.0.1:5173
cd frontend && npm run build                 # tsc -b && vite build

# 训练（Python 仅剩这里）
conda run -n base python training/train.py
conda run -n base python training/training_monitor.py
```

## Architecture

### Rust backend (`server/src/`)

| 模块 | 职责 |
|------|------|
| `main.rs` | axum 应用、CORS（5173/4173）、`/healthz`、监听 127.0.0.1:8000 |
| `config.rs` | 路径解析与模型发现；env：`GOMOKU_REPO_ROOT` / `KATAGO_ROOT` / `GOMOKU_DB_PATH` / `GOMOKU_SERVER_ADDR` |
| `rules.rs` | 围棋规则：提子、禁自杀、区域数子（贴目 6.5）、`board_key` 同形键 |
| `katago.rs` | KataGo GTP 子进程：惰性启动、每次落子前 `clear_board` + 重放手顺、`genmove`；stderr 合流 stdout |
| `session.rs` | Game/Battle 会话、undo/redo（history_index 截断重放）、全局同形拒手、stream generation/revision 失效机制 |
| `db.rs` | rusqlite（bundled）、WAL、与原 Python 相同的表结构与 JSON 列编码 |
| `api.rs` | REST `/api/*`；错误体 `{"detail": "..."}` 与 FastAPI 一致 |
| `ws.rs` | WebSocket `/ws/*`；done 帧后服务端关闭；会话缺失关闭码 4404 |

关键语义（移植自原 Python，勿破坏）：
- 落子 `[-1,-1]`=pass、`[-2,-2]`=resign；棋子 1=黑、-1=白。
- 会话状态 JSON 字段名/结构必须与前端 `types/game.ts` 保持一致。
- 引擎调用在 `spawn_blocking` 中执行，会话锁（tokio Mutex）在整个操作期间持有。
- research/nn 端点返回零矩阵（KataGo 直连引擎不提供逐点 NN 概览），前端已删除对应 UI。

### Frontend (`frontend/src/`)

| 文件 | 职责 |
|------|------|
| `hooks/useGameSession.ts` | 全部对局状态与操作；AI 自动落子 effect（带失败退避）；AI 模式悔棋连撤 |
| `hooks/useGameStream.ts` | autoplay/ai-move 共用的 WS 流 hook；所有回调校验句柄身份 |
| `api/client.ts` / `api/websocket.ts` | REST / WS 封装；`close()` 先摘除回调再关闭 |
| `components/BoardCanvas.tsx` | Canvas 棋盘；位图尺寸设置与重绘分离在两个 effect |
| `utils/boardRenderer.ts` | 绘制；`BOARD_SIZE = 19` 单一出处 |

localStorage 键：`katago:web:game-id`、`katago:web:ai-side`。

### Training (`training/`)

- `train.py`：`ValueCNN`、`MCTS`、`Config`（`total_rounds = 80`）、自对弈与训练循环；`Config.model_path` 锚定 `<repo>/models/checkpoints`，续训自动从最高轮次检查点恢复。
- `training_monitor.py`：监控 + 对弈测试；从 `legacy/python-scripts/ai_battle.py` 导入对弈函数（sys.path 注入）。
- 设备选择：cuda → mps → cpu；MPS 下自对弈单进程、树搜索在 CPU。
- 模型池：`models/checkpoints/{46,48,49,50,55}.pth`，默认 55。

### Models & engines

- `models/model_4090_trained.pth` 是指向 `experiments/model_4090_trained.pth` 的符号链接。
- `engines/` 整个目录不入库（第三方引擎与源码树）。
- `*.pth`、`*.bin.gz`、`data/` 均被 .gitignore 排除。

## Verification

```bash
cd server && cargo test                     # 规则/GTP 坐标/重放 单元测试
cd frontend && npm run build                # 类型检查 + 构建
conda run -n base python -m py_compile training/*.py
```

集成冒烟：起 `gomoku-server` + `vite preview`，浏览器验证建局/落子/悔棋/AI 落子（KataGo Metal 实测 8 visits 约 1.7s/手）。

## Known Caveats

- KataGo 每会话一个 GTP 进程，首次 `genmove` 前有 Metal 初始化延迟。
- 评估存在先手优势，短基准样本小；55 号模型是默认但并非全面碾压池内其他模型。
- `legacy/` 内脚本的相对路径基于旧目录结构，复活需改路径（见 `legacy/README.md`）。
