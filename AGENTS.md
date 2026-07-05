# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gomoku (五子棋) AI using AlphaZero-style MCTS + ResNet CNN on a 15x15 board.

## 2026-05-10 KataGo GUI Mode

This repository's web UI is currently repurposed as a local KataGo GUI:
- The React canvas board is now **19x19**.
- The FastAPI backend discovers and uses the sibling KataGo build at `../KataGo/cpp/build-metal/katago`.
- Default model path: `../KataGo/cpp/tests/models/g170-b6c96-s175395328-d26788732.bin.gz`.
- `backend/katago_runtime.py` launches KataGo in GTP mode with `maxVisits` mapped from the UI search step value.
- `backend/go_rules.py` handles basic Go legality for the GUI board state: placement, captures, suicide prevention, pass, and two-pass finish.
- Play flow verified through HTTP and browser UI: create game, human move, KataGo AI move, and manual canvas move.

Current run commands:
```bash
./KataGo_GUI.command
# or
./start_project.command

cd backend && conda run -n base uvicorn main:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev -- --host 127.0.0.1 --port 5173
```

Rule coverage added for real play:
- Captures and suicide prevention.
- Positional superko / ko repeat prevention.
- Pass and resignation.
- Area scoring after two consecutive passes with 6.5 komi.
- Frontend exposes `AI 落子`, `自动对弈`, `停一手`, `认输`, `悔棋`, and `前进`.

If this GUI is connected to an external online platform later, keep `backend/go_rules.py` as the legality gate before submitting moves to that platform.

Verification commands used:
```bash
conda run -n base python -m py_compile backend/*.py
cd frontend && npm run build
```

The older 15x15 Gomoku / AlphaZero notes below are retained for legacy training code and older entrypoints, but they no longer describe the default web GUI behavior.

The repository now contains:
- Web play UI (`FastAPI + React/Vite`)
- Web model-vs-model battle UI
- Legacy `pygame` play/battle entrypoints
- Training and training-monitor scripts
- Local benchmark/evaluation scripts for comparing saved models

## Current Status Snapshot

- **Current default model**: `gomoku_cnn_strong/55.pth`
- **Official retained model pool**: `46.pth`, `48.pth`, `49.pth`, `50.pth`, `55.pth`
- **Default UI simulations**: `500`
- **Training target rounds**: `Config.total_rounds = 80` in `train.py`
- **Quick launcher**: `start_project.command`
- **Experimental training artifacts**:
  - `gomoku_cnn_probe_fast/`
  - `gomoku_cnn_probe_ultrafast/`
- **Benchmark outputs**:
  - `benchmark_recent_45_50.json`
  - `benchmark_finalists_46_48_49_50.json`
  - `round55_vs_all.json`
  - `ultrafast_training_eval.json`

## Commands

```bash
# Quick start: launch backend + frontend together
./start_project.command

# Start the web backend API
cd backend && conda run -n base uvicorn main:app --reload

# Start the web frontend dev server
cd frontend && npm run dev

# Build the frontend
cd frontend && npm run build

# Continue training from latest checkpoint up to Config.total_rounds
conda run -n base python train.py

# Monitor training progress
conda run -n base python training_monitor.py

# Benchmark saved models
conda run -n base python benchmark_models.py --rounds 49,50,55 --simulations 12 --games-per-side 2

# Legacy: Run pygame GUI
conda run -n base python game_gui.py

# Legacy: Run pygame model battle GUI
conda run -n base python model_battle.py
```

## Repository Layout

```text
Gomoku/
├── train.py
├── backend/
├── frontend/
├── game_gui.py
├── model_battle.py
├── ai_battle.py
├── benchmark_models.py
├── training_monitor.py
├── start_project.command
├── gomoku_cnn_strong/
├── gomoku_cnn_probe_fast/
├── gomoku_cnn_probe_ultrafast/
└── README.md
```

## Architecture

### Core Module: `train.py`
This is still the central module imported by the backend, legacy GUIs, and benchmark scripts. It contains:
- **`ValueCNN`** — 3-channel input ResNet-like policy/value network
- **`MCTS`** — serial tree search implementation
- **`Config`** — training hyperparameters, now including `total_rounds`
- **`Model`** — wrapper for inference-time MCTS calls
- **`board_to_tensor()`**
- **`evaluation_func()`**
- **`generate_selfplay_data()`**
- **`show_nn()`**

### Board Representation
- `1` = current player
- `-1` = opponent
- `0` = empty
- After each move in search/self-play, the board is negated to swap perspective
- Global `board_size = 15`

### Model Files
- Official kept models live in `gomoku_cnn_strong/`
- Current retained rounds: `46`, `48`, `49`, `50`, `55`
- Discovery / default ordering is:
  1. `55`
  2. `50`
  3. `49`
  4. `48`
  5. `46`
- `55` was promoted into the official pool from the ultrafast probe experiments and is currently the default model for UI sessions

### Device Pattern
Used consistently across the codebase:
```python
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
```

Important implications:
- On Apple Silicon / `mps`, **tree search logic still runs on CPU**, while neural-network forward passes run on `mps`
- Self-play generation uses **single-process mode on MPS**
- CUDA paths use multiprocessing for self-play generation

## Web Backend

### Tech Stack
- **Backend API**: FastAPI
- **Persistence**: SQLite (`backend/gomoku_web.sqlite3`)
- **Concurrency model**: async routes + `asyncio.to_thread()` around CPU-heavy MCTS calls

### Backend Files
| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app, CORS, route mounting |
| `backend/api_routes.py` | REST endpoints |
| `backend/ws_routes.py` | WebSocket endpoints |
| `backend/game_service.py` | Session managers, in-memory MCTS roots, SQLite persistence |
| `backend/models_service.py` | Model discovery and default ordering |
| `backend/db.py` | SQLite schema and persistence helpers |
| `backend/schemas.py` | Pydantic request/response models |

### REST Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models` | List available models |
| POST | `/api/game/new` | Create play session |
| GET | `/api/game/{id}` | Restore play session |
| POST | `/api/game/{id}/move` | Human move |
| POST | `/api/game/{id}/ai-move` | Single AI move |
| POST | `/api/game/{id}/undo` | Undo |
| POST | `/api/game/{id}/redo` | Redo |
| GET | `/api/game/{id}/nn` | NN policy/value overlay |
| POST | `/api/battle/new` | Create battle session |
| GET | `/api/battle/{id}` | Restore battle session |

### WebSockets
| Path | Description |
|------|-------------|
| `/ws/game/{id}/research` | Streams visit/value matrices every 10 simulations |
| `/ws/game/{id}/autoplay` | AI-vs-AI autoplay |
| `/ws/battle/{id}` | Model-vs-model battle stream |

## Web Frontend

### Tech Stack
- React
- TypeScript
- Vite
- HTML Canvas board rendering

### Frontend Files
| File | Purpose |
|------|---------|
| `frontend/src/api/client.ts` | REST client |
| `frontend/src/api/websocket.ts` | WebSocket helper |
| `frontend/src/types/game.ts` | Shared frontend types |
| `frontend/src/hooks/useGameSession.ts` | Play session state + UI logic |
| `frontend/src/hooks/useResearch.ts` | Research WebSocket hook |
| `frontend/src/hooks/useAutoplay.ts` | Autoplay WebSocket hook |
| `frontend/src/components/BoardCanvas.tsx` | Canvas board / heatmap / hover tooltip |
| `frontend/src/components/ControlPanel.tsx` | Play controls |
| `frontend/src/components/ModelSelector.tsx` | Model + simulation controls |
| `frontend/src/components/StatusBar.tsx` | Status cards |
| `frontend/src/components/BattlePage.tsx` | Battle UI |
| `frontend/src/utils/boardRenderer.ts` | Canvas drawing helpers |
| `frontend/src/utils/colors.ts` | Heatmap colors |

### Current Frontend Behavior
- Default play session starts from model `55` and `500` simulations
- Existing sessions are restored from backend + `localStorage`
- If the frontend default model or default simulation count changes, old persisted session IDs are invalidated and a new session is created automatically
- `NN` overlay is now a **persistent toggle**: after a move, it recomputes for the new board instead of disappearing
- Play page includes a **“谁是电脑”** selector:
  - `双方手动`
  - `电脑执黑`
  - `电脑执白`
- Battle layout was fixed so the board area uses the full main content region instead of being squeezed into the left grid column

## Legacy GUI Notes

`game_gui.py` and `model_battle.py` are still supported.

Current legacy behavior:
- They reuse the same model discovery order as the backend
- Hitting Enter on model selection chooses the current default model (`55`)
- The `pygame` GUI keeps `NN` as a persistent toggle and recomputes it after moves

## Training Notes

- `train.py` now supports continuing past round 50 through `Config.total_rounds`
- Current default target is `80`
- Running `train.py` resumes automatically from the highest numbered checkpoint in `Config.model_path`
- On this codebase and hardware profile, full-strength training rounds can be slow on MPS; quick probe directories were used for shorter continuation experiments

## Benchmark / Evaluation Notes

### `benchmark_models.py`
Use this for round-robin comparisons among saved `.pth` models.

Useful options:
- `--rounds 49,50,55`
- `--min-round / --max-round`
- `--simulations`
- `--games-per-side`
- `--output-json`

### Known Evaluation Caveats
- Strong **first-move advantage** still affects small-sample results
- `55` is the current default because it performed well enough to promote, but it did **not** cleanly dominate every retained model in every short benchmark
- Short probe-trained models in `gomoku_cnn_probe_ultrafast/` are experimental, not official defaults

## Dependencies

### Python
- torch
- torchvision
- numpy
- pygame
- tqdm
- matplotlib
- fastapi
- uvicorn
- websockets
- pydantic

Install via:
```bash
conda run -n base python -m pip install -r requirements.txt
```

### Frontend
Install via:
```bash
cd frontend && npm install
```
