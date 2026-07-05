from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_routes import create_api_router
from db import Database
from game_service import BattleSessionManager, GameSessionManager
from ws_routes import create_ws_router


database = Database(Path(__file__).resolve().parent / "gomoku_web.sqlite3")
database.initialize()

game_manager = GameSessionManager(database)
battle_manager = BattleSessionManager(database)

app = FastAPI(title="KataGo Web API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(create_api_router(game_manager, battle_manager))
app.include_router(create_ws_router(game_manager, battle_manager))


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
