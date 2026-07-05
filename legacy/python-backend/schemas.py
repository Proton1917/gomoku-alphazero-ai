from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    path: str
    name: str
    round: int
    type: str
    priority: int


class MoveRecord(BaseModel):
    move: list[int] = Field(min_length=2, max_length=2)
    player: int


class ScoreState(BaseModel):
    black_area: int
    white_area: int
    komi: float
    white_score: float
    margin: float
    winner: int


class GameState(BaseModel):
    id: str
    board: list[list[int]]
    current_player: int
    move_history: list[MoveRecord]
    history_index: int
    status: Literal["active", "finished"]
    winner: int | None = None
    model_path: str
    simulations: int
    last_move: list[int] | None = None
    search_visits: int = 0
    can_undo: bool
    can_redo: bool
    score: ScoreState | None = None


class BattleState(BaseModel):
    id: str
    board: list[list[int]]
    current_player: int
    move_history: list[MoveRecord]
    move_count: int
    status: Literal["active", "finished"]
    winner: int | None = None
    black_model_path: str
    white_model_path: str
    simulations: int
    last_move: list[int] | None = None
    score: ScoreState | None = None


class CreateGameRequest(BaseModel):
    model_path: str | None = None
    simulations: int = Field(default=128, gt=0, le=2000)


class MoveRequest(BaseModel):
    row: int = Field(ge=0, lt=19)
    col: int = Field(ge=0, lt=19)


class CreateBattleRequest(BaseModel):
    black_model_path: str | None = None
    white_model_path: str | None = None
    simulations: int = Field(default=128, gt=0, le=2000)


class NNResponse(BaseModel):
    policy_matrix: list[list[float]]
    value_matrix: list[list[float]]
    current_player: int


class ResearchFrame(BaseModel):
    type: Literal["research_update"] = "research_update"
    game: GameState
    visit_count: int
    value: float
    visit_matrix: list[list[float]]
    value_matrix: list[list[float]]
    done: bool
    reason: str | None = None


class AutoplayFrame(BaseModel):
    type: Literal["autoplay_update"] = "autoplay_update"
    game: GameState
    done: bool
    reason: str | None = None


class AiMoveFrame(BaseModel):
    type: Literal["ai_move_update"] = "ai_move_update"
    game: GameState
    visit_count: int
    value: float
    visit_matrix: list[list[float]]
    value_matrix: list[list[float]]
    done: bool
    reason: str | None = None


class BattleFrame(BaseModel):
    type: Literal["battle_update"] = "battle_update"
    battle: BattleState
    done: bool
    reason: str | None = None


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
