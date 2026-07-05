export interface ModelInfo {
  path: string;
  name: string;
  round: number;
  type: string;
  priority: number;
}

export type AiSide = 'none' | 'black' | 'white';

export interface MoveRecord {
  move: number[];
  player: number;
}

export interface ScoreState {
  black_area: number;
  white_area: number;
  komi: number;
  white_score: number;
  margin: number;
  winner: number;
}

export interface GameState {
  id: string;
  board: number[][];
  current_player: number;
  move_history: MoveRecord[];
  history_index: number;
  status: 'active' | 'finished';
  winner: number | null;
  model_path: string;
  simulations: number;
  last_move: number[] | null;
  search_visits: number;
  can_undo: boolean;
  can_redo: boolean;
  score: ScoreState | null;
}

export interface BattleState {
  id: string;
  board: number[][];
  current_player: number;
  move_history: MoveRecord[];
  move_count: number;
  status: 'active' | 'finished';
  winner: number | null;
  black_model_path: string;
  white_model_path: string;
  simulations: number;
  last_move: number[] | null;
  score: ScoreState | null;
}

export interface NNResponse {
  policy_matrix: number[][];
  value_matrix: number[][];
  current_player: number;
}

export interface HeatmapOverlay {
  kind: 'research' | 'nn';
  primaryLabel: string;
  secondaryLabel: string;
  primaryMatrix: number[][];
  valueMatrix: number[][];
  visitCount?: number;
  value?: number;
}

export interface ResearchFrame {
  type: 'research_update';
  game: GameState;
  visit_count: number;
  value: number;
  visit_matrix: number[][];
  value_matrix: number[][];
  done: boolean;
  reason: string | null;
}

export interface AutoplayFrame {
  type: 'autoplay_update';
  game: GameState;
  done: boolean;
  reason: string | null;
}

export interface AiMoveFrame {
  type: 'ai_move_update';
  game: GameState;
  visit_count: number;
  value: number;
  visit_matrix: number[][];
  value_matrix: number[][];
  done: boolean;
  reason: string | null;
}

export interface BattleFrame {
  type: 'battle_update';
  battle: BattleState;
  done: boolean;
  reason: string | null;
}
