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
