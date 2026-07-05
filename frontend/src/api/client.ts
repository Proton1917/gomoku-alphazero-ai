import type { BattleState, GameState, ModelInfo, NNResponse } from '../types/game';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail ?? detail;
    } catch {
      detail = response.statusText;
    }
    throw new ApiError(detail, response.status);
  }

  return response.json() as Promise<T>;
}

export function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return '发生未知错误';
}

export function listModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>('/api/models');
}

export function createGame(payload: { model_path?: string; simulations: number }): Promise<GameState> {
  return request<GameState>('/api/game/new', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getGame(gameId: string): Promise<GameState> {
  return request<GameState>(`/api/game/${gameId}`);
}

export function makeMove(gameId: string, row: number, col: number): Promise<GameState> {
  return request<GameState>(`/api/game/${gameId}/move`, {
    method: 'POST',
    body: JSON.stringify({ row, col }),
  });
}

export function passMove(gameId: string): Promise<GameState> {
  return request<GameState>(`/api/game/${gameId}/pass`, { method: 'POST' });
}

export function resignGame(gameId: string): Promise<GameState> {
  return request<GameState>(`/api/game/${gameId}/resign`, { method: 'POST' });
}

export function aiMove(gameId: string): Promise<GameState> {
  return request<GameState>(`/api/game/${gameId}/ai-move`, { method: 'POST' });
}

export function undoMove(gameId: string): Promise<GameState> {
  return request<GameState>(`/api/game/${gameId}/undo`, { method: 'POST' });
}

export function redoMove(gameId: string): Promise<GameState> {
  return request<GameState>(`/api/game/${gameId}/redo`, { method: 'POST' });
}

export function getNN(gameId: string): Promise<NNResponse> {
  return request<NNResponse>(`/api/game/${gameId}/nn`);
}

export function createBattle(payload: {
  black_model_path?: string;
  white_model_path?: string;
  simulations: number;
}): Promise<BattleState> {
  return request<BattleState>('/api/battle/new', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getBattle(battleId: string): Promise<BattleState> {
  return request<BattleState>(`/api/battle/${battleId}`);
}
