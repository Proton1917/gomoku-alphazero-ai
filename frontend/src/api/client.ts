import type { GameState, ModelInfo } from '../types/game';

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
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
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

export function undoMove(gameId: string): Promise<GameState> {
  return request<GameState>(`/api/game/${gameId}/undo`, { method: 'POST' });
}

export function redoMove(gameId: string): Promise<GameState> {
  return request<GameState>(`/api/game/${gameId}/redo`, { method: 'POST' });
}
