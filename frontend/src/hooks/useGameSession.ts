import { useCallback, useEffect, useRef, useState } from 'react';

import {
  ApiError,
  createGame,
  describeError,
  getGame,
  listModels,
  makeMove,
  passMove,
  redoMove,
  resignGame,
  undoMove,
} from '../api/client';
import { useGameStream } from './useGameStream';
import type { AiMoveFrame, AiSide, AutoplayFrame, GameState, ModelInfo } from '../types/game';

const GAME_STORAGE_KEY = 'katago:web:game-id';
const AI_SIDE_STORAGE_KEY = 'katago:web:ai-side';
const DEFAULT_SIMULATIONS = 96;
// AI 自动落子连续失败达到该次数后暂停，等待用户操作，避免重连风暴
const MAX_AI_MOVE_FAILURES = 3;

export function useGameSession() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [game, setGame] = useState<GameState | null>(null);
  const [aiSide, setAiSide] = useState<AiSide>(() => {
    const stored = window.localStorage.getItem(AI_SIDE_STORAGE_KEY);
    return stored === 'black' || stored === 'white' ? stored : 'none';
  });
  const [selectedModelPath, setSelectedModelPath] = useState('');
  const [selectedSimulations, setSelectedSimulations] = useState(DEFAULT_SIMULATIONS);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const storedGameIdRef = useRef<string | null>(null);
  const aiMoveFailuresRef = useRef(0);

  const syncGame = useCallback((nextGame: GameState) => {
    setGame(nextGame);
    if (storedGameIdRef.current !== nextGame.id) {
      storedGameIdRef.current = nextGame.id;
      window.localStorage.setItem(GAME_STORAGE_KEY, nextGame.id);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      setLoading(true);
      try {
        const availableModels = await listModels();
        if (cancelled) {
          return;
        }
        setModels(availableModels);
        if (availableModels.length === 0) {
          setError('没有找到任何可用模型');
          return;
        }
        const restored = await restoreOrCreate(availableModels);
        if (cancelled) {
          return;
        }
        syncGame(restored);
        // 模型/搜索步长作为“新建对局”的草稿态，只在初始化时同步一次
        setSelectedModelPath(restored.model_path);
        setSelectedSimulations(restored.simulations);
      } catch (bootstrapError) {
        if (!cancelled) {
          setError(describeError(bootstrapError));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, [syncGame]);

  useEffect(() => {
    window.localStorage.setItem(AI_SIDE_STORAGE_KEY, aiSide);
  }, [aiSide]);

  const autoplay = useGameStream<AutoplayFrame>(
    game ? `/ws/game/${game.id}/autoplay` : null,
    '自动对弈连接失败',
    {
      onFrame: (frame) => syncGame(frame.game),
      onError: (message) => setError(message),
    },
  );

  const aiMove = useGameStream<AiMoveFrame>(
    game ? `/ws/game/${game.id}/ai-move` : null,
    'AI 搜索连接失败',
    {
      onFrame: (frame) => syncGame(frame.game),
      onError: (message) => setError(message),
      onEnd: (receivedDone) => {
        if (receivedDone) {
          aiMoveFailuresRef.current = 0;
          return;
        }
        aiMoveFailuresRef.current += 1;
        if (aiMoveFailuresRef.current === MAX_AI_MOVE_FAILURES) {
          setError('AI 落子连续失败，已暂停自动落子；请检查后端服务后重试');
        }
      },
    },
  );

  // AI 执黑/执白时轮到 AI 自动落子
  useEffect(() => {
    if (
      !game ||
      loading ||
      busy ||
      autoplay.active ||
      aiMove.active ||
      game.status !== 'active' ||
      aiSide === 'none'
    ) {
      return;
    }
    if ((aiSide === 'black' && game.current_player !== 1) || (aiSide === 'white' && game.current_player !== -1)) {
      return;
    }
    if (aiMoveFailuresRef.current >= MAX_AI_MOVE_FAILURES) {
      return;
    }
    aiMove.start();
  }, [aiMove, aiSide, autoplay.active, busy, game, loading]);

  // 切换 AI 侧或换对局时清除失败计数
  useEffect(() => {
    aiMoveFailuresRef.current = 0;
  }, [aiSide, game?.id]);

  async function restoreOrCreate(availableModels: ModelInfo[]): Promise<GameState> {
    const storedId = window.localStorage.getItem(GAME_STORAGE_KEY);
    if (storedId) {
      try {
        return await getGame(storedId);
      } catch (restoreError) {
        if (!(restoreError instanceof ApiError && restoreError.status === 404)) {
          throw restoreError;
        }
        window.localStorage.removeItem(GAME_STORAGE_KEY);
      }
    }
    return createGame({ model_path: availableModels[0]?.path, simulations: DEFAULT_SIMULATIONS });
  }

  async function runGameAction(action: () => Promise<GameState>) {
    setBusy(true);
    setError(null);
    try {
      syncGame(await action());
    } catch (actionError) {
      setError(describeError(actionError));
    } finally {
      setBusy(false);
    }
  }

  async function createNewGame() {
    if (!selectedModelPath) {
      setError('请先选择模型');
      return;
    }
    await runGameAction(() => createGame({ model_path: selectedModelPath, simulations: selectedSimulations }));
  }

  function isHumanBlocked(): boolean {
    return (
      !game ||
      busy ||
      autoplay.active ||
      aiMove.active ||
      game.status !== 'active' ||
      (aiSide === 'black' && game.current_player === 1) ||
      (aiSide === 'white' && game.current_player === -1)
    );
  }

  async function handleCellClick(row: number, col: number) {
    if (isHumanBlocked() || game!.board[row]?.[col] !== 0) {
      return;
    }
    await runGameAction(() => makeMove(game!.id, row, col));
  }

  async function handleAIMove() {
    if (!game || aiMove.active || autoplay.active || busy || game.status !== 'active') {
      return;
    }
    setError(null);
    aiMoveFailuresRef.current = 0;
    aiMove.start();
  }

  async function handleUndo() {
    if (!game || busy || autoplay.active || aiMove.active || !game.can_undo) {
      return;
    }
    const gameId = game.id;
    await runGameAction(async () => {
      let next = await undoMove(gameId);
      if (aiSide !== 'none') {
        // AI 对弈模式：连撤到轮到人类为止，否则 AI 会立刻把刚撤销的手下回来
        const aiPlayer = aiSide === 'black' ? 1 : -1;
        while (next.can_undo && next.status === 'active' && next.current_player === aiPlayer) {
          next = await undoMove(gameId);
        }
      }
      return next;
    });
  }

  async function handleRedo() {
    if (!game || busy || autoplay.active || aiMove.active || !game.can_redo) {
      return;
    }
    await runGameAction(() => redoMove(game.id));
  }

  async function handlePass() {
    if (isHumanBlocked()) {
      return;
    }
    await runGameAction(() => passMove(game!.id));
  }

  async function handleResign() {
    if (isHumanBlocked()) {
      return;
    }
    await runGameAction(() => resignGame(game!.id));
  }

  function toggleAutoplay() {
    if (aiMove.active) {
      return;
    }
    if (autoplay.active) {
      autoplay.stop();
      return;
    }
    autoplay.start();
  }

  const summary = {
    currentPlayer: describeCurrentPlayer(game),
    moveCount: String(game?.history_index !== undefined ? game.history_index + 1 : 0),
    searchVisits: describeSearchVisits(game),
    searchPower: describeSearchPower(game),
    searchMode: aiMove.active ? 'KataGo 思考中' : autoplay.active ? '自动对弈中' : '等待操作',
  };

  return {
    models,
    game,
    aiSide,
    selectedModelPath,
    selectedSimulations,
    loading,
    busy: busy || aiMove.active,
    error,
    autoplayActive: autoplay.active,
    aiMoveActive: aiMove.active,
    setAiSide,
    setSelectedModelPath,
    setSelectedSimulations,
    createNewGame,
    handleCellClick,
    handleAIMove,
    handlePass,
    handleResign,
    handleUndo,
    handleRedo,
    toggleAutoplay,
    summary,
  };
}

function describeSearchVisits(game: GameState | null): string {
  if (!game || game.search_visits <= 0) {
    return '—';
  }
  const seconds = game.search_millis > 0 ? ` / ${(game.search_millis / 1000).toFixed(1)}s` : '';
  return `${game.search_visits}${seconds}`;
}

function describeSearchPower(game: GameState | null): string {
  if (!game || game.visits_per_second <= 0) {
    return '—';
  }
  const vps = game.visits_per_second;
  const label = vps >= 1000 ? `${(vps / 1000).toFixed(2)}k` : String(vps);
  return `${label} visits/s`;
}

function describeCurrentPlayer(game: GameState | null): string {
  if (!game) {
    return '加载中';
  }
  if (game.status === 'finished') {
    if (game.score) {
      const winnerLabel = game.score.winner === 1 ? '黑胜' : game.score.winner === -1 ? '白胜' : '和棋';
      return `${winnerLabel} ${Math.abs(game.score.margin).toFixed(1)}目`;
    }
    return game.winner === 1 ? '黑棋胜' : game.winner === -1 ? '白棋胜' : '终局';
  }
  return game.current_player === 1 ? '黑棋回合' : '白棋回合';
}
