import { useEffect, useState } from 'react';

import {
  ApiError,
  createGame,
  describeError,
  getGame,
  getNN,
  listModels,
  makeMove,
  passMove,
  redoMove,
  resignGame,
  undoMove,
} from '../api/client';
import { useAiMove } from './useAiMove';
import { useAutoplay } from './useAutoplay';
import { useResearch } from './useResearch';
import type { AiMoveFrame, AiSide, AutoplayFrame, GameState, HeatmapOverlay, ModelInfo, ResearchFrame } from '../types/game';


const GAME_STORAGE_KEY = 'katago:web:game-id';
const AI_SIDE_STORAGE_KEY = 'katago:web:ai-side';
const DEFAULT_MODEL_STORAGE_KEY = 'katago:web:default-model-path';
const DEFAULT_SIMULATIONS_STORAGE_KEY = 'katago:web:default-visits';
const DEFAULT_SIMULATIONS = 96;


export function useGameSession() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [game, setGame] = useState<GameState | null>(null);
  const [overlay, setOverlay] = useState<HeatmapOverlay | null>(null);
  const [nnEnabled, setNnEnabled] = useState(false);
  const [manualResearchEnabled, setManualResearchEnabled] = useState(false);
  const [aiSide, setAiSide] = useState<AiSide>(() => {
    const stored = window.localStorage.getItem(AI_SIDE_STORAGE_KEY);
    return stored === 'black' || stored === 'white' ? stored : 'none';
  });
  const [selectedModelPath, setSelectedModelPath] = useState('');
  const [selectedSimulations, setSelectedSimulations] = useState(DEFAULT_SIMULATIONS);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function syncGame(nextGame: GameState) {
    setGame(nextGame);
    setSelectedModelPath(nextGame.model_path);
    setSelectedSimulations(nextGame.simulations);
    window.localStorage.setItem(GAME_STORAGE_KEY, nextGame.id);
  }

  async function restoreOrCreate(availableModels: ModelInfo[]) {
    const fallbackModel = availableModels[0]?.path;
    const storedDefaultModel = window.localStorage.getItem(DEFAULT_MODEL_STORAGE_KEY);
    const storedDefaultSimulations = window.localStorage.getItem(DEFAULT_SIMULATIONS_STORAGE_KEY);

    if (
      (fallbackModel && storedDefaultModel !== fallbackModel) ||
      storedDefaultSimulations !== String(DEFAULT_SIMULATIONS)
    ) {
      window.localStorage.removeItem(GAME_STORAGE_KEY);
      const created = await createGame({ model_path: fallbackModel, simulations: DEFAULT_SIMULATIONS });
      syncGame(created);
      window.localStorage.setItem(DEFAULT_MODEL_STORAGE_KEY, fallbackModel);
      window.localStorage.setItem(DEFAULT_SIMULATIONS_STORAGE_KEY, String(DEFAULT_SIMULATIONS));
      return;
    }

    const storedId = window.localStorage.getItem(GAME_STORAGE_KEY);
    if (storedId) {
      try {
        const restored = await getGame(storedId);
        syncGame(restored);
        if (fallbackModel) {
          window.localStorage.setItem(DEFAULT_MODEL_STORAGE_KEY, fallbackModel);
        }
        window.localStorage.setItem(DEFAULT_SIMULATIONS_STORAGE_KEY, String(DEFAULT_SIMULATIONS));
        return;
      } catch (restoreError) {
        if (!(restoreError instanceof ApiError && restoreError.status === 404)) {
          setError(describeError(restoreError));
        }
      }
    }

    const created = await createGame({ model_path: fallbackModel, simulations: DEFAULT_SIMULATIONS });
    syncGame(created);
    if (fallbackModel) {
      window.localStorage.setItem(DEFAULT_MODEL_STORAGE_KEY, fallbackModel);
    }
    window.localStorage.setItem(DEFAULT_SIMULATIONS_STORAGE_KEY, String(DEFAULT_SIMULATIONS));
  }

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
        setSelectedModelPath(availableModels[0].path);
        await restoreOrCreate(availableModels);
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
  }, []);

  useEffect(() => {
    window.localStorage.setItem(AI_SIDE_STORAGE_KEY, aiSide);
  }, [aiSide]);

  const research = useResearch(
    game?.id ?? null,
    (frame: ResearchFrame) => {
      syncGame(frame.game);
      if (manualResearchEnabled) {
        setOverlay({
          kind: 'research',
          primaryLabel: 'Visit',
          secondaryLabel: 'Value',
          primaryMatrix: frame.visit_matrix,
          valueMatrix: frame.value_matrix,
          visitCount: frame.visit_count,
          value: frame.value,
        });
      } else if (overlay?.kind === 'research') {
        setOverlay(null);
      }
    },
    (message) => setError(message),
  );

  const autoplay = useAutoplay(
    game?.id ?? null,
    (frame: AutoplayFrame) => {
      syncGame(frame.game);
      if (frame.done) {
        setOverlay(null);
      }
    },
    (message) => setError(message),
  );

  const aiMoveSearch = useAiMove(
    game?.id ?? null,
    (frame: AiMoveFrame) => {
      syncGame(frame.game);
      if (!frame.done) {
        setOverlay({
          kind: 'research',
          primaryLabel: 'Visit',
          secondaryLabel: 'Value',
          primaryMatrix: frame.visit_matrix,
          valueMatrix: frame.value_matrix,
          visitCount: frame.visit_count,
          value: frame.value,
        });
      } else {
        setOverlay(null);
      }
    },
    (message) => setError(message),
  );

  useEffect(() => {
    if (!manualResearchEnabled && overlay?.kind === 'research') {
      setOverlay(null);
    }
  }, [manualResearchEnabled, overlay?.kind]);

  const isAITurn =
    !!game &&
    aiSide !== 'none' &&
    game.status === 'active' &&
    ((aiSide === 'black' && game.current_player === 1) || (aiSide === 'white' && game.current_player === -1));

  const shouldAutoPonder = false;

  useEffect(() => {
    if (!game || game.status !== 'active' || busy || loading || autoplay.active || aiMoveSearch.active) {
      if (research.active) {
        research.stop();
      }
      return;
    }

    if (manualResearchEnabled || shouldAutoPonder) {
      research.start();
      return;
    }

    if (research.active) {
      research.stop();
    }
  }, [
    autoplay.active,
    aiMoveSearch.active,
    busy,
    game?.id,
    game?.status,
    loading,
    manualResearchEnabled,
    research,
    shouldAutoPonder,
  ]);

  useEffect(() => {
    let cancelled = false;

    async function refreshNNOverlay() {
      if (!game || !nnEnabled || research.active || autoplay.active || aiMoveSearch.active) {
        if (!nnEnabled && overlay?.kind === 'nn') {
          setOverlay(null);
        }
        return;
      }

      try {
        const result = await getNN(game.id);
        if (cancelled) {
          return;
        }
        setOverlay({
          kind: 'nn',
          primaryLabel: 'Prob',
          secondaryLabel: 'Value',
          primaryMatrix: result.policy_matrix,
          valueMatrix: result.value_matrix,
        });
      } catch (nnError) {
        if (!cancelled) {
          setError(describeError(nnError));
        }
      }
    }

    void refreshNNOverlay();
    return () => {
      cancelled = true;
    };
  }, [
    autoplay.active,
    aiMoveSearch.active,
    game?.current_player,
    game?.history_index,
    game?.id,
    game?.status,
    nnEnabled,
    overlay?.kind,
    research.active,
  ]);

  async function runGameAction(action: () => Promise<GameState>, clearOverlay = true) {
    setBusy(true);
    setError(null);
    try {
      const nextGame = await action();
      syncGame(nextGame);
      if (clearOverlay) {
        setOverlay(null);
      }
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

  async function handleCellClick(row: number, col: number) {
    if (!game || busy || autoplay.active || aiMoveSearch.active || game.status !== 'active') {
      return;
    }
    if ((aiSide === 'black' && game.current_player === 1) || (aiSide === 'white' && game.current_player === -1)) {
      return;
    }
    if (game.board[row]?.[col] !== 0) {
      return;
    }
    await runGameAction(() => makeMove(game.id, row, col));
  }

  async function handleAIMove() {
    if (!game || aiMoveSearch.active) {
      return;
    }
    if (research.active) {
      research.stop();
    }
    setError(null);
    setOverlay(null);
    aiMoveSearch.start();
  }

  useEffect(() => {
    if (
      !game ||
      loading ||
      busy ||
      autoplay.active ||
      aiMoveSearch.active ||
      research.active ||
      game.status !== 'active' ||
      aiSide === 'none'
    ) {
      return;
    }

    if ((aiSide === 'black' && game.current_player !== 1) || (aiSide === 'white' && game.current_player !== -1)) {
      return;
    }

    aiMoveSearch.start();
  }, [
    aiSide,
    autoplay.active,
    aiMoveSearch.active,
    busy,
    game?.current_player,
    game?.history_index,
    game?.id,
    game?.status,
    loading,
    research.active,
  ]);

  async function handleUndo() {
    if (!game) {
      return;
    }
    await runGameAction(() => undoMove(game.id));
  }

  async function handleRedo() {
    if (!game) {
      return;
    }
    await runGameAction(() => redoMove(game.id));
  }

  async function handlePass() {
    if (!game || busy || autoplay.active || aiMoveSearch.active || game.status !== 'active') {
      return;
    }
    if ((aiSide === 'black' && game.current_player === 1) || (aiSide === 'white' && game.current_player === -1)) {
      return;
    }
    await runGameAction(() => passMove(game.id));
  }

  async function handleResign() {
    if (!game || busy || autoplay.active || aiMoveSearch.active || game.status !== 'active') {
      return;
    }
    if ((aiSide === 'black' && game.current_player === 1) || (aiSide === 'white' && game.current_player === -1)) {
      return;
    }
    await runGameAction(() => resignGame(game.id));
  }

  async function toggleNN() {
    if (!game || busy || research.active || autoplay.active || aiMoveSearch.active) {
      return;
    }
    if (nnEnabled) {
      setNnEnabled(false);
      setOverlay(null);
      return;
    }
    if (game.status !== 'active') {
      return;
    }
    setError(null);
    setNnEnabled(true);
  }

  function toggleResearch() {
    setManualResearchEnabled(false);
    setOverlay(null);
  }

  function toggleAutoplay() {
    if (aiMoveSearch.active) {
      return;
    }
    if (manualResearchEnabled) {
      setManualResearchEnabled(false);
      setOverlay(null);
    }
    if (autoplay.active) {
      autoplay.stop();
      return;
    }
    setOverlay(null);
    autoplay.start();
  }

  const summary = {
    currentPlayer: describeCurrentPlayer(game),
    moveCount: String(game?.history_index !== undefined ? game.history_index + 1 : 0),
    searchVisits: String(game?.search_visits ?? 0),
    searchMode: aiMoveSearch.active ? 'KataGo 思考中' : autoplay.active ? '自动对弈中' : '等待操作',
  };

  return {
    models,
    game,
    overlay,
    aiSide,
    selectedModelPath,
    selectedSimulations,
    loading,
    busy: busy || aiMoveSearch.active,
    error,
    researchActive: manualResearchEnabled,
    autoplayActive: autoplay.active,
    aiMoveActive: aiMoveSearch.active,
    showNNActive: nnEnabled,
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
    toggleResearch,
    toggleAutoplay,
    toggleNN,
    summary,
  };
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
