import { useEffect, useRef, useState } from 'react';

import { ApiError, createBattle, describeError, getBattle, listModels } from '../api/client';
import { openJsonSocket, type SocketHandle } from '../api/websocket';
import type { BattleFrame, BattleState, ModelInfo } from '../types/game';
import { BoardCanvas } from './BoardCanvas';
import { ModelSelector } from './ModelSelector';
import { StatusBar } from './StatusBar';


const BATTLE_STORAGE_KEY = 'gomoku:web:battle-id';
const BATTLE_DEFAULT_MODEL_STORAGE_KEY = 'gomoku:web:battle-default-model';
const BATTLE_DEFAULT_SIMULATIONS_STORAGE_KEY = 'gomoku:web:battle-default-simulations';
const DEFAULT_SIMULATIONS = 128;


export function BattlePage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [battle, setBattle] = useState<BattleState | null>(null);
  const [blackModelPath, setBlackModelPath] = useState('');
  const [whiteModelPath, setWhiteModelPath] = useState('');
  const [simulations, setSimulations] = useState(DEFAULT_SIMULATIONS);
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<SocketHandle | null>(null);

  function syncBattle(nextBattle: BattleState) {
    setBattle(nextBattle);
    setBlackModelPath(nextBattle.black_model_path);
    setWhiteModelPath(nextBattle.white_model_path);
    setSimulations(nextBattle.simulations);
    window.localStorage.setItem(BATTLE_STORAGE_KEY, nextBattle.id);
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
          setError('没有找到可用于 Battle 的模型');
          return;
        }

        const defaultBlackModel = availableModels[0]?.path ?? '';
        const defaultWhiteModel = availableModels[1]?.path ?? availableModels[0]?.path ?? '';
        const storedDefaultModel = window.localStorage.getItem(BATTLE_DEFAULT_MODEL_STORAGE_KEY);
        const storedDefaultSimulations = window.localStorage.getItem(BATTLE_DEFAULT_SIMULATIONS_STORAGE_KEY);

        if (
          storedDefaultModel !== `${defaultBlackModel}::${defaultWhiteModel}` ||
          storedDefaultSimulations !== String(DEFAULT_SIMULATIONS)
        ) {
          window.localStorage.removeItem(BATTLE_STORAGE_KEY);
        }

        const storedId = window.localStorage.getItem(BATTLE_STORAGE_KEY);
        if (storedId) {
          try {
            const restored = await getBattle(storedId);
            if (!cancelled) {
              syncBattle(restored);
              window.localStorage.setItem(BATTLE_DEFAULT_MODEL_STORAGE_KEY, `${defaultBlackModel}::${defaultWhiteModel}`);
              window.localStorage.setItem(BATTLE_DEFAULT_SIMULATIONS_STORAGE_KEY, String(DEFAULT_SIMULATIONS));
              return;
            }
          } catch (restoreError) {
            if (!(restoreError instanceof ApiError && restoreError.status === 404) && !cancelled) {
              setError(describeError(restoreError));
            }
          }
        }

        const created = await createBattle({
          black_model_path: defaultBlackModel,
          white_model_path: defaultWhiteModel,
          simulations: DEFAULT_SIMULATIONS,
        });
        if (!cancelled) {
          syncBattle(created);
          window.localStorage.setItem(BATTLE_DEFAULT_MODEL_STORAGE_KEY, `${defaultBlackModel}::${defaultWhiteModel}`);
          window.localStorage.setItem(BATTLE_DEFAULT_SIMULATIONS_STORAGE_KEY, String(DEFAULT_SIMULATIONS));
        }
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
      socketRef.current?.close();
    };
  }, []);

  function stopBattle() {
    socketRef.current?.close();
    socketRef.current = null;
    setStreaming(false);
  }

  function startBattle() {
    if (!battle || socketRef.current) {
      return;
    }
    setError(null);
    socketRef.current = openJsonSocket<BattleFrame>(`/ws/battle/${battle.id}`, {
      onMessage: (frame) => {
        syncBattle(frame.battle);
        if (frame.done) {
          setStreaming(false);
          socketRef.current = null;
        }
      },
      onOpen: () => setStreaming(true),
      onClose: () => {
        setStreaming(false);
        socketRef.current = null;
      },
      onError: () => setError('Battle 流连接失败'),
    });
  }

  async function createNewBattle() {
    setLoading(true);
    setError(null);
    stopBattle();
    try {
      const nextBattle = await createBattle({
        black_model_path: blackModelPath,
        white_model_path: whiteModelPath,
        simulations,
      });
      syncBattle(nextBattle);
    } catch (creationError) {
      setError(describeError(creationError));
    } finally {
      setLoading(false);
    }
  }

  const battleItems = [
    { label: '当前回合', value: battle?.winner === 1 ? '黑棋获胜' : battle?.winner === -1 ? '白棋获胜' : battle?.winner === 0 ? '平局' : battle?.current_player === 1 ? '黑棋' : '白棋' },
    { label: '步数', value: String(battle?.move_count ?? 0) },
    { label: '流状态', value: streaming ? '推演中' : battle?.status === 'finished' ? '已完成' : '待机' },
  ];

  return (
    <section className="battle-layout">
      <aside className="sidebar-column">
        <ModelSelector
          disabled={loading || streaming}
          label="黑棋模型"
          models={models}
          summary="Battle 每步都会按搜索步长持续往上加，直到最佳着法基本稳定后再落子。"
          value={blackModelPath}
          onChange={setBlackModelPath}
        />
        <ModelSelector
          disabled={loading || streaming}
          label="白棋模型"
          models={models}
          value={whiteModelPath}
          onChange={setWhiteModelPath}
          actionLabel="生成新对战"
          onAction={createNewBattle}
        />
        <StatusBar
          eyebrow="Battle Feed"
          title="模型对战状态"
          items={battleItems}
          message={error}
        />
        <section className="control-panel">
          <div className="panel-header">
            <span className="eyebrow">Stream</span>
            <h3>对战流</h3>
          </div>
          <div className="control-grid">
            <button className={streaming ? 'is-active' : ''} disabled={loading || !battle || battle.status === 'finished'} onClick={streaming ? stopBattle : startBattle}>
              {streaming ? '停止 Battle' : '开始 Battle'}
            </button>
            <button disabled={loading || streaming} onClick={createNewBattle}>
              重置对战
            </button>
          </div>
        </section>
      </aside>

      <section className="board-column">
        <div className="battle-stage">
          <div className="stage-header">
            <span className="eyebrow">Autonomous Arena</span>
            <h2>双模型演化观察窗</h2>
            <p>每次连接 Battle WebSocket 后，后端会持续推进一步一步的对局并持久化到 SQLite。</p>
          </div>
          <BoardCanvas board={battle?.board ?? emptyBoard()} disabled={streaming} heatmap={null} lastMove={battle?.last_move ?? null} />
        </div>
      </section>
    </section>
  );
}

function emptyBoard() {
  return Array.from({ length: 15 }, () => Array.from({ length: 15 }, () => 0));
}
