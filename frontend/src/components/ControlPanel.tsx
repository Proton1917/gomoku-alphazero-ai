import type { AiSide } from '../types/game';

interface ControlPanelProps {
  aiSide: AiSide;
  autoplayActive: boolean;
  canUndo: boolean;
  canRedo: boolean;
  busy: boolean;
  gameFinished: boolean;
  onAiSideChange: (value: AiSide) => void;
  onToggleAutoplay: () => void;
  onAIMove: () => void;
  onPass: () => void;
  onResign: () => void;
  onUndo: () => void;
  onRedo: () => void;
}

export function ControlPanel({
  aiSide,
  autoplayActive,
  canUndo,
  canRedo,
  busy,
  gameFinished,
  onAiSideChange,
  onToggleAutoplay,
  onAIMove,
  onPass,
  onResign,
  onUndo,
  onRedo,
}: ControlPanelProps) {
  return (
    <section className="control-panel">
      <div className="panel-header">
        <span className="eyebrow">Controls</span>
        <h3>操作台</h3>
      </div>
      <label className="field-label">
        <span>电脑执哪边</span>
        <select disabled={busy || autoplayActive} value={aiSide} onChange={(event) => onAiSideChange(event.target.value as AiSide)}>
          <option value="none">双方手动</option>
          <option value="black">电脑执黑</option>
          <option value="white">电脑执白</option>
        </select>
      </label>
      <div className="control-grid">
        <button className={autoplayActive ? 'is-active' : ''} disabled={busy || gameFinished} onClick={onToggleAutoplay}>
          {autoplayActive ? '停止自动' : '自动对弈'}
        </button>
        <button disabled={busy || gameFinished || autoplayActive} onClick={onAIMove}>
          AI 落子
        </button>
        <button disabled={busy || gameFinished || autoplayActive} onClick={onPass}>
          停一手
        </button>
        <button disabled={busy || gameFinished || autoplayActive} onClick={onResign}>
          认输
        </button>
        <button disabled={busy || !canUndo || autoplayActive} onClick={onUndo}>
          悔棋
        </button>
        <button disabled={busy || !canRedo || autoplayActive} onClick={onRedo}>
          前进
        </button>
      </div>
    </section>
  );
}
