import { BoardCanvas } from './components/BoardCanvas';
import { ControlPanel } from './components/ControlPanel';
import { ModelSelector } from './components/ModelSelector';
import { StatusBar } from './components/StatusBar';
import { useGameSession } from './hooks/useGameSession';

export default function App() {
  const game = useGameSession();

  return (
    <div className="app-shell">
      <div className="atmosphere atmosphere-left" />
      <div className="atmosphere atmosphere-right" />

      <header className="masthead">
        <div>
          <span className="eyebrow">KataGo Local GUI</span>
          <h1>KataGo 对局台</h1>
        </div>
        <p>
          19 路围棋棋盘，后端直接调用本机 Metal 版 KataGo。点棋盘下人类手，点 AI 落子或自动对弈让 KataGo 走。
        </p>
      </header>

      <main className="workspace">
          <aside className="sidebar-column">
            <ModelSelector
              disabled={game.loading || game.busy || game.autoplayActive}
              label="KataGo 引擎"
              models={game.models}
              value={game.selectedModelPath}
              onChange={game.setSelectedModelPath}
              actionLabel="创建新对局"
              onAction={game.createNewGame}
              summary="默认加载本地 KataGo 测试模型，搜索步长对应 KataGo maxVisits。"
            />

            <StatusBar
              eyebrow="Session"
              title="局面摘要"
              items={[
                { label: '当前执手', value: game.summary.currentPlayer },
                { label: '总步数', value: game.summary.moveCount },
                { label: '搜索步数', value: game.summary.searchVisits },
              ]}
              message={game.error}
            />

            <ControlPanel
              aiSide={game.aiSide}
              autoplayActive={game.autoplayActive}
              busy={game.busy || game.loading}
              canRedo={game.game?.can_redo ?? false}
              canUndo={game.game?.can_undo ?? false}
              gameFinished={game.game?.status === 'finished'}
              onAiSideChange={game.setAiSide}
              onAIMove={game.handleAIMove}
              onPass={game.handlePass}
              onResign={game.handleResign}
              onRedo={game.handleRedo}
              onToggleAutoplay={game.toggleAutoplay}
              onUndo={game.handleUndo}
            />
          </aside>

          <section className="board-column">
            <div className="battle-stage">
              <div className="stage-header">
                <span className="eyebrow">Board</span>
                <h2>19 路棋盘</h2>
                <p>黑白双方按围棋规则交替行棋，后端会自动处理提子和禁自杀。</p>
              </div>
              <BoardCanvas
                board={game.game?.board ?? emptyBoard()}
                disabled={game.loading || game.busy || game.autoplayActive}
                heatmap={game.overlay}
                lastMove={game.game?.last_move ?? null}
                onCellClick={game.handleCellClick}
              />
            </div>
          </section>

          <aside className="notes-column">
            <section className="status-bar">
              <div className="panel-header">
                <span className="eyebrow">Rules</span>
                <h3>规则状态</h3>
              </div>
              <div className="legend-stack">
                <p><strong>棋盘:</strong> 19 路。</p>
                <p><strong>规则:</strong> 自动提子，禁止自杀手。</p>
                <p><strong>终局:</strong> 连续两次停一手后结束。</p>
              </div>
            </section>
          </aside>
        </main>
    </div>
  );
}

function emptyBoard() {
  return Array.from({ length: 19 }, () => Array.from({ length: 19 }, () => 0));
}
