import type { MouseEvent } from 'react';
import { useEffect, useRef, useState } from 'react';

import type { HeatmapOverlay } from '../types/game';
import { drawBoardScene, getCellFromPoint } from '../utils/boardRenderer';

interface BoardCanvasProps {
  board: number[][];
  lastMove: number[] | null;
  heatmap?: HeatmapOverlay | null;
  onCellClick?: (row: number, col: number) => void;
  disabled?: boolean;
}

interface HoverState {
  row: number;
  col: number;
  x: number;
  y: number;
}

export function BoardCanvas({ board, lastMove, heatmap, onCellClick, disabled = false }: BoardCanvasProps) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [canvasSize, setCanvasSize] = useState(620);
  const [hover, setHover] = useState<HoverState | null>(null);

  useEffect(() => {
    const element = wrapperRef.current;
    if (!element) {
      return undefined;
    }
    const resizeObserver = new ResizeObserver((entries) => {
      const nextSize = Math.max(320, Math.min(entries[0].contentRect.width, 760));
      setCanvasSize(nextSize);
    });
    resizeObserver.observe(element);
    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasSize * dpr;
    canvas.height = canvasSize * dpr;
    canvas.style.width = `${canvasSize}px`;
    canvas.style.height = `${canvasSize}px`;
    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawBoardScene(context, canvasSize, board, lastMove, heatmap);
  }, [board, canvasSize, heatmap, lastMove]);

  const tooltip = hover && heatmap ? buildTooltip(heatmap, hover.row, hover.col) : null;

  function handleClick(event: MouseEvent<HTMLCanvasElement>) {
    if (!onCellClick || disabled) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const cell = getCellFromPoint(event.clientX - rect.left, event.clientY - rect.top, canvasSize);
    if (!cell) {
      return;
    }
    onCellClick(cell.row, cell.col);
  }

  function handleMouseMove(event: MouseEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const cell = getCellFromPoint(event.clientX - rect.left, event.clientY - rect.top, canvasSize);
    if (!cell) {
      setHover(null);
      return;
    }
    setHover({
      row: cell.row,
      col: cell.col,
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
  }

  return (
    <div className={`board-panel ${disabled ? 'is-disabled' : ''}`} ref={wrapperRef}>
      <canvas
        ref={canvasRef}
        className="board-canvas"
        onClick={handleClick}
        onMouseLeave={() => setHover(null)}
        onMouseMove={handleMouseMove}
      />
      {tooltip && hover ? (
        <div
          className="board-tooltip"
          style={{
            left: Math.min(hover.x + 18, canvasSize - 164),
            top: Math.max(hover.y - 12, 12),
          }}
        >
          <span>{tooltip.primaryLabel}: {tooltip.primaryValue}</span>
          <span>{tooltip.secondaryLabel}: {tooltip.secondaryValue}</span>
        </div>
      ) : null}
      <div className="board-caption">
        <span>{heatmap?.kind === 'research' ? '搜索热度层' : heatmap?.kind === 'nn' ? '神经网络层' : '纯棋盘视图'}</span>
        {heatmap?.visitCount !== undefined ? <span>累计模拟 {heatmap.visitCount}</span> : null}
        {heatmap?.value !== undefined ? <span>局面值 {heatmap.value.toFixed(3)}</span> : null}
      </div>
    </div>
  );
}

function buildTooltip(heatmap: HeatmapOverlay, row: number, col: number) {
  const primary = heatmap.primaryMatrix[row]?.[col] ?? 0;
  const secondary = heatmap.valueMatrix[row]?.[col] ?? 0;
  if (primary <= 0) {
    return null;
  }
  return {
    primaryLabel: heatmap.primaryLabel,
    secondaryLabel: heatmap.secondaryLabel,
    primaryValue: Number.isInteger(primary) ? String(primary) : primary.toFixed(3),
    secondaryValue: secondary.toFixed(3),
  };
}
