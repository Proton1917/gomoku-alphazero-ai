import type { MouseEvent } from 'react';
import { useEffect, useRef, useState } from 'react';

import { drawBoardScene, getCellFromPoint } from '../utils/boardRenderer';

interface BoardCanvasProps {
  board: number[][];
  lastMove: number[] | null;
  onCellClick?: (row: number, col: number) => void;
  disabled?: boolean;
}

export function BoardCanvas({ board, lastMove, onCellClick, disabled = false }: BoardCanvasProps) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [canvasSize, setCanvasSize] = useState(620);

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

  // 尺寸变化才重设画布位图（对 canvas.width 赋值会清空整个位图，代价高）
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
  }, [canvasSize]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawBoardScene(context, canvasSize, board, lastMove);
  }, [board, canvasSize, lastMove]);

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

  return (
    <div className={`board-panel ${disabled ? 'is-disabled' : ''}`} ref={wrapperRef}>
      <canvas ref={canvasRef} className="board-canvas" onClick={handleClick} />
    </div>
  );
}
