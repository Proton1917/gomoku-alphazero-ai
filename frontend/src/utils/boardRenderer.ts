export const BOARD_SIZE = 19;

export interface BoardGeometry {
  margin: number;
  cellSize: number;
  pieceRadius: number;
}

export function getBoardGeometry(size: number): BoardGeometry {
  const margin = size * 0.08;
  const cellSize = (size - margin * 2) / (BOARD_SIZE - 1);
  return {
    margin,
    cellSize,
    pieceRadius: cellSize * 0.42,
  };
}

export function getCellFromPoint(x: number, y: number, size: number): { row: number; col: number } | null {
  const { margin, cellSize } = getBoardGeometry(size);
  const row = Math.round((y - margin) / cellSize);
  const col = Math.round((x - margin) / cellSize);
  if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) {
    return null;
  }
  return { row, col };
}

export function drawBoardScene(
  context: CanvasRenderingContext2D,
  size: number,
  board: number[][],
  lastMove: number[] | null,
): void {
  const { margin, cellSize, pieceRadius } = getBoardGeometry(size);

  context.clearRect(0, 0, size, size);

  const background = context.createLinearGradient(0, 0, size, size);
  background.addColorStop(0, '#ebd3a0');
  background.addColorStop(1, '#b37b34');
  context.fillStyle = background;
  context.fillRect(0, 0, size, size);

  context.fillStyle = 'rgba(255, 255, 255, 0.05)';
  for (let index = 0; index < size; index += 24) {
    context.fillRect(index, 0, 1, size);
    context.fillRect(0, index, size, 1);
  }

  context.strokeStyle = 'rgba(38, 24, 7, 0.78)';
  context.lineWidth = 1.3;
  context.beginPath();
  for (let index = 0; index < BOARD_SIZE; index += 1) {
    const linePosition = margin + index * cellSize;
    context.moveTo(margin, linePosition);
    context.lineTo(size - margin, linePosition);
    context.moveTo(linePosition, margin);
    context.lineTo(linePosition, size - margin);
  }
  context.stroke();

  const starPoints = [3, 9, 15];
  context.fillStyle = '#2f1a0c';
  for (const row of starPoints) {
    for (const col of starPoints) {
      context.beginPath();
      context.arc(margin + col * cellSize, margin + row * cellSize, 4, 0, Math.PI * 2);
      context.fill();
    }
  }

  context.shadowColor = 'rgba(0, 0, 0, 0.35)';
  context.shadowBlur = 12;
  context.shadowOffsetY = 4;
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      const stone = board[row]?.[col] ?? 0;
      if (stone === 0) {
        continue;
      }
      const centerX = margin + col * cellSize;
      const centerY = margin + row * cellSize;
      const gradient = context.createRadialGradient(
        centerX - pieceRadius * 0.3,
        centerY - pieceRadius * 0.3,
        pieceRadius * 0.2,
        centerX,
        centerY,
        pieceRadius,
      );
      if (stone === 1) {
        gradient.addColorStop(0, '#60646f');
        gradient.addColorStop(0.45, '#17191d');
        gradient.addColorStop(1, '#08090b');
      } else {
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(0.55, '#efe8de');
        gradient.addColorStop(1, '#d6cabf');
      }
      context.fillStyle = gradient;
      context.beginPath();
      context.arc(centerX, centerY, pieceRadius, 0, Math.PI * 2);
      context.fill();

      context.shadowBlur = 0;
      context.strokeStyle = stone === 1 ? 'rgba(255,255,255,0.08)' : 'rgba(33, 27, 18, 0.22)';
      context.lineWidth = 1.2;
      context.stroke();
      context.shadowBlur = 12;
    }
  }

  context.shadowBlur = 0;
  context.shadowOffsetY = 0;
  if (lastMove && lastMove.length === 2 && lastMove[0] >= 0 && lastMove[1] >= 0) {
    const [row, col] = lastMove;
    const centerX = margin + col * cellSize;
    const centerY = margin + row * cellSize;
    context.strokeStyle = '#ff5050';
    context.lineWidth = 2.2;
    context.beginPath();
    context.arc(centerX, centerY, pieceRadius * 0.45, 0, Math.PI * 2);
    context.stroke();
  }
}
