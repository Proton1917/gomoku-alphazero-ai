export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function heatmapColor(value: number, maxValue: number, meanValue: number): string {
  if (maxValue <= 0 || value <= 0) {
    return 'rgba(0,0,0,0)';
  }
  const alpha = 255 * Math.pow(value / maxValue, 0.75);
  const diff = clamp((meanValue - value) * 3 * 255, 0, 255);
  return `rgba(${diff}, ${255 - diff}, 0, ${alpha / 255})`;
}
