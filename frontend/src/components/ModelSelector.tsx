import type { ModelInfo } from '../types/game';

interface ModelSelectorProps {
  label: string;
  models: ModelInfo[];
  value: string;
  onChange: (value: string) => void;
  simulations?: number;
  onSimulationsChange?: (value: number) => void;
  actionLabel?: string;
  onAction?: () => void;
  disabled?: boolean;
  summary?: string;
}

export function ModelSelector({
  label,
  models,
  value,
  onChange,
  simulations,
  onSimulationsChange,
  actionLabel,
  onAction,
  disabled = false,
  summary,
}: ModelSelectorProps) {
  return (
    <section className="selector-panel">
      <div className="panel-header">
        <span className="eyebrow">Model Bank</span>
        <h3>{label}</h3>
      </div>
      {summary ? <p className="panel-summary">{summary}</p> : null}
      <label className="field-label">
        <span>模型</span>
        <select disabled={disabled} value={value} onChange={(event) => onChange(event.target.value)}>
          {models.map((model) => (
            <option key={model.path} value={model.path}>
              {model.name}
            </option>
          ))}
        </select>
      </label>
      {typeof simulations === 'number' && onSimulationsChange ? (
        <label className="field-label">
          <span>搜索步长</span>
          <input
            disabled={disabled}
            min={1}
            max={2000}
            type="number"
            value={simulations}
            onChange={(event) => onSimulationsChange(Number(event.target.value || 1))}
          />
        </label>
      ) : null}
      {actionLabel && onAction ? (
        <button className="accent-button" disabled={disabled || models.length === 0} onClick={onAction}>
          {actionLabel}
        </button>
      ) : null}
    </section>
  );
}
