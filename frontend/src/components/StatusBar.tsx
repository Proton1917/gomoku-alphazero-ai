interface StatusItem {
  label: string;
  value: string;
}

interface StatusBarProps {
  eyebrow: string;
  title: string;
  items: StatusItem[];
  message?: string | null;
}

export function StatusBar({ eyebrow, title, items, message }: StatusBarProps) {
  return (
    <section className="status-bar">
      <div className="panel-header">
        <span className="eyebrow">{eyebrow}</span>
        <h3>{title}</h3>
      </div>
      <div className="status-grid">
        {items.map((item) => (
          <article key={`${item.label}-${item.value}`} className="status-card">
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </article>
        ))}
      </div>
      {message ? <p className="status-message">{message}</p> : null}
    </section>
  );
}
