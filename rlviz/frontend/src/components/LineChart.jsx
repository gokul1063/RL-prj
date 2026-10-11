/** Minimal dependency-free SVG line chart. series: [{name,color,data[]}] */
export default function LineChart({ title, series, unit = "", height = 110, digits = 1, yMin, yMax }) {
  const W = 400;
  const H = height;
  const all = series.flatMap((s) => s.data).filter((v) => Number.isFinite(v));
  let lo = yMin ?? (all.length ? Math.min(...all) : 0);
  let hi = yMax ?? (all.length ? Math.max(...all) : 1);
  if (hi - lo < 1e-6) hi = lo + 1;
  const pad = (hi - lo) * 0.08;
  lo -= yMin === undefined ? pad : 0;
  hi += pad;
  const X = (i, n) => (n <= 1 ? 0 : (i / (n - 1)) * W);
  const Y = (v) => H - ((v - lo) / (hi - lo)) * H;

  return (
    <div className="chart">
      <div className="chart-head">
        <span className="chart-title">{title}</span>
        <span className="chart-legend">
          {series.map((s) => {
            const last = s.data[s.data.length - 1];
            return (
              <span key={s.name} className="legend-item">
                <i style={{ background: s.color }} />
                {s.name}
                <b>{last === undefined ? "–" : `${last.toFixed(digits)}${unit}`}</b>
              </span>
            );
          })}
        </span>
      </div>
      <div className="chart-body" style={{ height }}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" width="100%" height="100%">
          {[0.25, 0.5, 0.75].map((g) => (
            <line key={g} x1="0" x2={W} y1={H * g} y2={H * g} className="grid" vectorEffect="non-scaling-stroke" />
          ))}
          {series.map((s, si) => {
            const n = s.data.length;
            if (n < 2) return null;
            const pts = s.data.map((v, i) => `${X(i, n).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
            return (
              <g key={s.name}>
                {si === 0 && <polygon points={`0,${H} ${pts} ${W},${H}`} fill={s.color} opacity="0.1" />}
                <polyline points={pts} fill="none" stroke={s.color} strokeWidth="1.8"
                  strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
              </g>
            );
          })}
        </svg>
        <span className="axis axis-hi">{hi.toFixed(digits)}{unit}</span>
        <span className="axis axis-lo">{lo.toFixed(digits)}{unit}</span>
      </div>
    </div>
  );
}
