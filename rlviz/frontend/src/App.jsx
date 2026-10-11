import { useEffect, useState } from "react";
import { useSimulation, WS_URL } from "./useSimulation";
import ClusterCanvas from "./components/ClusterCanvas";
import AgentPanel from "./components/AgentPanel";
import LineChart from "./components/LineChart";
import Inspector from "./components/Inspector";
import Controls from "./components/Controls";

const pct = (v) => `${(v * 100).toFixed(1)}%`;

function movingAvg(arr, w) {
  return arr.map((_, i) => {
    const s = Math.max(0, i - w + 1);
    let sum = 0;
    for (let j = s; j <= i; j++) sum += arr[j];
    return sum / (i - s + 1);
  });
}

function Kpi({ label, value, sub, tone }) {
  return (
    <div className={`kpi ${tone || ""}`}>
      <span className="muted small">{label}</span>
      <b>{value}</b>
      <span className="muted small">{sub}</span>
    </div>
  );
}

function Legend() {
  return (
    <div className="legend">
      <span><i className="dot green" />Healthy</span>
      <span><i className="dot red" />Failed</span>
      <span><i className="dot amber" />Recovering</span>
      <span><i className="ln grey" />Traffic</span>
      <span><i className="ln blue" />Degraded traffic</span>
      <span><i className="halo" />Cluster</span>
    </div>
  );
}

export default function App() {
  const { status, latest, history, send } = useSimulation();
  const [cursorT, setCursorT] = useState(null);
  const [selected, setSelected] = useState(null);
  const hist = history.current;

  useEffect(() => {
    setCursorT(null);
    setSelected(null);
  }, [latest?.run]);

  if (!latest) {
    return (
      <div className="splash">
        <h1>ClusterRL</h1>
        <p>Waiting for the backend at <code>{WS_URL}</code></p>
        <pre>cd backend{"\n"}pip install -r requirements.txt{"\n"}uvicorn main:app --port 8000</pre>
      </div>
    );
  }

  let view = latest;
  if (cursorT != null && hist.length) {
    const i = cursorT - hist[0].t;
    if (hist[i] && hist[i].t === cursorT) view = hist[i];
  }
  const replay = view !== latest;
  const viewIdx = replay ? hist.indexOf(view) : hist.length - 1;
  const upto = viewIdx + 1;
  const slice = hist.slice(Math.max(0, upto - 300), upto);
  const rlSeries = movingAvg(slice.map((f) => f.rl.failure_rate * 100), 8);
  const baseSeries = movingAvg(slice.map((f) => f.baseline.failure_rate * 100), 8);
  const recent = hist.slice(Math.max(0, upto - 200), upto);
  const sumRL = recent.reduce((a, f) => a + f.rl.failure_rate, 0);
  const sumB = recent.reduce((a, f) => a + f.baseline.failure_rate, 0);
  const improvement = sumB > 0 ? 1 - sumRL / sumB : 0;

  const selNode = selected != null ? view.rl.nodes.find((n) => n.id === selected) : null;
  const selCluster = selNode ? view.rl.clusters.find((c) => c.id === selNode.cluster) : null;

  const inject = () => send({ cmd: "inject", cluster: selNode ? selNode.cluster : undefined });
  const onScrub = (e) => {
    const idx = Number(e.target.value);
    if (idx >= hist.length - 1) { setCursorT(null); return; }
    setCursorT(hist[idx].t);
    if (latest.playing) send({ cmd: "pause" });
  };
  const goLive = () => { setCursorT(null); send({ cmd: "play" }); };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" />
          <div>
            <h1>ClusterRL</h1>
            <p>Reinforcement-learned container placement · real Google Borg 2011 workload</p>
          </div>
        </div>
        <Controls
          status={status} playing={latest.playing} speed={latest.speed}
          onPlay={() => send({ cmd: "play" })} onPause={() => send({ cmd: "pause" })}
          onSpeed={(v) => send({ cmd: "speed", value: v })}
          onInject={inject} onReset={() => send({ cmd: "reset" })}
          injectLabel={selNode ? `Fail cluster ${selCluster?.label}` : "Inject failure"}
        />
      </header>

      <main className="main">
        <section className="panel stage">
          <ClusterCanvas frame={view.rl} selectedId={selected} onSelect={setSelected} />
          <div className="stage-title">
            <h2>Live cluster topology</h2>
            <span className="badge">{view.rl.method}</span>
            <span className="badge badge-dim">t = {view.t}</span>
          </div>
          {replay && <div className="replay-badge">REPLAY</div>}
          <Legend />
          {selNode && <Inspector node={selNode} cluster={selCluster} onClose={() => setSelected(null)} />}
        </section>

        <aside className="side">
          <div className="kpis">
            <Kpi label="Failure rate · RL" value={pct(view.rl.failure_rate)}
              sub={`${view.rl.failed_count} / ${view.rl.nodes.length} down`}
              tone={view.rl.failure_rate > 0.15 ? "bad" : "ok"} />
            <Kpi label="Failure rate · baseline" value={pct(view.baseline.failure_rate)}
              sub={`${view.baseline.failed_count} / ${view.baseline.nodes.length} down`} />
            <Kpi label="Improvement (last 200 ticks)"
              value={`${improvement >= 0 ? "+" : ""}${(improvement * 100).toFixed(0)}%`}
              sub="fewer failures vs baseline" tone={improvement > 0 ? "ok" : "bad"} />
            <Kpi label="Traffic load" value={`${view.rl.traffic.toFixed(2)}×`} sub={view.agent.state_label} />
          </div>

          <AgentPanel agent={view.agent} />

          <section className="panel pad">
            <LineChart title="Failure rate (smoothed)" unit="%" yMin={0}
              series={[
                { name: "RL", color: "#22c55e", data: rlSeries },
                { name: "Baseline", color: "#94a3b8", data: baseSeries },
              ]} />
          </section>
          <section className="panel pad">
            <LineChart title="Reward per decision" digits={2} height={90}
              series={[{ name: "reward", color: "#a78bfa", data: view.agent.rewards }]} />
          </section>

          <section className="panel baseline">
            <ClusterCanvas compact frame={view.baseline}>
              <div className="stage-title small-title">
                <h2>Baseline</h2>
                <span className="badge badge-dim">{view.baseline_name}</span>
              </div>
            </ClusterCanvas>
          </section>
        </aside>
      </main>

      <footer className="timeline panel">
        <span className="muted small mono">t={hist[0]?.t ?? 0}</span>
        <input type="range" min={0} max={Math.max(hist.length - 1, 0)} value={viewIdx} onChange={onScrub} />
        <span className="muted small mono">t={view.t}</span>
        <button className={`btn ${replay ? "btn-live" : ""}`} onClick={goLive}>● {replay ? "Resume live" : "Live"}</button>
      </footer>
    </div>
  );
}
