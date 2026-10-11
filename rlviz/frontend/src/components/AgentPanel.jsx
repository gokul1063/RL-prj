export default function AgentPanel({ agent }) {
  const { q, visits, actions, action_names, state_labels, state } = agent;
  const seen = [];
  q.forEach((row, i) => row.forEach((v, j) => visits[i][j] > 0 && seen.push(v)));
  const lo = seen.length ? Math.min(...seen) : 0;
  const hi = seen.length ? Math.max(...seen) : 0;
  const color = (v) => {
    const t = hi === lo ? 0.5 : (v - lo) / (hi - lo);
    return `hsl(${Math.round(t * 130)} 62% ${24 + t * 6}%)`;
  };
  const best = q.map((row, i) => {
    let b = -1;
    row.forEach((v, j) => { if (visits[i][j] > 0 && (b < 0 || v > row[b])) b = j; });
    return b;
  });
  const progress = ((agent.window - agent.next_in) / agent.window) * 100;

  return (
    <section className="panel agent">
      <div className="panel-head">
        <h3>RL agent</h3>
        <span className={`tag ${agent.explored ? "tag-explore" : "tag-exploit"}`}>
          {agent.explored ? "EXPLORING" : "EXPLOITING"}
        </span>
      </div>

      <div className="action">
        <div className="muted small">Chosen clustering method</div>
        <div className="action-name">{agent.action_name}</div>
        <div className="bar"><i style={{ width: `${progress}%` }} /></div>
        <div className="muted small">next decision in {agent.next_in} ticks · state: {agent.state_label}</div>
      </div>

      <div className="agent-stats">
        <div><span className="muted small">epsilon</span><b>{agent.epsilon.toFixed(2)}</b></div>
        <div><span className="muted small">last reward</span><b>{agent.reward.toFixed(2)}</b></div>
        <div><span className="muted small">cumulative</span><b>{agent.cum_reward.toFixed(0)}</b></div>
        <div><span className="muted small">decisions</span><b>{agent.decisions}</b></div>
      </div>

      <div className="muted small qtitle">Learned Q-values (traffic state × method)</div>
      <table className="qtable">
        <thead>
          <tr>
            <th />
            {actions.map((a, j) => <th key={a} title={action_names[j]}>{a}</th>)}
          </tr>
        </thead>
        <tbody>
          {q.map((row, i) => (
            <tr key={i} className={i === state ? "cur" : ""}>
              <th>{state_labels[i]}</th>
              {row.map((v, j) => {
                const v0 = visits[i][j] === 0;
                return (
                  <td key={j} className={best[i] === j ? "best" : ""}
                    style={v0 ? undefined : { background: color(v) }}
                    title={`${action_names[j]} · visits ${visits[i][j]}`}>
                    {v0 ? "·" : v.toFixed(1)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
