const Bar = ({ label, v }) => (
  <div className="ibar">
    <span>{label}</span>
    <div><i style={{ width: `${Math.round(v * 100)}%` }} /></div>
    <em>{Math.round(v * 100)}</em>
  </div>
);

export default function Inspector({ node, cluster, onClose }) {
  return (
    <div className="inspector panel">
      <button className="x" onClick={onClose} aria-label="close">×</button>
      <div className="insp-name">{node.name}</div>
      <div className="muted small">image: {node.image}</div>
      <div className={`pill ${node.failed ? "pill-red" : "pill-green"}`}>{node.failed ? "FAILED" : "HEALTHY"}</div>
      <Bar label="CPU" v={node.cpu} />
      <Bar label="MEM" v={node.mem} />
      <Bar label="NET" v={node.net} />
      <div className="insp-row"><span>demand</span><b>{node.demand.toFixed(2)}×</b></div>
      <div className="insp-row"><span>failures so far</span><b>{node.fails}</b></div>
      {cluster && (
        <>
          <div className="insp-row"><span>cluster</span><b>{cluster.label} ({cluster.size})</b></div>
          <div className="insp-row"><span>cluster load</span><b>{Math.round(cluster.load * 100)}%</b></div>
        </>
      )}
    </div>
  );
}
