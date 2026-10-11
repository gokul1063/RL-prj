const SPEEDS = [1, 2, 5, 10];

export default function Controls({ status, playing, speed, onPlay, onPause, onSpeed, onInject, onReset, injectLabel }) {
  return (
    <div className="controls">
      <span className={`conn conn-${status}`}>
        <i /> {status === "live" ? "connected" : status}
      </span>
      <button className="btn" onClick={playing ? onPause : onPlay}>
        {playing ? "❚❚ Pause" : "▶ Play"}
      </button>
      <div className="seg">
        {SPEEDS.map((v) => (
          <button key={v} className={Math.abs(speed - v) < 0.01 ? "on" : ""} onClick={() => onSpeed(v)}>{v}×</button>
        ))}
      </div>
      <button className="btn btn-danger" onClick={onInject}>⚡ {injectLabel}</button>
      <button className="btn" onClick={onReset}>↺ Reset</button>
    </div>
  );
}
