import { useCallback, useEffect, useRef, useState } from "react";

export const WS_URL =
  import.meta.env.VITE_WS_URL || `ws://${window.location.hostname}:8000/ws`;
const MAX_HISTORY = 4000;

/** Connects to the backend and keeps a rolling history of frames. */
export function useSimulation() {
  const [status, setStatus] = useState("connecting");
  const [latest, setLatest] = useState(null);
  const history = useRef([]);
  const ws = useRef(null);

  useEffect(() => {
    let closed = false;
    let retry;
    const connect = () => {
      const sock = new WebSocket(WS_URL);
      ws.current = sock;
      sock.onopen = () => setStatus("live");
      sock.onmessage = (e) => {
        const f = JSON.parse(e.data);
        if (f.type !== "frame") return;
        const h = history.current;
        if (h.length && h[h.length - 1].run !== f.run) h.length = 0;
        // snapshot sent while paused repeats the same tick: replace instead of append
        if (h.length && h[h.length - 1].t === f.t && h[h.length - 1].run === f.run) h[h.length - 1] = f;
        else h.push(f);
        if (h.length > MAX_HISTORY) h.shift();
        setLatest(f);
      };
      sock.onclose = () => {
        setStatus("offline");
        if (!closed) retry = setTimeout(connect, 1500);
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      ws.current?.close();
    };
  }, []);

  const send = useCallback((msg) => {
    const s = ws.current;
    if (s && s.readyState === 1) s.send(JSON.stringify(msg));
  }, []);

  return { status, latest, history, send };
}
