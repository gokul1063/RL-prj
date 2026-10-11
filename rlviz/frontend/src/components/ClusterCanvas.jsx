import { useEffect, useRef } from "react";
import { forceSimulation, forceX, forceY, forceCollide } from "d3-force";

const GREEN = [34, 197, 94];
const RED = [239, 68, 68];
const AMBER = [245, 158, 11];
const BLUE = "#3b82f6";
const RECOVER_MS = 1500;

const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const rgba = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

function bezier(p0, c, p1, t) {
  const u = 1 - t;
  return { x: u * u * p0.x + 2 * u * t * c.x + t * t * p1.x, y: u * u * p0.y + 2 * u * t * c.y + t * t * p1.y };
}

/** Clusters are laid out on an ellipse around the ingress hub, far apart. */
function layout(s) {
  s.centers.clear();
  if (!s.frame || !s.w) return;
  const ids = s.frame.clusters.map((c) => c.id).sort((a, b) => a - b);
  const k = ids.length;
  const cx = s.w / 2;
  const cy = s.h / 2;
  const rx = s.w * (s.compact ? 0.33 : 0.35);
  const ry = s.h * (s.compact ? 0.31 : 0.34);
  ids.forEach((id, i) => {
    if (k === 1) return s.centers.set(id, { x: cx, y: cy - ry * 0.6 });
    const a = -Math.PI / 2 + (2 * Math.PI * i) / k;
    s.centers.set(id, { x: cx + Math.cos(a) * rx, y: cy + Math.sin(a) * ry });
  });
}

export default function ClusterCanvas({ frame, compact = false, selectedId = null, onSelect, children }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const ref = useRef(null);
  if (ref.current === null) {
    ref.current = {
      nodes: new Map(), centers: new Map(), halos: new Map(), particles: new Map(),
      w: 0, h: 0, dpr: 1, frame: null, sim: null, dash: 0,
      compact, selected: null, onSelect: null,
    };
  }
  const s = ref.current;
  s.compact = compact;
  s.selected = selectedId;
  s.onSelect = onSelect;

  // ---- mount: simulation, resize, render loop, click
  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    const ctx = canvas.getContext("2d");

    s.sim = forceSimulation([])
      .alphaDecay(0.02)
      .velocityDecay(0.4)
      .alphaTarget(0.012)
      .force("x", forceX((d) => s.centers.get(d.cluster)?.x ?? s.w / 2).strength(0.16))
      .force("y", forceY((d) => s.centers.get(d.cluster)?.y ?? s.h / 2).strength(0.16))
      .force("collide", forceCollide((d) => d.r + 3).strength(0.85));

    const resize = () => {
      const r = wrap.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      s.w = r.width;
      s.h = r.height;
      s.dpr = dpr;
      canvas.width = Math.max(1, Math.floor(r.width * dpr));
      canvas.height = Math.max(1, Math.floor(r.height * dpr));
      layout(s);
      s.sim.nodes([...s.nodes.values()]);
      s.sim.alpha(0.5).restart();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    resize();

    let raf;
    let last = performance.now();
    const draw = (now) => {
      raf = requestAnimationFrame(draw);
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const { w, h, dpr } = s;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const f = s.frame;
      if (!f || !w) return;
      const comp = s.compact;
      const nodes = [...s.nodes.values()];
      const meta = new Map(f.clusters.map((c) => [c.id, c]));

      // ---- cluster halos (smoothed around the live centroid)
      const groups = new Map();
      for (const n of nodes) {
        const g = groups.get(n.cluster) || { sx: 0, sy: 0, c: 0, list: [] };
        g.sx += n.x; g.sy += n.y; g.c += 1; g.list.push(n);
        groups.set(n.cluster, g);
      }
      const ease = 1 - Math.exp(-dt * 7);
      for (const [id, g] of groups) {
        const cx = g.sx / g.c;
        const cy = g.sy / g.c;
        let r = 0;
        for (const n of g.list) r = Math.max(r, Math.hypot(n.x - cx, n.y - cy) + n.r);
        r += comp ? 10 : 18;
        let hl = s.halos.get(id);
        if (!hl) { hl = { x: cx, y: cy, r }; s.halos.set(id, hl); }
        hl.x += (cx - hl.x) * ease;
        hl.y += (cy - hl.y) * ease;
        hl.r += (r - hl.r) * ease;
      }
      for (const id of [...s.halos.keys()]) if (!groups.has(id)) s.halos.delete(id);

      for (const [id, hl] of s.halos) {
        const m = meta.get(id);
        const over = m && m.load > 1;
        ctx.beginPath();
        ctx.arc(hl.x, hl.y, hl.r, 0, Math.PI * 2);
        ctx.fillStyle = over ? "rgba(245,158,11,0.06)" : "rgba(139,92,246,0.07)";
        ctx.fill();
        ctx.setLineDash([5, 6]);
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = over ? "rgba(245,158,11,0.6)" : "rgba(167,139,250,0.4)";
        ctx.stroke();
        ctx.setLineDash([]);
        if (m) {
          ctx.textAlign = "center";
          ctx.font = `600 ${comp ? 11 : 13}px Inter, system-ui, sans-serif`;
          ctx.fillStyle = "rgba(226,232,240,0.92)";
          ctx.fillText(comp ? m.label : `Cluster ${m.label}`, hl.x, hl.y - hl.r - (comp ? 5 : 20));
          if (!comp) {
            ctx.font = "500 11px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.fillStyle = over ? "rgba(251,191,36,0.95)" : "rgba(148,163,184,0.9)";
            ctx.fillText(`${m.size} containers · load ${Math.round(m.load * 100)}%`, hl.x, hl.y - hl.r - 6);
          }
        }
      }

      // ---- ingress -> cluster traffic links with flowing particles
      const hub = { x: w / 2, y: h / 2 };
      const hubR = comp ? 9 : 16;
      s.dash = (s.dash + dt * 40) % 1000;
      for (const lk of f.links) {
        const hl = s.halos.get(lk.target);
        if (!hl) continue;
        const dx = hl.x - hub.x;
        const dy = hl.y - hub.y;
        const dist = Math.hypot(dx, dy) || 1;
        const ux = dx / dist;
        const uy = dy / dist;
        const p0 = { x: hub.x + ux * (hubR + 4), y: hub.y + uy * (hubR + 4) };
        const p1 = { x: hl.x - ux * (hl.r + 7), y: hl.y - uy * (hl.r + 7) };
        const bend = dist * 0.09;
        const c = { x: (p0.x + p1.x) / 2 - uy * bend, y: (p0.y + p1.y) / 2 + ux * bend };
        const lw = Math.min(comp ? 6 : 10, (comp ? 0.8 : 1.2) + lk.volume * (comp ? 9 : 13));
        const deg = lk.degraded;

        ctx.save();
        ctx.lineWidth = lw;
        ctx.lineCap = "round";
        if (deg) {
          ctx.strokeStyle = BLUE;
          ctx.shadowColor = BLUE;
          ctx.shadowBlur = 14;
          ctx.setLineDash([10, 8]);
          ctx.lineDashOffset = -s.dash;
        } else {
          ctx.strokeStyle = "rgba(148,163,184,0.42)";
        }
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.quadraticCurveTo(c.x, c.y, p1.x, p1.y);
        ctx.stroke();
        ctx.restore();

        // arrowhead
        const tx = p1.x - c.x;
        const ty = p1.y - c.y;
        const tl = Math.hypot(tx, ty) || 1;
        const ax = tx / tl;
        const ay = ty / tl;
        const as = (comp ? 6 : 9) + lw * 0.6;
        ctx.beginPath();
        ctx.moveTo(p1.x + ax * 2, p1.y + ay * 2);
        ctx.lineTo(p1.x - ax * as - ay * as * 0.55, p1.y - ay * as + ax * as * 0.55);
        ctx.lineTo(p1.x - ax * as + ay * as * 0.55, p1.y - ay * as - ax * as * 0.55);
        ctx.closePath();
        ctx.fillStyle = deg ? BLUE : "rgba(148,163,184,0.75)";
        ctx.fill();

        // particles
        let ps = s.particles.get(lk.target);
        if (!ps) { ps = []; s.particles.set(lk.target, ps); }
        const want = Math.max(3, Math.min(comp ? 8 : 20, Math.round(lk.volume * (comp ? 22 : 34))));
        while (ps.length < want) ps.push(Math.random());
        ps.length = want;
        const speed = (deg ? 0.1 : 0.3) * (0.6 + lk.volume);
        for (let i = 0; i < ps.length; i++) {
          ps[i] = (ps[i] + speed * dt) % 1;
          const pt = bezier(p0, c, p1, ps[i]);
          ctx.globalAlpha = Math.sin(Math.PI * ps[i]);
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, comp ? 1.4 : 2.1, 0, Math.PI * 2);
          ctx.fillStyle = deg ? "#bfdbfe" : "rgba(226,232,240,0.95)";
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      // ---- ingress hub
      const ring = hubR + 6 + Math.sin(now / 350) * 2 * f.traffic;
      ctx.beginPath();
      ctx.arc(hub.x, hub.y, ring, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(148,163,184,${0.15 + Math.min(f.traffic, 1.8) * 0.12})`;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(hub.x, hub.y, hubR, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(15,23,42,0.95)";
      ctx.fill();
      ctx.strokeStyle = "rgba(203,213,225,0.85)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      if (!comp) {
        ctx.textAlign = "center";
        ctx.font = "700 10px Inter, system-ui, sans-serif";
        ctx.fillStyle = "#e2e8f0";
        ctx.fillText("LB", hub.x, hub.y + 3.5);
        ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(148,163,184,0.95)";
        ctx.fillText(`ingress ${f.traffic.toFixed(2)}×`, hub.x, hub.y + hubR + 22);
      }

      // ---- container dots (healthy first, failed on top)
      const glow = nodes.length < 160;
      for (const pass of [false, true]) {
        for (const n of nodes) {
          if (n.failed !== pass) continue;
          const p = (now - n.recoveredAt) / RECOVER_MS;
          const recovering = !n.failed && p >= 0 && p < 1;
          const col = n.failed ? RED : recovering ? mix(AMBER, GREEN, p) : GREEN;
          const rr = n.failed ? n.r * (1.08 + 0.1 * Math.sin(now / 180 + n.phase * 6)) : n.r;
          if (glow) {
            ctx.shadowColor = rgba(col, 0.9);
            ctx.shadowBlur = n.failed ? 18 : 8;
          }
          ctx.beginPath();
          ctx.arc(n.x, n.y, rr, 0, Math.PI * 2);
          ctx.fillStyle = rgba(col);
          ctx.fill();
          ctx.shadowBlur = 0;
          if (n.failed) {
            const ph = (now / 900 + n.phase) % 1;
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r + ph * 14, 0, Math.PI * 2);
            ctx.strokeStyle = rgba(RED, (1 - ph) * 0.7);
            ctx.lineWidth = 1.5;
            ctx.stroke();
          } else if (recovering) {
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r + p * 12, 0, Math.PI * 2);
            ctx.strokeStyle = rgba(GREEN, (1 - p) * 0.8);
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
          if (s.selected === n.id) {
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r + 5, 0, Math.PI * 2);
            ctx.strokeStyle = "rgba(255,255,255,0.95)";
            ctx.lineWidth = 1.6;
            ctx.stroke();
          }
        }
      }
    };
    raf = requestAnimationFrame(draw);

    const onClick = (e) => {
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left;
      const y = e.clientY - r.top;
      let best = null;
      let bd = Infinity;
      for (const n of s.nodes.values()) {
        const d = Math.hypot(n.x - x, n.y - y);
        if (d < n.r + 6 && d < bd) { best = n; bd = d; }
      }
      s.onSelect?.(best ? best.id : null);
    };
    canvas.addEventListener("click", onClick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("click", onClick);
      s.sim.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- new frame: sync nodes, clusters, colours
  useEffect(() => {
    if (!frame || !s.sim) return;
    const now = performance.now();
    s.frame = frame;
    layout(s);
    let reheat = false;
    const seen = new Set();
    for (const n of frame.nodes) {
      seen.add(n.id);
      const r = (s.compact ? 3.2 : 4.6) + n.demand * (s.compact ? 1.5 : 2.3);
      let sn = s.nodes.get(n.id);
      if (!sn) {
        const c = s.centers.get(n.cluster) || { x: s.w / 2, y: s.h / 2 };
        sn = {
          id: n.id, x: c.x + (Math.random() - 0.5) * 40, y: c.y + (Math.random() - 0.5) * 40,
          cluster: n.cluster, failed: n.failed, recoveredAt: -1e9, phase: Math.random(),
        };
        s.nodes.set(n.id, sn);
        reheat = true;
      }
      if (sn.cluster !== n.cluster) { sn.cluster = n.cluster; reheat = true; }
      if (sn.failed && !n.failed) sn.recoveredAt = now;
      sn.failed = n.failed;
      sn.r = r;
    }
    for (const id of [...s.nodes.keys()]) if (!seen.has(id)) s.nodes.delete(id);
    s.sim.nodes([...s.nodes.values()]);
    if (reheat) s.sim.alpha(1).restart();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  return (
    <div ref={wrapRef} className="canvas-wrap">
      <canvas ref={canvasRef} />
      {children}
    </div>
  );
}
