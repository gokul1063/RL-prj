"""Interactive Streamlit dashboard for the container-placement environment.

Run with:
    bash scripts/run_ui.sh          # or
    PYTHONPATH=src .venv/bin/streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_rl.baselines import (  # noqa: E402
    BestFitPolicy,
    FirstFitPolicy,
    LeastRequestedPolicy,
    RandomPolicy,
    RiskAwarePolicy,
    RoundRobinPolicy,
    WorstFitPolicy,
)
from cluster_rl.env import EnvConfig, PlacementEnv  # noqa: E402
from cluster_rl.viz.state import heat_color, service_color, snapshot  # noqa: E402

st.set_page_config(page_title="Cluster Placement RL", layout="wide")

MANUAL = "Manual (click to place)"

POLICY_FACTORIES = {
    "Random": lambda s: RandomPolicy(seed=s),
    "Round-robin": lambda s: RoundRobinPolicy(),
    "First-fit": lambda s: FirstFitPolicy(),
    "Best-fit": lambda s: BestFitPolicy(),
    "Worst-fit": lambda s: WorstFitPolicy(),
    "Least-requested": lambda s: LeastRequestedPolicy(),
    "Risk-aware": lambda s: RiskAwarePolicy(),
}

# Optional: attach a trained PPO policy for a live comparison.
_RL_MODEL_PATH = Path(
    __import__("os").environ.get("RL_MODEL", "runs/ppo_big/ppo_final.zip")
)
if _RL_MODEL_PATH.exists():
    try:
        from sb3_contrib import MaskablePPO

        from cluster_rl.baselines.sb3_policy import SB3Policy

        _RL_MODEL = MaskablePPO.load(str(_RL_MODEL_PATH))
        POLICY_FACTORIES["RL (PPO)"] = lambda s: SB3Policy(_RL_MODEL, name="rl_ppo")
        st.caption(f"Loaded RL model {_RL_MODEL_PATH}")
    except Exception as _e:  # pragma: no cover - optional model
        st.sidebar.warning(f"Could not load RL model: {_e}")


# --------------------------------------------------------------------------- #
# session state
# --------------------------------------------------------------------------- #
def _default_config() -> dict:
    cfg = EnvConfig.from_yaml("configs/smoke.yaml")
    return cfg.to_dict()


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("config", _default_config())
    ss.setdefault("policy_name", "Least-requested")
    ss.setdefault("history", [])
    ss.setdefault("log", [])
    ss.setdefault("autoplay", False)
    ss.setdefault("speed", 0.25)
    ss.setdefault("seed", 0)
    ss.setdefault("failed_seen", 0)
    if "env" not in ss:
        reset_env()


def make_policy(name: str):
    if name == MANUAL:
        return None
    return POLICY_FACTORIES[name](st.session_state.seed)


def reset_env() -> None:
    ss = st.session_state
    cfg = EnvConfig.from_dict(ss.config)
    env = PlacementEnv(cfg)
    env.reset(seed=ss.seed)
    ss.env = env
    ss.history = []
    ss.log = []
    ss.failed_seen = 0
    ss.policy = make_policy(ss.policy_name)
    record(env, "reset", 0.0)


def record(env, action_desc: str, reward: float, image: str = "") -> None:
    ss = st.session_state
    snap = snapshot(env)
    k = snap["kpis"]
    ss.history.append(
        {
            "step": k["steps"],
            "tick": k["tick"],
            "scheduled": k["scheduled"],
            "failed": k["failed"],
            "unscheduled": k["unscheduled"],
            "pending": k["pending"],
            "running": k["running"],
            "cpu_util": k["cpu_util_mean"],
            "mem_util": k["mem_util_mean"],
            "imbalance": k["imbalance"],
            "failure_rate": k["failure_rate"],
            "sla_violation_rate": k["sla_violation_rate"],
            "wait_p95": k["wait_p95"],
            "traffic_imbalance": k["traffic_imbalance"],
            "reward": reward,
        }
    )
    if env.metrics.failed > ss.failed_seen:
        ss.failed_seen = env.metrics.failed
        ss.log.append(f"[tick {k['tick']:>3}] !! a container failed on the cluster")
    if action_desc:
        tag = f" ({image})" if image else ""
        ss.log.append(f"[tick {k['tick']:>3}] {action_desc}{tag}  reward {reward:+.2f}")
    ss.log = ss.log[-200:]


# --------------------------------------------------------------------------- #
# simulation driving
# --------------------------------------------------------------------------- #
def step_once(action: int) -> None:
    env = st.session_state.env
    if env._done:
        return
    pending_inst = env.episode.instances[env.pending[0]] if env.pending else None
    image = pending_inst.docker_image if pending_inst else ""
    if action == env.HOLD:
        desc = "deferred"
    else:
        desc = f"placed on node-{action:02d}"
    obs, reward, terminated, truncated, _ = env.step(action)
    record(env, desc, float(reward), image)


def autoplay_frame() -> None:
    env = st.session_state.env
    policy = st.session_state.policy
    steps_per_frame = 3
    for _ in range(steps_per_frame):
        if env._done:
            break
        if not env.pending:
            step_once(env.HOLD)
            continue
        mask = env.action_masks()
        action = policy.act(env, env._obs(), mask)
        step_once(action)
    time.sleep(max(0.0, st.session_state.speed))
    st.rerun()


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def show_chart(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def render_kpis(k: dict) -> None:
    c = st.columns(9)
    c[0].metric("Tick", f"{k['tick']}/{k['episode_ticks']}")
    c[1].metric("Queue", k["pending"])
    c[2].metric("Running", k["running"])
    c[3].metric("Scheduled", k["scheduled"])
    c[4].metric("Failed", k["failed"], delta=f"{k['failure_rate']*100:.1f}%")
    c[5].metric("Dropped", k["unscheduled"], delta=f"{k['sla_violation_rate']*100:.1f}%")
    c[6].metric("CPU util", f"{k['cpu_util_mean']*100:.1f}%")
    c[7].metric("Imbalance", f"{k['imbalance']:.3f}")
    traffic = k["traffic_imbalance"]
    c[8].metric("Traffic imb.", "n/a" if traffic != traffic else f"{traffic:.3f}")


def render_pending(snap: dict) -> None:
    st.subheader("Incoming container")
    p = snap["pending"]
    if p is None:
        st.info("No container waiting. Press **Step** / **Auto-play** to advance time.")
        return
    col = st.columns([2, 1])
    with col[0]:
        st.markdown(
            f"<div style='padding:12px;border-radius:10px;border:1px solid #444;"
            f"background:{service_color(p['service'])}22'>"
            f"<span style='font-size:22px'>🐳</span> "
            f"<b>{p['docker_image']}</b> "
            f"<span style='opacity:.8'>· service <b>{p['service']}</b></span><br/>"
            f"cpu <b>{p['cpu']:.3f}</b> · mem <b>{p['mem']:.3f}</b> · "
            f"priority <b>{p['priority']}</b> · class <b>{p['sched_class']}</b> · "
            f"duration <b>{p['duration_ticks']} ticks</b> · "
            f"waited <b>{p['wait_ticks']} ticks</b> · "
            f"base risk <b>{p['base_risk']*100:.1f}%</b></div>",
            unsafe_allow_html=True,
        )
    with col[1]:
        st.caption("Choose a node (disabled = not enough CPU/mem):")
        nodes = snap["nodes"]
        grid = st.columns(5)
        key_suffix = f"{st.session_state.env._steps}_{st.session_state.env.current_tick}"
        for i, node in enumerate(nodes):
            with grid[i % 5]:
                if st.button(
                    f"node-{node['index']:02d}",
                    key=f"place_{i}_{key_suffix}",
                    disabled=not node["feasible"] or st.session_state.env._done,
                    help=f"cpu {node['cpu_util']*100:.0f}% · mem {node['mem_util']*100:.0f}%",
                ):
                    step_once(node["index"])
                    st.rerun()
        if st.button("⏸ HOLD", key=f"hold_{key_suffix}", disabled=st.session_state.env._done):
            step_once(st.session_state.env.HOLD)
            st.rerun()


def render_nodes(snap: dict) -> None:
    st.subheader("Cluster")
    nodes = snap["nodes"]
    cols = st.columns(4)
    for i, node in enumerate(nodes):
        with cols[i % 4]:
            cpu_c = heat_color(node["cpu_util"])
            mem_c = heat_color(node["mem_util"])
            border = "#E45756" if node["recent_fail"] > 0.15 else "#333"
            chips = ""
            for ct in node["containers"]:
                bg = "#E45756" if ct["failed"] else service_color(ct["service"])
                short = ct["docker_image"].split(":")[0][:12]
                chips += (
                    f"<span title='{ct['docker_image']}' style='display:inline-block;"
                    f"margin:2px;padding:1px 6px;border-radius:6px;background:{bg};"
                    f"color:white;font-size:11px'>{short}</span>"
                )
            st.markdown(
                f"<div style='border:2px solid {border};border-radius:10px;padding:8px;"
                f"margin-bottom:8px;background:#1a1c23'>"
                f"<b>node-{node['index']:02d}</b> "
                f"<span style='opacity:.6;font-size:11px'>m{node['machine_id']}</span> "
                f"<span style='float:right'>{node['running_count']} ctr</span><br/>"
                f"<small>CPU {node['cpu_used']:.2f}/{node['cpu_cap']:.2f}</small>"
                f"<div style='height:8px;background:#333;border-radius:4px'>"
                f"<div style='height:8px;width:{min(node['cpu_util'],1)*100:.0f}%;"
                f"background:{cpu_c};border-radius:4px'></div></div>"
                f"<small>MEM {node['mem_used']:.2f}/{node['mem_cap']:.2f}</small>"
                f"<div style='height:8px;background:#333;border-radius:4px'>"
                f"<div style='height:8px;width:{min(node['mem_util'],1)*100:.0f}%;"
                f"background:{mem_c};border-radius:4px'></div></div>"
                f"<div style='margin-top:6px'>{chips}</div></div>",
                unsafe_allow_html=True,
            )


def render_charts(history: list[dict]) -> None:
    if len(history) < 2:
        return
    df = pd.DataFrame(history)
    st.subheader("Metrics over time")
    left, right = st.columns(2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["step"], y=df["cpu_util"], name="CPU util"))
    fig.add_trace(go.Scatter(x=df["step"], y=df["mem_util"], name="MEM util"))
    fig.add_trace(go.Scatter(x=df["step"], y=df["imbalance"], name="imbalance"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Load")
    with left:
        show_chart(fig)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["step"], y=df["failure_rate"], name="failure rate"))
    fig2.add_trace(go.Scatter(x=df["step"], y=df["sla_violation_rate"], name="SLA violations"))
    fig2.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Reliability")
    with right:
        show_chart(fig2)

    left2, right2 = st.columns(2)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["step"], y=df["traffic_imbalance"], name="traffic imbalance"))
    fig3.add_trace(go.Scatter(x=df["step"], y=df["wait_p95"], name="wait p95 (ticks)"))
    fig3.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Traffic / latency")
    with left2:
        show_chart(fig3)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df["step"], y=df["reward"], name="reward"))
    fig4.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Reward")
    with right2:
        show_chart(fig4)


def render_reward(snap: dict) -> None:
    rb = snap["reward_breakdown"]
    if not rb:
        return
    st.subheader("Last reward breakdown")
    items = {k[2:]: v for k, v in rb.items() if k.startswith("r_") and k != "r_total"}
    fig = go.Figure(
        go.Bar(x=list(items.keys()), y=list(items.values()),
               marker_color=["#54A24B" if v >= 0 else "#E45756" for v in items.values()])
    )
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
    show_chart(fig)


def render_log(log: list[str]) -> None:
    st.subheader("Event log")
    st.code("\n".join(log[-25:]) or "(empty)", language="text")


def render_download(history: list[dict]) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    st.download_button(
        "⬇ Download metrics CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cluster_env_metrics.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# sidebar + main
# --------------------------------------------------------------------------- #
init_state()

with st.sidebar:
    st.title("🐳 Cluster Placement RL")
    st.caption("Real Google Borg 2011 trace · Gymnasium env")

    st.header("Environment")
    cfg = st.session_state.config
    cfg["processed_dir"] = st.text_input("Processed data dir", cfg["processed_dir"])
    cfg["max_nodes"] = st.slider("Nodes", 4, 24, int(cfg["max_nodes"]))
    cfg["episode_ticks"] = st.slider("Episode ticks", 12, 96, int(cfg["episode_ticks"]))
    cfg["max_arrivals"] = st.slider("Containers", 100, 1500, int(cfg["max_arrivals"]), step=50)
    st.session_state.seed = st.number_input("Seed", value=int(st.session_state.seed), step=1)

    st.header("Policy")
    names = [MANUAL] + list(POLICY_FACTORIES)
    st.session_state.policy_name = st.selectbox(
        "Placement policy", names, index=names.index(st.session_state.policy_name)
    )
    st.session_state.autoplay = st.toggle("Auto-play", value=st.session_state.autoplay)
    st.session_state.speed = st.slider("Frame delay (s)", 0.0, 1.0, float(st.session_state.speed), 0.05)

    st.divider()
    if st.button("▶ Step", width="stretch"):
        env = st.session_state.env
        if env._done:
            st.warning("Episode finished — Reset.")
        else:
            policy = st.session_state.policy
            if policy is None:
                st.info("Manual mode: place using the node buttons.")
            else:
                mask = env.action_masks()
                step_once(policy.act(env, env._obs(), mask))
                st.rerun()
    if st.button("🔄 Reset", width="stretch"):
        reset_env()
        st.rerun()

    st.divider()
    st.caption("Presenter scenarios")
    if st.button("Scenario: bad (Random)", width="stretch"):
        st.session_state.policy_name = "Random"
        st.session_state.autoplay = True
        reset_env()
        st.rerun()
    if st.button("Scenario: good (Risk-aware)", width="stretch"):
        st.session_state.policy_name = "Risk-aware"
        st.session_state.autoplay = True
        reset_env()
        st.rerun()
    if "RL (PPO)" in POLICY_FACTORIES and st.button("Scenario: RL (PPO)", width="stretch"):
        st.session_state.policy_name = "RL (PPO)"
        st.session_state.autoplay = True
        reset_env()
        st.rerun()
    if st.button("Scenario: manual", width="stretch"):
        st.session_state.policy_name = MANUAL
        st.session_state.autoplay = False
        reset_env()
        st.rerun()

env = st.session_state.env

# advance one animation frame before drawing
if st.session_state.autoplay and not env._done and st.session_state.policy is not None:
    autoplay_frame()

snap = snapshot(env)
k = snap["kpis"]

st.markdown(f"### Live cluster view — policy: **{st.session_state.policy_name}**")
if k["done"]:
    st.success("Episode finished. Summary below. Press Reset to run again.")
render_kpis(k)
render_pending(snap)
render_nodes(snap)
render_reward(snap)
render_charts(st.session_state.history)
render_download(st.session_state.history)
render_log(st.session_state.log)

if k["done"]:
    st.subheader("Episode summary")
    st.json(env.episode_metrics())
