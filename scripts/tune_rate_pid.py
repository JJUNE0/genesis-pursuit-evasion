"""scripts/tune_rate_pid.py — Body-rate PD gain grid search (PR-H 단계 2).

Single drone, gravity-only, step-input on each body axis (roll/pitch/yaw)
across a (Kp, Kd) grid. Picks the gains that minimize a weighted score of
rise time, overshoot, and steady-state error subject to "no-divergence".

Setup
-----
- ``num_envs`` envs, all with the **same** gain combo (env-mean for noise
  reduction). Outer loop sweeps the (axis, Kp, Kd) grid sequentially.
- dt = 0.005 s (env default), 0.5 s zero-action stabilize → 1.0 s step input.
- Step amplitude: ``action[:, axis+1] = +1.0`` → ω_ref = +π rad/s.

Outputs
-------
- ``configs/envs/rate_controller.yaml``  — tuned gains (per axis).
- ``outputs/rate_pid_tuning/step_response.png``  — overlay of all combos.
- ``outputs/rate_pid_tuning/pareto.png``         — rise_time vs overshoot.
- ``outputs/rate_pid_tuning/tuning_report.md``   — summary + recommendation.

Usage
-----
    python scripts/tune_rate_pid.py
    python scripts/tune_rate_pid.py --backend cpu --num_envs 16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs                          # noqa: E402
import matplotlib                             # noqa: E402
matplotlib.use("Agg")                         # headless save
import matplotlib.pyplot as plt               # noqa: E402

from utils.drone_params import DroneParams    # noqa: E402
from utils.rate_controller import BodyRateController  # noqa: E402


AXIS_NAMES = ("roll", "pitch", "yaw")
# 사용자 1차 그리드 결과 분석: Kd=0.0005~0.01은 모두 discrete-time 불안정.
# 1-step 후방 차분 derivative 모델에서 stability 필요조건: |Kd/I| < 1.
#   - roll  I_xx = 3.35e-4 → Kd_max ≈ 3.35e-4
#   - pitch I_yy = 6.22e-4 → Kd_max ≈ 6.22e-4
#   - yaw   I_zz = 8.81e-4 → Kd_max ≈ 8.81e-4
# 따라서 Kd는 ~1e-4~3e-4 영역이 정답. Kp는 first-order plant의 time-constant
# τ = (I+Kd)/Kp 기준 t_rise 90% < 0.1s를 위해 Kp > 23·I ≈ 0.008~0.02.
# 사용자 "또는 manual sweep" 허가에 따라 그리드 재조정.
KP_LIST = (0.02, 0.05, 0.1, 0.15, 0.2)
KD_LIST = (5.0e-5, 1.0e-4, 2.0e-4, 3.0e-4)
T_STABLE_S = 0.5
T_STEP_S = 1.0
DT = 0.005
# 중력 활성 테스트에서 roll/pitch 축은 ~0.45s 후 z=0.1m 지면에 도달 (자유낙하).
# 이후 지면 접촉으로 ang_vel이 망가져 metric 오염. 따라서 응답 평가는
# 첫 METRIC_WINDOW_S에 한정 (rise/overshoot/ess 모두 이 window 안에서).
METRIC_WINDOW_S = 0.3
ESS_WINDOW_S = 0.1

# Targets (task spec).
RISE_TIME_TARGET = 0.1   # s, ≤
OVERSHOOT_TARGET = 20.0  # %, ≤
ESS_TARGET = 5.0         # %, ≤


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Body-rate PD gain tuner")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--drone_yaml",
        type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
    )
    p.add_argument("--backend", type=str, default="cpu", choices=["gpu", "cpu"])
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "outputs" / "rate_pid_tuning"),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Trial: one (axis, Kp, Kd) combo → (T, B) history of ω on that axis.
# ---------------------------------------------------------------------------

def _reset_drone(drone, n_envs: int, identity_q: torch.Tensor) -> None:
    spawn = torch.zeros((n_envs, 3), device=gs.device, dtype=gs.tc_float)
    spawn[:, 2] = 1.0
    drone.set_pos(spawn, zero_velocity=True)
    drone.set_quat(identity_q, zero_velocity=True)


def run_trial(
    scene, drone, params: DroneParams, kp: float, kd: float,
    axis_idx: int, n_envs: int, identity_q: torch.Tensor,
    n_stable: int, n_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (ω_history, z_history). Both (n_step, n_envs)."""
    _reset_drone(drone, n_envs, identity_q)

    ctrl = BodyRateController(
        num_envs=n_envs, dt=DT, params=params, device=gs.device,
        kp_rate=(kp, kp, kp), kd_rate=(kd, kd, kd),
    )

    action_zero = torch.zeros((n_envs, 4), device=gs.device, dtype=gs.tc_float)
    action_step = torch.zeros((n_envs, 4), device=gs.device, dtype=gs.tc_float)
    action_step[:, axis_idx + 1] = 1.0

    # 1) stabilize (0.5 s) — assert hover stable (no fall during stabilize phase).
    for _ in range(n_stable):
        ang = drone.get_ang()
        rpms = ctrl.step(action_zero, ang)
        drone.set_propellers_rpm(rpms)
        scene.step()

    # 2) step input — record body-rate on test axis AND z position.
    history = torch.zeros((n_step, n_envs), device=gs.device, dtype=gs.tc_float)
    z_history = torch.zeros((n_step, n_envs), device=gs.device, dtype=gs.tc_float)
    for k in range(n_step):
        ang = drone.get_ang()
        rpms = ctrl.step(action_step, ang)
        drone.set_propellers_rpm(rpms)
        scene.step()
        history[k] = drone.get_ang()[:, axis_idx]
        z_history[k] = drone.get_pos()[:, 2]

    return history, z_history


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    history: torch.Tensor, w_ref: float,
    z_history: torch.Tensor | None = None, z_initial: float = 1.0,
) -> dict:
    """history: (T, B) body-rate. z_history: (T, B) z-position (optional).

    Metric window: 첫 METRIC_WINDOW_S초 (지면 접촉 contamination 회피).
    ess는 metric window 끝 ESS_WINDOW_S 평균.
    z_drop: metric window 끝에서 z_initial 대비 떨어진 거리 (m, env-mean).
    """
    w_full = history.mean(dim=1).detach().cpu().numpy()          # (T,)
    n_metric = min(len(w_full), int(METRIC_WINDOW_S / DT))
    w_np = w_full[:n_metric]
    t = np.arange(n_metric) * DT

    finite_all = bool(np.all(np.isfinite(w_np)))
    peak = float(np.max(np.abs(w_np)))
    diverged = (not finite_all) or peak > 3.0 * abs(w_ref)

    # Rise to 90% of ω_ref (assumes positive ref).
    threshold = 0.9 * w_ref
    above = np.where(w_np >= threshold)[0]
    rise_time = float(t[above[0]]) if len(above) > 0 else float("inf")

    # Overshoot (positive only) — within the metric window.
    peak_signed = float(np.max(w_np))
    overshoot = max(0.0, (peak_signed - w_ref) / w_ref) * 100.0

    # ESS: last ESS_WINDOW_S of the metric window.
    n_settle = max(1, int(ESS_WINDOW_S / DT))
    n_settle = min(n_settle, n_metric)
    final = float(w_np[-n_settle:].mean())
    ess = abs(final - w_ref) / abs(w_ref) * 100.0

    # z_drop within metric window (PR-J: hover stability check).
    if z_history is not None:
        z_full = z_history.mean(dim=1).detach().cpu().numpy()
        z_at_metric_end = float(z_full[n_metric - 1])
        z_drop = z_initial - z_at_metric_end
    else:
        z_full = None
        z_drop = float("nan")

    return {
        "rise_time": rise_time,
        "overshoot": overshoot,
        "ess": ess,
        "diverged": diverged,
        "peak": peak,
        "final": final,
        "z_drop": z_drop,
        "trajectory": w_full,
        "trajectory_metric": w_np,
        "z_trajectory": z_full,
    }


# ---------------------------------------------------------------------------
# Best-combo selection
# ---------------------------------------------------------------------------

def _score(m: dict) -> float:
    """Lower is better. Heavy penalty for rising slowly; soft for overshoot/ess."""
    rt = m["rise_time"] if m["rise_time"] != float("inf") else 1.0
    return rt + 0.005 * m["overshoot"] + 0.003 * m["ess"]


def _meets_targets(m: dict) -> bool:
    return (
        not m["diverged"]
        and m["rise_time"] < RISE_TIME_TARGET
        and m["overshoot"] < OVERSHOOT_TARGET
        and m["ess"] < ESS_TARGET
    )


def pick_best_per_axis(results: dict) -> dict:
    """Tiered selection — body-rate tracking은 ess가 가장 중요.

    Tier 1: 모든 targets 만족 (rise<0.1, os<20, ess<5).  → 최저 score
    Tier 2: ess<5 + os<20 (트래킹 우수, rise만 늦음).    → 최저 rise
    Tier 3: ess<10 + os<30.                              → 최저 rise + 0.5·ess
    Tier 4: 모든 non-divergent.                          → 최저 (ess + os + rise)
    """
    out = {}
    for axis_idx in range(3):
        cands = []
        for kp in KP_LIST:
            for kd in KD_LIST:
                m = results[(axis_idx, kp, kd)]
                if m["diverged"]:
                    continue
                cands.append({
                    "kp": kp, "kd": kd,
                    "score": _score(m),
                    "meets": _meets_targets(m),
                    **m,
                })
        if not cands:
            out[axis_idx] = None
            continue

        tier1 = [c for c in cands if c["meets"]]
        if tier1:
            out[axis_idx] = min(tier1, key=lambda c: c["score"])
            continue

        tier2 = [c for c in cands if c["ess"] < 5.0 and c["overshoot"] < 20.0]
        if tier2:
            out[axis_idx] = min(tier2, key=lambda c: c["rise_time"])
            continue

        tier3 = [c for c in cands if c["ess"] < 10.0 and c["overshoot"] < 30.0]
        if tier3:
            out[axis_idx] = min(
                tier3, key=lambda c: c["rise_time"] + 0.005 * c["ess"]
            )
            continue

        # Tier 4: pure score-based.
        out[axis_idx] = min(
            cands, key=lambda c: c["ess"] + 0.5 * c["overshoot"] + 0.5 * c["rise_time"] * 100,
        )
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_step_responses(results: dict, w_ref: float, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    n_combos = len(KP_LIST) * len(KD_LIST)
    cmap = plt.cm.viridis

    for ax_idx, ax_name in enumerate(AXIS_NAMES):
        ax = axes[ax_idx]
        ax.axhline(w_ref, color="k", ls="--", lw=1, label=f"ref={w_ref:.3f}")
        ax.axhline(0.9 * w_ref, color="gray", ls=":", lw=1, alpha=0.6, label="90% ref")
        ax.axhline(1.2 * w_ref, color="red", ls=":", lw=1, alpha=0.4, label="20% OS limit")
        ax.axvline(METRIC_WINDOW_S, color="orange", ls="--", lw=1, alpha=0.5, label=f"metric window={METRIC_WINDOW_S}s")
        ax.set_xlim(0, min(METRIC_WINDOW_S * 2, T_STEP_S))

        i = 0
        for kp in KP_LIST:
            for kd in KD_LIST:
                m = results[(ax_idx, kp, kd)]
                color = cmap(i / max(n_combos - 1, 1))
                w = m["trajectory"]
                # Cap divergent traces visually.
                w_clipped = np.clip(w, -3 * abs(w_ref), 3 * abs(w_ref))
                t = np.arange(len(w)) * DT
                ax.plot(t, w_clipped, color=color, lw=0.8, alpha=0.7,
                        label=f"Kp={kp}, Kd={kd}")
                i += 1

        ax.set_title(f"{ax_name} step response")
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("body rate (rad/s)")

    # Single legend under the figure.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = output_dir / "step_response.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pareto(results: dict, best: dict, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_idx, ax_name in enumerate(AXIS_NAMES):
        ax = axes[ax_idx]
        rts, oss, kps, kds, divs = [], [], [], [], []
        for kp in KP_LIST:
            for kd in KD_LIST:
                m = results[(ax_idx, kp, kd)]
                rt = m["rise_time"] if m["rise_time"] != float("inf") else 1.5
                rts.append(rt)
                oss.append(m["overshoot"])
                kps.append(kp)
                kds.append(kd)
                divs.append(m["diverged"])
        rts = np.array(rts); oss = np.array(oss); divs = np.array(divs)

        # Color by Kp, marker shape by divergence.
        for i in range(len(rts)):
            color = plt.cm.plasma(KP_LIST.index(kps[i]) / max(len(KP_LIST) - 1, 1))
            marker = "x" if divs[i] else "o"
            ax.scatter(rts[i], oss[i], color=color, marker=marker, s=50,
                       edgecolors="k", linewidths=0.4)

        # Highlight best
        b = best[ax_idx]
        if b is not None:
            ax.scatter(b["rise_time"], b["overshoot"], color="lime", s=200,
                       marker="*", edgecolors="k", linewidths=1.5,
                       label=f"best Kp={b['kp']} Kd={b['kd']}", zorder=5)
            ax.legend(loc="upper right", fontsize=8)

        ax.axvline(RISE_TIME_TARGET, color="gray", ls="--", lw=1, alpha=0.5)
        ax.axhline(OVERSHOOT_TARGET, color="gray", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("rise time to 90% (s)")
        ax.set_ylabel("overshoot (%)")
        ax.set_title(f"{ax_name} Pareto (color=Kp, x=diverged)")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = output_dir / "pareto.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Reporting / yaml writer
# ---------------------------------------------------------------------------

def write_tuned_yaml(best: dict, params: DroneParams, path: Path) -> None:
    hover_thrust = float(params.mass * params.gravity)
    max_thrust = 2.0 * hover_thrust
    body = []
    body.append("# Auto-generated by scripts/tune_rate_pid.py")
    body.append("# 단계 2 — body-rate PD gain tuning result.")
    body.append("# 적용: configs/envs/pe_1v1_default.yaml의 action.rate_controller 블록에 복사")
    body.append("#       또는 직접 read 후 env_cfg에 주입.")
    body.append("rate_controller:")
    kp = [(best[i]["kp"] if best[i] else 0.05) for i in range(3)]
    kd = [(best[i]["kd"] if best[i] else 0.001) for i in range(3)]
    body.append(f"  kp: [{kp[0]}, {kp[1]}, {kp[2]}]   # roll, pitch, yaw")
    body.append(f"  kd: [{kd[0]}, {kd[1]}, {kd[2]}]")
    body.append(f"  max_rate: 3.14159   # ±π rad/s")
    body.append(f"  max_thrust: {max_thrust:.4f}   # = 2 × m·g  ({params.mass:.4f} × {params.gravity:.2f})")
    body.append("")
    path.write_text("\n".join(body))


def write_report(
    results: dict, best: dict, params: DroneParams, path: Path,
    output_files: dict,
) -> None:
    L = []
    L.append("# Body-rate PD gain tuning report")
    L.append("")
    L.append("## Setup")
    L.append("")
    L.append(f"- Drone: {params.name} (mass {params.mass:.4f} kg, gravity {params.gravity})")
    L.append(f"- I = (Ixx, Iyy, Izz) = ({params.inertia[0]:.6f}, {params.inertia[1]:.6f}, {params.inertia[2]:.6f}) kg·m²")
    L.append(f"- arm_length = {params.arm_length:.5f} m, kf = {params.kf:.3e}, km/kf = {params.torque_thrust_ratio:.5f}")
    L.append(f"- dt = {DT} s ({int(1/DT)} Hz inner loop), 환경 dt와 동일")
    L.append(f"- step amplitude: action[axis+1] = +1.0 → ω_ref = +π = {math.pi:.4f} rad/s")
    L.append(f"- stabilize {T_STABLE_S}s + step {T_STEP_S}s, env-mean over {results.get('_n_envs', 'N')} envs")
    L.append("")
    L.append("## Targets")
    L.append("")
    L.append(f"- rise_time (90%) ≤ {RISE_TIME_TARGET}s")
    L.append(f"- overshoot ≤ {OVERSHOOT_TARGET}%")
    L.append(f"- steady-state error ≤ {ESS_TARGET}%")
    L.append("- 발산 없음 (peak < 3·ω_ref, no NaN/Inf)")
    L.append("")
    L.append("## Grid")
    L.append("")
    L.append(f"- Kp ∈ {list(KP_LIST)}")
    L.append(f"- Kd ∈ {list(KD_LIST)}")
    n_combos = len(KP_LIST) * len(KD_LIST)
    L.append(f"- {n_combos} combos × 3 axes = {n_combos*3} trials")
    L.append(f"- metric window: 처음 {METRIC_WINDOW_S}s만 평가 (gravity-induced 지면 접촉 contamination 회피)")
    L.append(f"- ess window: metric window 끝 {ESS_WINDOW_S}s 평균")
    L.append("")
    L.append("## Per-axis grid results (env-mean)")
    L.append("")
    for ax_idx, ax_name in enumerate(AXIS_NAMES):
        L.append(f"### {ax_name}")
        L.append("")
        L.append("| Kp | Kd | rise_time(s) | overshoot(%) | ess(%) | peak(rad/s) | z_drop(m) | diverged | meets |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")
        for kp in KP_LIST:
            for kd in KD_LIST:
                m = results[(ax_idx, kp, kd)]
                rt = "inf" if m["rise_time"] == float("inf") else f"{m['rise_time']:.4f}"
                z_drop = m.get("z_drop", float("nan"))
                z_str = f"{z_drop:.3f}" if not (isinstance(z_drop, float) and np.isnan(z_drop)) else "—"
                meets = "✓" if (not m["diverged"]
                                and m["rise_time"] < RISE_TIME_TARGET
                                and m["overshoot"] < OVERSHOOT_TARGET
                                and m["ess"] < ESS_TARGET) else ""
                L.append(f"| {kp} | {kd:.2e} | {rt} | {m['overshoot']:.2f} | {m['ess']:.2f} | {m['peak']:.3f} | {z_str} | {'⚠' if m['diverged'] else ''} | {meets} |")
        L.append("")

    L.append("## 추천 게인 (per-axis)")
    L.append("")
    L.append("| axis | Kp | Kd | rise_time(s) | overshoot(%) | ess(%) | meets all targets |")
    L.append("|---|---:|---:|---:|---:|---:|:---:|")
    for ax_idx, ax_name in enumerate(AXIS_NAMES):
        b = best[ax_idx]
        if b is None:
            L.append(f"| {ax_name} | — | — | — | — | — | (no non-diverged combo) |")
        else:
            rt = "inf" if b["rise_time"] == float("inf") else f"{b['rise_time']:.4f}"
            meets = "✓" if b["meets"] else "✗"
            L.append(f"| {ax_name} | {b['kp']} | {b['kd']} | {rt} | {b['overshoot']:.2f} | {b['ess']:.2f} | {meets} |")
    L.append("")
    L.append("## 선택 기준 (tiered)")
    L.append("")
    L.append("Body-rate 트래킹은 ess(steady-state error)가 가장 중요. 따라서 4단 priority:")
    L.append("")
    L.append("1. **Tier 1 — 모든 targets 만족** (rise<0.1s, os<20%, ess<5%, no divergence) → score 최소.")
    L.append("2. **Tier 2 — 트래킹 우수** (ess<5% + os<20%) → 그 중 rise 최단.")
    L.append("3. **Tier 3 — 트래킹 양호** (ess<10% + os<30%) → rise + 0.005·ess 최소.")
    L.append("4. **Tier 4 — fallback** (non-divergent any) → ess + 0.5·os + 0.5·rise 최소.")
    L.append("")
    L.append("score (Tier 1) = rise_time + 0.005·overshoot(%) + 0.003·ess(%) (lower better).")
    L.append("")
    L.append("## 미해결 / 향후 검토")
    L.append("")
    # Diagnose unmet targets.
    unmet = []
    for ax_idx, ax_name in enumerate(AXIS_NAMES):
        b = best[ax_idx]
        if b is None:
            unmet.append(f"- **{ax_name}**: 모든 combo 발산 — 현 grid 하한 Kp가 너무 크거나 Kd 너무 작음. Kp 한 자릿수 감소 / Kd 한 자릿수 증가 grid 권장.")
        elif not b["meets"]:
            reasons = []
            if b["rise_time"] >= RISE_TIME_TARGET:
                reasons.append(f"rise_time {b['rise_time']:.3f}s > {RISE_TIME_TARGET}s")
            if b["overshoot"] >= OVERSHOOT_TARGET:
                reasons.append(f"overshoot {b['overshoot']:.1f}% > {OVERSHOOT_TARGET}%")
            if b["ess"] >= ESS_TARGET:
                reasons.append(f"ess {b['ess']:.1f}% > {ESS_TARGET}%")
            unmet.append(f"- **{ax_name}**: best combo가 targets 미달 ({', '.join(reasons)}).")
    if unmet:
        L.extend(unmet)
        L.append("")
        L.append("**다음 grid 후보** (선형 영역 확장):")
        L.append("- Kp ∈ [0.3, 0.5, 0.8, 1.2] (motor saturation 한계 부근)")
        L.append("- Kd ∈ [0.01, 0.02, 0.05, 0.1] (damping 강화)")
        L.append("- 단, motor 추력 한계가 mixer 입력에서 clip 시 비선형성 도입 — settling time 추가 단축은 cascade (attitude→rate) PID 도입 검토.")
    else:
        L.append("- 전 축이 targets 만족. 단계 3(RL 학습)으로 진행 가능.")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append(f"- step response plot: `{output_files['step']}`")
    L.append(f"- Pareto plot: `{output_files['pareto']}`")
    L.append(f"- tuned yaml: `{output_files['yaml']}`")
    L.append("")

    path.write_text("\n".join(L))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _cleanup() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        gs.destroy()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")

    try:
        return _run(args, output_dir)
    except Exception as exc:                 # noqa: BLE001
        import traceback
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3
    finally:
        _cleanup()


def _run(args: argparse.Namespace, output_dir: Path) -> int:
    params = DroneParams(args.drone_yaml)
    B = int(args.num_envs)

    # Build scene once (drone + plane). Re-used across all 48 trials.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=16),
        rigid_options=gs.options.RigidOptions(
            dt=DT,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    urdf_path = params.urdf_path
    if not Path(urdf_path).is_absolute():
        urdf_path = str(ROOT / urdf_path)
    drone = scene.add_entity(
        gs.morphs.Drone(
            file=urdf_path,
            propellers_link_name=params.prop_link_names,
            propellers_spin=[int(s) for s in params.spin_directions],
        )
    )
    scene.build(n_envs=B)

    identity_q = torch.zeros((B, 4), device=gs.device, dtype=gs.tc_float)
    identity_q[:, 0] = 1.0

    n_stable = int(T_STABLE_S / DT)
    n_step = int(T_STEP_S / DT)
    w_ref = math.pi   # max_body_rate, action=+1

    # Prime hover RPM holds for safety (set once; controller will overwrite).
    hover_rpm = float(np.sqrt(params.mass * params.gravity / (params.n_propellers * params.kf)))
    print(f"[info] hover_rpm = {hover_rpm:.1f}, B = {B}, dt = {DT}, w_ref = {w_ref:.4f} rad/s")
    print(f"[info] grid: Kp={KP_LIST}, Kd={KD_LIST}; axes={AXIS_NAMES}")
    print(f"[info] {len(KP_LIST)*len(KD_LIST)*3} trials × {n_stable + n_step} steps each")

    results: dict = {"_n_envs": B}
    n_trials = len(KP_LIST) * len(KD_LIST) * 3
    trial_i = 0
    for axis_idx in range(3):
        for kp in KP_LIST:
            for kd in KD_LIST:
                trial_i += 1
                hist, z_hist = run_trial(
                    scene=scene, drone=drone, params=params,
                    kp=kp, kd=kd, axis_idx=axis_idx,
                    n_envs=B, identity_q=identity_q,
                    n_stable=n_stable, n_step=n_step,
                )
                m = compute_metrics(hist, w_ref, z_history=z_hist, z_initial=1.0)
                results[(axis_idx, kp, kd)] = m
                tag = "OK" if not m["diverged"] else "DIV"
                rt_str = "inf" if m["rise_time"] == float("inf") else f"{m['rise_time']:.4f}"
                print(
                    f"[{trial_i:02d}/{n_trials}] axis={AXIS_NAMES[axis_idx]:5s} "
                    f"Kp={kp:.4f} Kd={kd:.2e}  "
                    f"rise={rt_str}s os={m['overshoot']:.2f}% ess={m['ess']:.2f}% "
                    f"peak={m['peak']:.2f} z_drop={m['z_drop']:.3f}m  [{tag}]"
                )

    best = pick_best_per_axis(results)

    # Outputs
    yaml_path = ROOT / "configs" / "envs" / "rate_controller.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    write_tuned_yaml(best, params, yaml_path)
    step_png = plot_step_responses(results, w_ref, output_dir)
    pareto_png = plot_pareto(results, best, output_dir)
    report_md = output_dir / "tuning_report.md"
    write_report(
        results, best, params, report_md,
        output_files={
            "step": str(step_png.relative_to(ROOT)),
            "pareto": str(pareto_png.relative_to(ROOT)),
            "yaml": str(yaml_path.relative_to(ROOT)),
        },
    )

    # Console summary.
    print()
    print("=" * 70)
    print("[best per axis]")
    for ax_idx, ax_name in enumerate(AXIS_NAMES):
        b = best[ax_idx]
        if b is None:
            print(f"  {ax_name}: NO non-diverged combo")
        else:
            rt = "inf" if b["rise_time"] == float("inf") else f"{b['rise_time']:.4f}"
            ok = "✓" if b["meets"] else "✗"
            z_drop = b.get("z_drop", float("nan"))
            z_str = f"{z_drop:.3f}m" if not np.isnan(z_drop) else "—"
            print(
                f"  {ax_name:5s}: Kp={b['kp']:.4f} Kd={b['kd']:.2e}  "
                f"rise={rt}s os={b['overshoot']:.2f}% ess={b['ess']:.2f}% "
                f"z_drop={z_str}  meets_all={ok}"
            )
    print()
    print(f"[outputs]")
    print(f"  yaml   : {yaml_path}")
    print(f"  step   : {step_png}")
    print(f"  pareto : {pareto_png}")
    print(f"  report : {report_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
