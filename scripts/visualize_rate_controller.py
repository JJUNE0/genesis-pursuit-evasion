"""scripts/visualize_rate_controller.py — body-rate PD 응답 시각화.

두 mode 지원:

    --mode step (default)
        축별 step response (rise/settling). Yaml의 게인을 즉시 검증.

    --mode trajectory
        configs/envs/trajectory.yaml에 정의된 [T, wx, wy, wz] sequence를
        controller에 입력하고 결과를 plot. coning / figure8 / yaw_spin /
        random_smooth 등 multi-axis coordinated motion 검증용.

Usage
-----
    # step mode (default — 축별 step response)
    python scripts/visualize_rate_controller.py

    # trajectory mode — coning, figure8 등 (yaml에서 type 변경)
    python scripts/visualize_rate_controller.py --mode trajectory
    python scripts/visualize_rate_controller.py --mode trajectory \\
        --trajectory_yaml configs/envs/trajectory.yaml

    # 게인 다른 yaml에서 로드
    python scripts/visualize_rate_controller.py --config configs/envs/rate_controller.yaml

    # 게인 CLI 직접 지정
    python scripts/visualize_rate_controller.py --kp 0.05 0.10 0.10 --kd 1e-4 5e-5 1e-4

    # Genesis 3D 실시간 viewer
    python scripts/visualize_rate_controller.py --show_viewer
    python scripts/visualize_rate_controller.py --mode trajectory --show_viewer
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs                        # noqa: E402
import matplotlib                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402

from utils.drone_params import DroneParams  # noqa: E402
from utils.rate_controller import BodyRateController  # noqa: E402


AXIS_NAMES = ("roll", "pitch", "yaw")
DT = 0.005
T_STABLE_S = 0.5
T_STEP_S = 1.0


# ---------------------------------------------------------------------------
# CLI / cfg loading
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize body-rate PD response")
    p.add_argument("--mode", type=str, default="step",
                   choices=["step", "trajectory"],
                   help="step: axis별 step response. trajectory: yaml-defined sequence.")
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "envs" / "pe_1v1_default.yaml"),
        help="yaml with action.rate_controller block (default: pe_1v1_default.yaml)",
    )
    p.add_argument(
        "--trajectory_yaml",
        type=str,
        default=str(ROOT / "configs" / "envs" / "trajectory.yaml"),
        help="trajectory definition (--mode trajectory에서 사용)",
    )
    p.add_argument(
        "--drone_yaml", type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
    )
    p.add_argument("--kp", type=float, nargs=3, default=None,
                   help="override Kp [roll pitch yaw] (skips yaml read)")
    p.add_argument("--kd", type=float, nargs=3, default=None,
                   help="override Kd [roll pitch yaw]")
    p.add_argument("--max_body_rate", type=float, default=math.pi)
    p.add_argument("--num_envs", type=int, default=1,
                   help="single drone for cleanest curve; >1 averages env-mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", type=str, default="cpu", choices=["gpu", "cpu"])
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="output png path (default: outputs/rate_pid_tuning/{mode}_response.png)",
    )
    p.add_argument("--show_viewer", action="store_true",
                   help="open Genesis viewer (live 3D)")
    return p.parse_args()


def load_gains_from_yaml(yaml_path: Path) -> tuple[list[float], list[float], float]:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    # Two schemas supported:
    #   (A) pe_1v1_*.yaml: action.rate_controller.{kp_rate,kd_rate,max_body_rate}
    #   (B) rate_controller.yaml standalone: rate_controller.{kp,kd,max_rate}
    rc = (raw.get("action") or {}).get("rate_controller") or raw.get("rate_controller") or {}
    if not rc:
        raise KeyError(f"no rate_controller block found in {yaml_path}")
    kp = rc.get("kp_rate") or rc.get("kp")
    kd = rc.get("kd_rate") or rc.get("kd")
    max_rate = float(rc.get("max_body_rate") or rc.get("max_rate") or math.pi)
    if kp is None or kd is None:
        raise KeyError(f"kp/kd missing in rate_controller block of {yaml_path}")
    return list(map(float, kp)), list(map(float, kd)), max_rate


# ---------------------------------------------------------------------------
# Sim trial
# ---------------------------------------------------------------------------

def run_axis_trial(
    scene, drone, params: DroneParams,
    kp_vec: tuple[float, float, float],
    kd_vec: tuple[float, float, float],
    max_body_rate: float,
    axis_idx: int, n_envs: int, identity_q: torch.Tensor,
    n_stable: int, n_step: int, debug_stabilize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (ω_hist, z_hist), each shape (n_step, n_envs).

    debug_stabilize=True → 첫 10 stabilize step에서 z, ang_vel, rpm 출력 (PR-J 진단).
    """
    spawn = torch.zeros((n_envs, 3), device=gs.device, dtype=gs.tc_float)
    spawn[:, 2] = 1.0
    drone.set_pos(spawn, zero_velocity=True)
    drone.set_quat(identity_q, zero_velocity=True)

    ctrl = BodyRateController(
        num_envs=n_envs, dt=DT, params=params, device=gs.device,
        kp_rate=tuple(kp_vec), kd_rate=tuple(kd_vec), max_body_rate=max_body_rate,
    )

    a_zero = torch.zeros((n_envs, 4), device=gs.device, dtype=gs.tc_float)
    a_step = a_zero.clone()
    a_step[:, axis_idx + 1] = 1.0

    for k in range(n_stable):
        ang = drone.get_ang()
        rpms = ctrl.step(a_zero, ang)
        if debug_stabilize and k < 10:
            z = drone.get_pos()[0, 2].item()
            ang_np = ang[0].cpu().numpy()
            rpm0 = rpms[0, 0].item()
            print(f"[stabilize k={k:2d}] z={z:.4f} ang_vel={ang_np} rpm0={rpm0:.1f}")
        drone.set_propellers_rpm(rpms)
        scene.step()

    history = torch.zeros((n_step, n_envs), device=gs.device, dtype=gs.tc_float)
    z_history = torch.zeros((n_step, n_envs), device=gs.device, dtype=gs.tc_float)
    for k in range(n_step):
        ang = drone.get_ang()
        rpms = ctrl.step(a_step, ang)
        drone.set_propellers_rpm(rpms)
        scene.step()
        history[k] = drone.get_ang()[:, axis_idx]
        z_history[k] = drone.get_pos()[:, 2]
    return history, z_history


# ---------------------------------------------------------------------------
# Continuous-time CL prediction (1st-order: τ = (I+Kd)/Kp)
# ---------------------------------------------------------------------------

def predict_continuous(
    kp: float, kd: float, inertia: float,
    w_ref: float, t: np.ndarray,
) -> np.ndarray:
    """1st-order CL: ω(t) = ω_ref · (1 - exp(-t/τ)), τ = (I+Kd)/Kp."""
    tau = (inertia + kd) / kp
    return w_ref * (1.0 - np.exp(-t / tau))


# ---------------------------------------------------------------------------
# Metrics on a single axis trace
# ---------------------------------------------------------------------------

def axis_metrics(w: np.ndarray, w_ref: float, dt: float) -> dict:
    finite = bool(np.all(np.isfinite(w)))
    peak = float(np.max(w))
    threshold = 0.9 * w_ref
    above = np.where(w >= threshold)[0]
    rise = float(above[0] * dt) if len(above) > 0 else float("inf")
    overshoot = max(0.0, (peak - w_ref) / w_ref) * 100.0
    n_settle = max(1, int(0.1 / dt))
    final = float(w[-n_settle:].mean())
    ess = abs(final - w_ref) / abs(w_ref) * 100.0
    return {
        "rise": rise, "overshoot": overshoot, "ess": ess,
        "peak": peak, "final": final, "finite": finite,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_axes(
    histories: dict[int, np.ndarray],
    z_histories: dict[int, np.ndarray],
    kp_vec: list[float], kd_vec: list[float],
    inertias: tuple[float, float, float], w_ref: float,
    metric_window_s: float, output: Path,
) -> Path:
    """3-row layout (PR-J):
       row 0: zoom-in 0~60ms — rise dynamics.
       row 1: full metric window 0~0.3s — settling.
       row 2: z position over full step phase — hover stability proof.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharey=False)
    n_step = next(iter(histories.values())).shape[0]
    t = np.arange(n_step) * DT
    zoom_xlim = 0.06
    full_xlim = metric_window_s

    for ax_idx, ax_name in enumerate(AXIS_NAMES):
        w_obs = histories[ax_idx]
        z_obs = z_histories[ax_idx]
        w_pred = predict_continuous(
            kp=kp_vec[ax_idx], kd=kd_vec[ax_idx],
            inertia=inertias[ax_idx], w_ref=w_ref, t=t,
        )
        m = axis_metrics(w_obs[: int(metric_window_s / DT)], w_ref, DT)
        rise_str = f"{m['rise']*1000:.1f}ms" if m["rise"] != float("inf") else "—"
        tau_ms = (inertias[ax_idx] + kd_vec[ax_idx]) / kp_vec[ax_idx] * 1e3

        # ----- top: zoom-in (rise dynamics) -----
        ax = axes[0, ax_idx]
        ax.axhline(w_ref, color="k", ls="--", lw=1, alpha=0.7, label=f"ref = {w_ref:.3f}")
        ax.axhline(0.9 * w_ref, color="gray", ls=":", lw=1, alpha=0.5, label="90% ref")
        ax.plot(t, w_pred, color="red", lw=1.5, ls="--", alpha=0.7,
                label=f"continuous-time (τ={tau_ms:.1f}ms)")
        ax.plot(t, w_obs, color="C0", lw=2.0, marker="o", markersize=3.5,
                label="Genesis sim (env-mean)")
        ax.text(
            0.97, 0.04,
            f"Kp = {kp_vec[ax_idx]:.4g}\n"
            f"Kd = {kd_vec[ax_idx]:.4g}\n"
            f"rise (90%) = {rise_str}\n"
            f"overshoot  = {m['overshoot']:.2f}%\n"
            f"ess        = {m['ess']:.2f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
        )
        ax.set_title(f"{ax_name}  (I = {inertias[ax_idx]*1e4:.2f}e-4 kg·m²)  — zoom")
        ax.set_xlim(0, zoom_xlim)
        ax.set_ylim(-0.4, w_ref * 1.25)
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("body rate ω (rad/s)")

        # ----- middle: full metric window (settling) -----
        ax = axes[1, ax_idx]
        ax.axhline(w_ref, color="k", ls="--", lw=1, alpha=0.7)
        ax.axhline(0.9 * w_ref, color="gray", ls=":", lw=1, alpha=0.5)
        ax.plot(t, w_pred, color="red", lw=1.2, ls="--", alpha=0.6)
        ax.plot(t, w_obs, color="C0", lw=1.5)
        ax.set_xlim(0, full_xlim)
        ax.set_ylim(-0.4, w_ref * 1.25)
        ax.set_title(f"{ax_name} — full metric window (0~{full_xlim}s)")
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("body rate ω (rad/s)")

        # ----- bottom: z position (hover stability proof, PR-J) -----
        ax = axes[2, ax_idx]
        z_drop_metric = 1.0 - z_obs[int(metric_window_s / DT) - 1]
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5, label="z_initial=1.0m")
        ax.axhline(0.95, color="orange", ls=":", lw=1, alpha=0.4, label="z_drop=0.05m line")
        ax.axvline(metric_window_s, color="orange", ls="--", lw=1, alpha=0.4,
                   label=f"metric window={metric_window_s}s")
        ax.plot(t, z_obs, color="C2", lw=1.5, label="z(t)")
        ax.text(
            0.97, 0.04,
            f"z_drop @ {metric_window_s}s = {z_drop_metric*1000:.1f}mm\n"
            f"z @ end = {z_obs[-1]:.4f}m",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
        )
        ax.set_title(f"{ax_name} — z position (step phase 1.0s)")
        ax.set_xlabel("t (s)")
        ax.set_xlim(0, T_STEP_S)
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("z position (m)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Tuned body-rate PD step response — Nova drone, ω_ref = +π rad/s\n"
                 "(rise/settling + hover stability via z position)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Trajectory mode (PR-K) — yaml-defined action sequence
# ---------------------------------------------------------------------------

def load_trajectory_yaml(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    traj = raw.get("trajectory")
    if traj is None:
        raise KeyError(f"no 'trajectory' block in {path}")
    return traj


def build_action_sequence(
    traj_cfg: dict, n_envs: int, dt: float,
) -> tuple[torch.Tensor, dict]:
    """yaml의 trajectory cfg → action 시퀀스 (T, n_envs, 4) ∈ [-1, 1].

    Returns (actions, info_dict).
    """
    traj_type = str(traj_cfg["type"])
    duration_s = float(traj_cfg["duration_s"])
    thrust_norm = float(traj_cfg.get("thrust_norm", 0.0))
    params = traj_cfg.get("params", {}) or {}
    n_step = int(round(duration_s / dt))
    t = np.arange(n_step) * dt                      # shape (T,)

    omega_ref_norm = np.zeros((n_step, 3), dtype=np.float64)

    if traj_type == "yaw_spin":
        wz = float(params["wz_norm"])
        omega_ref_norm[:, 2] = wz

    elif traj_type == "coning":
        amp = float(params["amplitude"])
        f = float(params["frequency_hz"])
        omega_ref_norm[:, 0] = amp * np.sin(2.0 * np.pi * f * t)   # wx
        omega_ref_norm[:, 1] = amp * np.cos(2.0 * np.pi * f * t)   # wy

    elif traj_type == "figure8":
        amp_x = float(params["amp_x"])
        amp_y = float(params["amp_y"])
        fx = float(params["freq_x_hz"])
        fy = float(params["freq_y_hz"])
        omega_ref_norm[:, 0] = amp_x * np.sin(2.0 * np.pi * fx * t)
        omega_ref_norm[:, 1] = amp_y * np.sin(2.0 * np.pi * fy * t)

    elif traj_type == "random_smooth":
        sigma = float(params["sigma"])
        tau = float(params["correlation_time_s"])
        seed = int(params.get("seed", 42))
        rng = np.random.default_rng(seed)
        alpha = dt / (tau + dt)                      # OU process coef
        for k in range(1, n_step):
            innov = sigma * rng.standard_normal(3)
            omega_ref_norm[k] = (1.0 - alpha) * omega_ref_norm[k - 1] + alpha * innov

    else:
        raise ValueError(f"unknown trajectory type: {traj_type!r}")

    # action shape (T, 4): [T_norm, wx_ref_norm, wy_ref_norm, wz_ref_norm]
    actions = np.zeros((n_step, 4), dtype=np.float32)
    actions[:, 0] = thrust_norm
    actions[:, 1:4] = omega_ref_norm.astype(np.float32)
    actions = np.clip(actions, -1.0, 1.0)

    # broadcast to n_envs
    actions_t = torch.tensor(actions, device=gs.device, dtype=gs.tc_float)
    actions_t = actions_t.unsqueeze(1).expand(-1, n_envs, -1).contiguous()  # (T, B, 4)

    info = {
        "type": traj_type,
        "duration_s": duration_s,
        "thrust_norm": thrust_norm,
        "params": dict(params),
        "n_step": n_step,
        "dt": dt,
    }
    return actions_t, info


def run_trajectory_trial(
    scene, drone, params: DroneParams,
    kp_vec: list[float], kd_vec: list[float], max_body_rate: float,
    actions_seq: torch.Tensor, n_envs: int, identity_q: torch.Tensor,
    n_stable: int, z_initial: float = 1.0,
) -> dict:
    """actions_seq: (T, B, 4). Stabilize → trajectory 실행, 측정 반환."""
    spawn = torch.zeros((n_envs, 3), device=gs.device, dtype=gs.tc_float)
    spawn[:, 2] = float(z_initial)
    drone.set_pos(spawn, zero_velocity=True)
    drone.set_quat(identity_q, zero_velocity=True)

    ctrl = BodyRateController(
        num_envs=n_envs, dt=DT, params=params, device=gs.device,
        kp_rate=tuple(kp_vec), kd_rate=tuple(kd_vec), max_body_rate=max_body_rate,
    )

    a_zero = torch.zeros((n_envs, 4), device=gs.device, dtype=gs.tc_float)
    for _ in range(n_stable):
        ang = drone.get_ang()
        rpms = ctrl.step(a_zero, ang)
        drone.set_propellers_rpm(rpms)
        scene.step()

    n_step = actions_seq.shape[0]
    omega_obs = torch.zeros((n_step, n_envs, 3), device=gs.device, dtype=gs.tc_float)
    pos = torch.zeros((n_step, n_envs, 3), device=gs.device, dtype=gs.tc_float)
    for k in range(n_step):
        ang = drone.get_ang()
        rpms = ctrl.step(actions_seq[k], ang)
        drone.set_propellers_rpm(rpms)
        scene.step()
        omega_obs[k] = drone.get_ang()
        pos[k] = drone.get_pos()

    return {
        "omega_obs": omega_obs.cpu().numpy(),    # (T, B, 3)
        "pos": pos.cpu().numpy(),                # (T, B, 3)
        "actions": actions_seq.cpu().numpy(),    # (T, B, 4)
    }


def plot_trajectory(
    result: dict, info: dict, max_body_rate: float, output: Path,
) -> Path:
    from mpl_toolkits.mplot3d import Axes3D       # noqa: F401  (registers projection)

    omega_obs = result["omega_obs"][:, 0, :]       # (T, 3) — first env
    pos = result["pos"][:, 0, :]                   # (T, 3)
    actions = result["actions"][:, 0, :]           # (T, 4)
    omega_ref = actions[:, 1:4] * max_body_rate    # (T, 3)
    n_step = omega_obs.shape[0]
    t = np.arange(n_step) * info["dt"]

    fig = plt.figure(figsize=(15, 9))
    gs_ = fig.add_gridspec(3, 4, width_ratios=[1.4, 1, 1, 1], wspace=0.30, hspace=0.35)

    # Left tall: 3D position trajectory
    ax3d = fig.add_subplot(gs_[:, 0], projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="C0", lw=1.5)
    ax3d.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]], color="green", s=60, label="start")
    ax3d.scatter([pos[-1, 0]], [pos[-1, 1]], [pos[-1, 2]], color="red", s=60, label="end")
    ax3d.set_title(f"3D position\n({info['type']}, {info['duration_s']}s)")
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.legend(loc="upper right", fontsize=8)

    # Right: 3 axes ω_obs vs ω_ref + tracking error
    axis_names = ("wx (roll)", "wy (pitch)", "wz (yaw)")
    for i in range(3):
        ax = fig.add_subplot(gs_[i, 1])
        ax.plot(t, omega_ref[:, i], color="red", ls="--", lw=1.2,
                label="ω_ref" if i == 0 else None)
        ax.plot(t, omega_obs[:, i], color="C0", lw=1.5,
                label="ω_obs" if i == 0 else None)
        ax.set_title(axis_names[i])
        ax.grid(alpha=0.3)
        ax.set_ylabel("rad/s")
        if i == 2:
            ax.set_xlabel("t (s)")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

        # tracking error
        ax_err = fig.add_subplot(gs_[i, 2])
        err = omega_obs[:, i] - omega_ref[:, i]
        ax_err.plot(t, err, color="C3", lw=1.0)
        ax_err.axhline(0, color="k", lw=0.5)
        ax_err.set_title(f"e[{i}] = ω_obs − ω_ref")
        ax_err.grid(alpha=0.3)
        ax_err.set_ylabel("rad/s")
        if i == 2:
            ax_err.set_xlabel("t (s)")

        # action ref (normalized)
        ax_a = fig.add_subplot(gs_[i, 3])
        ax_a.plot(t, actions[:, i + 1], color="C2", lw=1.0)
        ax_a.axhline(0, color="k", lw=0.5)
        ax_a.set_title(f"action[{i+1}] (norm)")
        ax_a.set_ylim(-1.1, 1.1)
        ax_a.grid(alpha=0.3)
        if i == 2:
            ax_a.set_xlabel("t (s)")

    # Tracking metrics annotation
    rmse = np.sqrt(np.mean((omega_obs - omega_ref) ** 2, axis=0))
    z_drift = float(pos[-1, 2] - pos[0, 2])
    fig.suptitle(
        f"Trajectory tracking — type='{info['type']}', "
        f"thrust_norm={info['thrust_norm']:.2f}, max_body_rate={max_body_rate:.3f} rad/s\n"
        f"RMSE [wx, wy, wz] = [{rmse[0]:.3f}, {rmse[1]:.3f}, {rmse[2]:.3f}] rad/s   "
        f"|   z drift = {z_drift*1000:+.1f} mm",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Viewer mode (sequential per-axis demo)
# ---------------------------------------------------------------------------

def run_viewer_demo(
    scene, drone, params: DroneParams,
    kp_vec: list[float], kd_vec: list[float], max_body_rate: float,
    n_envs: int, identity_q: torch.Tensor,
) -> None:
    """3 axis × (0.5s stabilize + 1.0s step) 순차. 사이에 reset."""
    n_stable = int(T_STABLE_S / DT)
    n_step = int(T_STEP_S / DT)
    print(f"[viewer] sequential demo — each axis 1.5s, total ~4.5s")
    for axis_idx, ax_name in enumerate(AXIS_NAMES):
        print(f"[viewer] axis = {ax_name}")
        history = run_axis_trial(
            scene=scene, drone=drone, params=params,
            kp_vec=kp_vec, kd_vec=kd_vec, max_body_rate=max_body_rate,
            axis_idx=axis_idx, n_envs=n_envs, identity_q=identity_q,
            n_stable=n_stable, n_step=n_step,
        )
        m = axis_metrics(history.mean(dim=1).cpu().numpy()[: int(0.3 / DT)],
                         math.pi, DT)
        rise_str = f"{m['rise']*1000:.1f}ms" if m["rise"] != float("inf") else "inf"
        print(f"  rise={rise_str} os={m['overshoot']:.2f}% ess={m['ess']:.2f}% peak={m['peak']:.3f}")


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

    if args.kp is not None and args.kd is not None:
        kp_vec = list(args.kp)
        kd_vec = list(args.kd)
        max_rate = float(args.max_body_rate)
        print(f"[gains] from CLI: Kp={kp_vec}, Kd={kd_vec}, max_rate={max_rate}")
    else:
        kp_vec, kd_vec, max_rate = load_gains_from_yaml(Path(args.config))
        print(f"[gains] from {args.config}: Kp={kp_vec}, Kd={kd_vec}, max_rate={max_rate}")

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")

    try:
        return _run(args, kp_vec, kd_vec, max_rate)
    except Exception as exc:                  # noqa: BLE001
        import traceback
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3
    finally:
        _cleanup()


def _run(args, kp_vec, kd_vec, max_rate) -> int:
    params = DroneParams(args.drone_yaml)
    B = int(args.num_envs)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=16),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,
            camera_pos=(2.5, -2.5, 2.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=45,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=DT,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=bool(args.show_viewer),
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
    inertias = (float(params.inertia[0]), float(params.inertia[1]),
                float(params.inertia[2]))

    # ----- trajectory mode (PR-K) — viewer는 scene 자체에 attach되므로
    # show_viewer 켜져있으면 trajectory가 3D viewer에 자동 표시됨.
    if args.mode == "trajectory":
        traj_cfg = load_trajectory_yaml(Path(args.trajectory_yaml))
        actions_seq, info = build_action_sequence(traj_cfg, n_envs=B, dt=DT)
        print(f"[trajectory] type={info['type']}, duration={info['duration_s']}s, "
              f"n_step={info['n_step']}, params={info['params']}")
        if args.show_viewer:
            print("[viewer] trajectory가 Genesis 3D viewer에 표시됩니다. "
                  "trajectory 끝나면 창이 자동으로 닫힙니다.")
        result = run_trajectory_trial(
            scene=scene, drone=drone, params=params,
            kp_vec=kp_vec, kd_vec=kd_vec, max_body_rate=max_rate,
            actions_seq=actions_seq, n_envs=B, identity_q=identity_q,
            n_stable=n_stable, z_initial=float(traj_cfg.get("z_initial", 1.0)),
        )
        # Quick console summary
        omega_obs = result["omega_obs"][:, 0, :]
        omega_ref = result["actions"][:, 0, 1:4] * max_rate
        rmse = np.sqrt(np.mean((omega_obs - omega_ref) ** 2, axis=0))
        z_drift = float(result["pos"][-1, 0, 2] - result["pos"][0, 0, 2])
        print(f"[summary] RMSE [wx,wy,wz] = [{rmse[0]:.4f}, {rmse[1]:.4f}, {rmse[2]:.4f}] rad/s  "
              f"z_drift = {z_drift*1000:+.2f}mm")

        out = Path(args.output) if args.output else (
            ROOT / "outputs" / "rate_pid_tuning" / f"trajectory_{info['type']}.png"
        )
        plot_trajectory(result, info, max_rate, out)
        print(f"[plot] saved to {out}")
        return 0

    # ----- step mode (default) -----
    if args.show_viewer:
        run_viewer_demo(
            scene, drone, params, kp_vec, kd_vec, max_rate,
            n_envs=B, identity_q=identity_q,
        )
        return 0

    # Plot path: collect axis histories + z trajectories.
    histories: dict[int, np.ndarray] = {}
    z_histories: dict[int, np.ndarray] = {}
    for axis_idx, ax_name in enumerate(AXIS_NAMES):
        debug = (axis_idx == 0)   # first axis prints stabilize debug
        if debug:
            print(f"\n[stabilize debug — {ax_name} axis trial]")
        hist, z_hist = run_axis_trial(
            scene=scene, drone=drone, params=params,
            kp_vec=kp_vec, kd_vec=kd_vec, max_body_rate=max_rate,
            axis_idx=axis_idx, n_envs=B, identity_q=identity_q,
            n_stable=n_stable, n_step=n_step,
            debug_stabilize=debug,
        )
        w_np = hist.mean(dim=1).cpu().numpy()
        z_np = z_hist.mean(dim=1).cpu().numpy()
        histories[axis_idx] = w_np
        z_histories[axis_idx] = z_np
        m = axis_metrics(w_np[: int(0.3 / DT)], math.pi, DT)
        rise_str = f"{m['rise']*1000:.1f}ms" if m["rise"] != float("inf") else "inf"
        z_drop_metric = 1.0 - z_np[int(0.3 / DT) - 1]
        print(f"[{ax_name:5s}] rise={rise_str} os={m['overshoot']:.2f}% "
              f"ess={m['ess']:.2f}% peak={m['peak']:.3f} "
              f"z_drop@0.3s={z_drop_metric*1000:.1f}mm")

    out = Path(args.output) if args.output else (
        ROOT / "outputs" / "rate_pid_tuning" / "tuned_response.png"
    )
    plot_axes(histories, z_histories, kp_vec, kd_vec, inertias, w_ref=math.pi,
              metric_window_s=0.3, output=out)
    print(f"[plot] saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
