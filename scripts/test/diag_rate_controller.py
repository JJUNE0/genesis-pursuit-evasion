"""Phase B+ pre-flight diagnostic — BodyRateController layer audit.

Targets the 3-way scenario split (CLAUDE.md DA mode):
  A. reset transient   — first 1-2 steps after reset have ang_acc spikes
                         due to last_ang_vel=0 vs ang_vel(spawn) ≠ 0.
  B. bang-bang         — sustained ±max rate_ref → mixer saturation,
                         negative thrust clamped to 0 → wrench loss.
  C. mixer / numerical — condition(M) > 100, or hover sanity broken
                         (4-motor RPM not equal at T_norm=0, ω_ref=0).

No env / sim — direct controller unit tests with synthetic inputs.
Wall-clock < 5 s.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs   # noqa: E402

from utils.drone_params import DroneParams   # noqa: E402
from utils.rate_controller import (   # noqa: E402
    BodyRateController, build_mixer_matrix,
)


# ---------------------------------------------------------------------------
# Scenario A — reset transient
# ---------------------------------------------------------------------------

def diag_reset_transient(ctrl: BodyRateController, params: DroneParams) -> dict:
    """Simulate spawn drift: ang_vel ≠ 0 right after reset.

    Genesis spawns with small numerical noise — we test 3 typical magnitudes:
    0, 0.1, 0.5 rad/s. Compare ang_acc, torque, saturation.
    """
    out = {}
    for ω0 in (0.0, 0.1, 0.5):
        ctrl.reset()  # last_ang_vel ← 0
        action = torch.zeros((1, 4), device=ctrl.device)  # hover command
        ang_vel = torch.tensor([[ω0, ω0, ω0]], device=ctrl.device, dtype=torch.float32)
        # Inspect internal pre-clamp values via re-implementing the math.
        rate_ref = action[:, 1:4] * ctrl.max_body_rate
        rate_err = (rate_ref - ang_vel).abs().max().item()
        ang_acc = ((ang_vel - ctrl.last_ang_vel) / ctrl.dt).abs().max().item()
        torque = (
            ctrl.kp * (rate_ref - ang_vel) - ctrl.kd * ((ang_vel - ctrl.last_ang_vel) / ctrl.dt)
        ).abs().max().item()
        rpms = ctrl.step(action, ang_vel)
        sat_max = (rpms >= ctrl.max_rpm - 1.0).float().mean().item()
        sat_zero = (rpms <= 1.0).float().mean().item()
        out[f"ω0={ω0}"] = {
            "rate_err_max":  round(rate_err, 4),
            "ang_acc_max":   round(ang_acc, 2),
            "torque_max":    round(torque, 5),
            "rpms_min":      round(rpms.min().item(), 1),
            "rpms_max":      round(rpms.max().item(), 1),
            "sat_max_frac":  round(sat_max, 3),
            "sat_zero_frac": round(sat_zero, 3),
        }
    return out


# ---------------------------------------------------------------------------
# Scenario B — bang-bang sustained
# ---------------------------------------------------------------------------

def diag_bang_bang(ctrl: BodyRateController, params: DroneParams) -> dict:
    """Sustained ±max rate_ref. Should saturate motors hard."""
    ctrl.reset()
    out = {}
    test_actions = [
        ("bang_+all",  [+1.0, +1.0, +1.0, +1.0]),
        ("bang_T0",    [-1.0, +1.0, +1.0, +1.0]),  # zero thrust + max torque
        ("bang_-all",  [-1.0, -1.0, -1.0, -1.0]),
        ("hover",      [ 0.0,  0.0,  0.0,  0.0]),
    ]
    for name, a in test_actions:
        ctrl.reset()
        action = torch.tensor([a], device=ctrl.device, dtype=torch.float32)
        ang_vel = torch.zeros((1, 3), device=ctrl.device, dtype=torch.float32)
        # Run 5 sustained steps; record final-step saturation.
        for _ in range(5):
            rpms = ctrl.step(action, ang_vel)
        # Inspect pre-clamp thrusts by recomputing.
        T_norm = action[:, 0:1]
        rate_ref = action[:, 1:4] * ctrl.max_body_rate
        T_total = (T_norm + 1.0) * ctrl.hover_thrust_total
        rate_err = rate_ref - ang_vel
        ang_acc = (ang_vel - ctrl.last_ang_vel) / ctrl.dt
        torque = ctrl.kp * rate_err - ctrl.kd * ang_acc
        wrench = torch.cat([T_total, torque], dim=1)
        thrusts_unclipped = wrench @ ctrl.M_inv_T
        thrusts_clipped = thrusts_unclipped.clamp(
            min=0.0, max=ctrl.max_thrust_per_motor
        )
        wrench_loss_frac = float(
            (thrusts_unclipped - thrusts_clipped).abs().sum(dim=1)
            / thrusts_unclipped.abs().sum(dim=1).clamp_min(1e-6)
        )
        n_neg = int((thrusts_unclipped < 0.0).sum().item())
        n_over = int((thrusts_unclipped > ctrl.max_thrust_per_motor).sum().item())
        out[name] = {
            "T_total_N":          round(T_total.item(), 3),
            "torque_max_Nm":      round(torque.abs().max().item(), 4),
            "thrusts_unclipped":  [round(x, 3) for x in thrusts_unclipped[0].tolist()],
            "n_neg_thrust":       n_neg,
            "n_oversaturated":    n_over,
            "wrench_loss_frac":   round(wrench_loss_frac, 3),
            "rpms":               [round(x, 1) for x in rpms[0].tolist()],
        }
    return out


# ---------------------------------------------------------------------------
# Scenario C — mixer / numerical
# ---------------------------------------------------------------------------

def diag_mixer(params: DroneParams) -> dict:
    """Mixer matrix conditioning + hover sanity.

    A well-conditioned X-quadcopter has cond(M) ≈ 5-10. ≥ 100 = degenerate.
    Hover sanity: T_norm=0, all torques=0 → 4 motors share thrust equally.
    """
    M = build_mixer_matrix(params)
    cond = float(np.linalg.cond(M))
    M_inv = np.linalg.inv(M) if M.shape[0] == M.shape[1] else np.linalg.pinv(M)
    # Hover wrench: T = m·g, τ = 0
    hover_thrust = float(params.mass * params.gravity)
    wrench_hover = np.array([hover_thrust, 0.0, 0.0, 0.0], dtype=np.float64)
    f_per_motor = M_inv @ wrench_hover
    expected = hover_thrust / params.n_propellers
    deviation = float(np.abs(f_per_motor - expected).max())
    return {
        "cond_M":          round(cond, 2),
        "M_shape":         list(M.shape),
        "hover_thrust_N":  round(hover_thrust, 3),
        "expected_per_motor_N": round(expected, 3),
        "actual_per_motor_N":   [round(x, 3) for x in f_per_motor.tolist()],
        "max_deviation_N": round(deviation, 6),
        "hover_sane":      bool(deviation < 1e-6),
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def render_verdict(rt: dict, bb: dict, mx: dict) -> list[str]:
    out = []
    # A — reset transient
    high_acc = rt["ω0=0.5"]["ang_acc_max"]
    if high_acc > 50.0:
        out.append(f"A (reset transient) STRONG: ang_acc spikes to {high_acc:.0f} rad/s² with ω0=0.5")
    elif high_acc > 10.0:
        out.append(f"A (reset transient) WEAK: ang_acc {high_acc:.0f} at ω0=0.5")
    # B — bang-bang
    bang = bb.get("bang_T0", {})
    if bang.get("n_neg_thrust", 0) > 0 or bang.get("wrench_loss_frac", 0) > 0.1:
        out.append(
            f"B (bang-bang) STRONG: T_norm=-1+max τ → {bang['n_neg_thrust']} neg thrusts, "
            f"wrench_loss={bang['wrench_loss_frac']:.2f}"
        )
    bang_max = bb.get("bang_+all", {})
    if bang_max.get("n_oversaturated", 0) > 0:
        out.append(
            f"B (bang-bang) STRONG: T_norm=+1+max τ → {bang_max['n_oversaturated']} oversaturated"
        )
    # C — mixer
    if mx["cond_M"] >= 100.0:
        out.append(f"C (numerical) STRONG: cond(M)={mx['cond_M']:.0f}")
    if not mx["hover_sane"]:
        out.append(f"C (mixer) STRONG: hover deviation {mx['max_deviation_N']:.3e}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    gs.init(backend=gs.cpu, seed=0, logging_level="warning")
    params = DroneParams(str(ROOT / "configs" / "drones" / "nova.yaml"))
    print("---- DroneParams ----")
    print(f"mass={params.mass:.4f} kg  gravity={params.gravity:.2f} m/s²  "
          f"hover_thrust_total={params.mass*params.gravity:.3f} N")
    print(f"n_props={params.n_propellers}  kf={params.kf:.3e}  "
          f"max_thrust_per_motor={params.max_thrust_per_motor:.3f} N")
    print(f"inertia={params.inertia}")
    print()

    ctrl = BodyRateController(
        num_envs=1, dt=0.005, params=params, device=torch.device("cpu"),
        kp_rate=(0.05, 0.10, 0.10), kd_rate=(1e-4, 5e-5, 1e-4),
        max_body_rate=3.141592653589793,
    )

    print("---- Scenario A: reset transient ----")
    rt = diag_reset_transient(ctrl, params)
    for k, v in rt.items():
        print(f"  {k}: {v}")
    print()

    print("---- Scenario B: bang-bang sustained ----")
    bb = diag_bang_bang(ctrl, params)
    for k, v in bb.items():
        print(f"  {k}: {v}")
    print()

    print("---- Scenario C: mixer / numerical ----")
    mx = diag_mixer(params)
    for k, v in mx.items():
        print(f"  {k}: {v}")
    print()

    verdicts = render_verdict(rt, bb, mx)
    print("---- VERDICT ----")
    if verdicts:
        for v in verdicts:
            print(f"  • {v}")
    else:
        print("  (no scenario STRONG. Re-evaluate Phase B+ reward processing.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
