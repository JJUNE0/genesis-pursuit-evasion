"""V1 precondition — single-drone batched hover with strict tolerances.

Why this exists
---------------
The dual-drone V1 test (``two_drones_in_one_env.py``) silently presupposes:

  1.  ``params.gravity`` is positive in the magnitude sense (its sign is
      consistent with ``thrust = m·g`` for hover).
  2.  ``params.kf`` follows the **RPM²** convention — i.e.
      ``thrust = kf · ω²`` with ω in **rev/min**, which is what
      ``Drone.set_propellers_rpm`` consumes.
  3.  ``params.spin_directions`` are torque-balanced enough to keep yaw
      stationary under uniform RPM (the URDF prop offsets aren't biased).

If **any** of these is wrong, ``two_drones_in_one_env.py`` will fail in a way
that *looks* like a Plan A architectural problem but is actually a unit /
convention / URDF bug — pushing us to Plan B for the wrong reason.

This script isolates those preconditions on a **single** drone with **strict**
tolerances (z_tol=0.01 m, yaw_tol=0.1°). It is the gate that must PASS before
the dual-drone V1 result is interpretable.

Pass criteria
-------------
1.  ``params.gravity > 0`` and inside [5, 15] m/s² — print and assert.
2.  ``hover_rpm = sqrt(m·g / (n_props·kf))`` lands in [1000, 50000] RPM —
    if outside, ``kf`` is almost certainly using rad/s convention.
3.  After ``--steps`` steps (default 300 = 3.0 s) of constant hover RPM:
       max |z - 1.0| < ``--z_tol``   (default 0.01 m)
       max |xy|     < ``--xy_tol``   (default 0.05 m)
       max |vel|    < ``--vel_tol``  (default 0.10 m/s)
       max |yaw|    < ``--yaw_tol_deg`` (default 0.1°)
4.  No NaN / Inf in pos, vel, quat across the whole batch.

Failure → fix the convention before trusting any dual-drone result.

Usage
-----
    python tests/sanity/single_drone_hover.py
    python tests/sanity/single_drone_hover.py --num_envs 64 --drone_yaml configs/drones/nova.yaml
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs  # noqa: E402

from utils.drone_params import DroneParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V1 precondition: single-drone hover")
    p.add_argument("--num_envs", type=int, default=64, help="batch size B (single drone is cheap)")
    p.add_argument("--steps", type=int, default=300, help="rollout length (steps); dt=0.01s")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--drone_yaml",
        type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
    )
    p.add_argument("--z_tol", type=float, default=0.01, help="strict |z - 1.0| (m)")
    p.add_argument("--xy_tol", type=float, default=0.05, help="strict |xy| drift (m)")
    p.add_argument("--vel_tol", type=float, default=0.10, help="strict |vel| (m/s)")
    p.add_argument("--yaw_tol_deg", type=float, default=0.1, help="strict |yaw| change (deg)")
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--show_viewer", action="store_true", help="open Genesis viewer for visual check")
    p.add_argument("--num_rendered_envs", type=int, default=4, help="how many envs to draw (clamped to num_envs)")
    p.add_argument("--max_fps", type=int, default=60, help="viewer FPS cap")
    p.add_argument(
        "--env_spacing",
        type=float,
        default=None,
        help="distance between env origins (m). default: 3.0 with --show_viewer (so envs don't overlap), 0.0 otherwise",
    )
    return p.parse_args()


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not Path(args.drone_yaml).is_file():
        print(f"[FAIL] DroneParams YAML not found: {args.drone_yaml}", file=sys.stderr)
        return 2

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")

    try:
        return _run(args)
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3
    finally:
        _cleanup()


def _run(args: argparse.Namespace) -> int:
    params = DroneParams(args.drone_yaml)
    B = int(args.num_envs)
    T = int(args.steps)
    dt = 0.01

    # ---------------------------------------------------------------------
    # P0 — Convention diagnostics. Loud and explicit so any mismatch is
    # obvious *before* the simulation runs.
    # ---------------------------------------------------------------------
    print(f"[params] name           : {params.name}")
    print(f"[params] mass           : {params.mass} kg")
    print(f"[params] gravity        : {params.gravity}  (sign matters!)")
    print(f"[params] kf             : {params.kf:.3e}  (assumed: thrust = kf * RPM^2)")
    print(f"[params] n_propellers   : {params.n_propellers}")
    print(f"[params] spin_directions: {params.spin_directions.tolist()}")
    print(f"[params] prop_link_names: {params.prop_link_names}")

    # Convention guard #1: gravity sign / magnitude.
    g = float(params.gravity)
    if g <= 0:
        print(
            f"[FAIL] params.gravity={g} is non-positive. "
            "DroneParams should expose g as a positive magnitude (sqrt(m*g/...) needs g>0).",
            file=sys.stderr,
        )
        return 1
    if not (5.0 <= g <= 15.0):
        print(
            f"[FAIL] params.gravity={g} m/s² is outside plausible Earth range [5, 15].",
            file=sys.stderr,
        )
        return 1

    # Convention guard #2: spin sum should be ~0 (torque-balanced X-frame).
    spin_sum = float(np.sum(params.spin_directions))
    if abs(spin_sum) > 1e-6:
        print(
            f"[WARN] spin_directions sum to {spin_sum} (≠0). "
            "Yaw will not be torque-balanced under uniform RPM — yaw_tol may trip "
            "for reasons unrelated to Plan A."
        )

    # Convention guard #3: hover_rpm must land in a typical RPM band.
    n = float(params.n_propellers)
    hover_rpm = float(np.sqrt(params.mass * g / (n * params.kf)))
    print(f"[hover] RPM = sqrt(m*g / (n*kf)) = {hover_rpm:.1f}")
    if not (1000.0 <= hover_rpm <= 50000.0):
        print(
            f"[FAIL] hover RPM={hover_rpm:.1f} is outside [1000, 50000]. "
            "Most likely kf is on a rad/s² basis instead of RPM². "
            "If thrust = kf * ω² with ω in rad/s, multiply hover_rpm by 60/(2π)≈9.55 "
            "to recover the correct RPM command. Adjust the YAML or the formula.",
            file=sys.stderr,
        )
        return 1

    # ---------------------------------------------------------------------
    # Scene with ONE drone (V1 isolates dual-drone questions; this is solo).
    # ---------------------------------------------------------------------
    n_rendered = max(1, min(int(args.num_rendered_envs), B))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=4),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(args.max_fps),
            camera_pos=(3.0, 0.0, 3.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(n_rendered))),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=args.show_viewer,
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
    spacing = args.env_spacing if args.env_spacing is not None else (3.0 if args.show_viewer else 0.0)
    scene.build(n_envs=B, env_spacing=(spacing, spacing))
    print(f"[scene] env_spacing={spacing}  rendered_envs={n_rendered}")

    # Spawn at z=1.0, identity orientation.
    spawn = torch.zeros((B, 3), device=gs.device, dtype=gs.tc_float)
    spawn[:, 2] = 1.0
    drone.set_pos(spawn, zero_velocity=True)

    quat0 = drone.get_quat().clone()  # shape: (B, 4)

    # ----------------------------------------------------------
    # P0-A: Quaternion ordering check.
    # Genesis is documented as (w, x, y, z) but we *verify* by
    # observing which component is ≈1 at identity orientation.
    # If wrong, all rotation math below produces silent garbage
    # that may incidentally pass the rotation tolerance.
    # ----------------------------------------------------------
    q_first_abs = float(quat0[0, 0].abs())
    q_last_abs = float(quat0[0, 3].abs())
    print(f"[diag] quat[0] = {quat0[0].tolist()}")
    print(f"[diag]   |q[0]|={q_first_abs:.4f}  |q[3]|={q_last_abs:.4f}")
    if q_first_abs > 0.99 and q_last_abs < 0.1:
        print("[diag]   → (w, x, y, z) order confirmed — rotation math is valid.")
    elif q_last_abs > 0.99 and q_first_abs < 0.1:
        print(
            "[FAIL] Genesis returned quat in (x, y, z, w) order. "
            "The rotation math in this script assumes w-first; "
            "swap _qmul ordering or rebuild quat0/quat with q[..., [3,0,1,2]] before "
            "trusting the body-rotation tolerance.",
            file=sys.stderr,
        )
        return 1
    else:
        print(
            f"[WARN] identity quat is neither w-first nor w-last (|q[0]|={q_first_abs}, "
            f"|q[3]|={q_last_abs}). The drone may have been spawned with non-identity "
            "orientation, or the convention is non-standard. Proceed with skepticism."
        )

    rpm_tensor = torch.full(
        (B, params.n_propellers),
        hover_rpm,
        device=gs.device,
        dtype=gs.tc_float,
    )  # shape: (B, n_props)

    # Roll out constant hover RPM.
    for _ in range(T):
        drone.set_propellers_rpm(rpm_tensor)
        scene.step()

    pos = drone.get_pos()    # shape: (B, 3)
    vel = drone.get_vel()    # shape: (B, 3)
    quat = drone.get_quat()  # shape: (B, 4)

    finite_all = bool(
        torch.isfinite(pos).all()
        and torch.isfinite(vel).all()
        and torch.isfinite(quat).all()
    )

    z_err = (pos[:, 2] - 1.0).abs()              # shape: (B,)
    xy_drift = pos[:, :2].norm(dim=1)            # shape: (B,)
    speed = vel.norm(dim=1)                      # shape: (B,)

    # Yaw drift. Genesis quaternions are (w, x, y, z); we extract a relative
    # rotation q_rel = q · q0⁻¹ and read its z-component for the yaw axis.
    # We *don't* import Genesis-specific helpers because their unit
    # conventions vary across versions — going through cosine of half-angle
    # is robust.
    q0 = quat0
    q1 = quat
    # Conjugate of q0 (unit quat assumption — drone is rigid).
    q0_inv = torch.empty_like(q0)
    q0_inv[:, 0] = q0[:, 0]
    q0_inv[:, 1:] = -q0[:, 1:]

    def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        w = a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3]
        x = a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2]
        y = a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1]
        z = a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0]
        return torch.stack([w, x, y, z], dim=1)

    q_rel = _qmul(q1, q0_inv)
    # Total angle of the relative rotation: 2 · acos(|w|). This bounds yaw,
    # roll, and pitch drift in one number. For a hover that should keep the
    # body upright, this is what the user actually cares about.
    cos_half = q_rel[:, 0].abs().clamp(-1.0, 1.0)
    angle_rad = 2.0 * torch.acos(cos_half)        # shape: (B,)
    angle_deg = angle_rad * (180.0 / math.pi)

    z_max = float(z_err.max())
    xy_max = float(xy_drift.max())
    v_max = float(speed.max())
    ang_max_deg = float(angle_deg.max())

    ok_z = z_max < args.z_tol
    ok_xy = xy_max < args.xy_tol
    ok_vel = v_max < args.vel_tol
    ok_yaw = ang_max_deg < args.yaw_tol_deg

    print("\n[result] after T steps:")
    print(f"  pos[0]              = {pos[0].tolist()}  vel[0] = {vel[0].tolist()}")
    print(f"  quat[0]             = {quat[0].tolist()}")
    print(f"  max |z-1.0|         = {z_max:.5f} m   (tol {args.z_tol})")
    print(f"  max horizontal err  = {xy_max:.5f} m   (tol {args.xy_tol})")
    print(f"  max |vel|           = {v_max:.5f} m/s (tol {args.vel_tol})")
    print(f"  max body rotation   = {ang_max_deg:.5f} deg (tol {args.yaw_tol_deg})")

    print("\n[verdict]")
    print(f"  no NaN/Inf:                  {finite_all}")
    print(f"  z holds 1.0 ± {args.z_tol}:        {ok_z}")
    print(f"  xy drift ≤ {args.xy_tol}:          {ok_xy}")
    print(f"  velocity ≤ {args.vel_tol}:         {ok_vel}")
    print(f"  body rotation ≤ {args.yaw_tol_deg}°:    {ok_yaw}")

    all_ok = all([finite_all, ok_z, ok_xy, ok_vel, ok_yaw])
    if all_ok:
        print(
            "\n[PASS] single-drone hover OK — kf/gravity/spin conventions are "
            "consistent. Dual-drone V1 results are now interpretable."
        )
        return 0
    print(
        "\n[FAIL] single-drone hover failed at strict tolerances. "
        "Do NOT trust dual-drone V1 verdict until this passes — the "
        "convention/URDF/solver quality is not yet established."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
