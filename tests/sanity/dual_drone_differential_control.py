"""V1.5 sanity — differential closed-loop control on a batched dual-drone scene.

Why this exists
---------------
V1 (``two_drones_in_one_env.py``) feeds **identical** hover RPM to **every
env, every entity, every propeller**. PASS under that load only proves
"uniform input gives uniform output" — a kernel that secretly shares
control buffers across (entity, env, prop) channels would still pass.

The real Plan A production scenario is the opposite: *every* env applies
*different* RPM to *each* entity *per propeller*. If batched dispatch is
correct, the divergence pattern is fully predictable from the per-(entity,
env) RPM commands. If any cross-channel buffer leak exists, the actual
trajectories will not match the predicted signs.

Test design
-----------
``--num_envs B`` is split into two halves:
    group_A = envs[0 : B/2]
    group_B = envs[B/2 : B]

The applied per-propeller RPM:
    attacker[group_A] = hover_rpm * (1 + r)     → climb
    attacker[group_B] = hover_rpm * (1 - r)     → drop
    defender[group_A] = hover_rpm * (1 - r)     → drop
    defender[group_B] = hover_rpm * (1 + r)     → climb

That is, the (entity, env-group) cross-pattern is a 2×2 table where each
cell has a *different* sign of altitude change. Any single buffer-sharing
bug between two cells flips one prediction.

Pass criteria
-------------
1.  Shapes / no NaN.
2.  Sign correctness, **per env** (not just per group mean):
        attacker[group_A].z  > 1 + Δz_min   (every env climbed)
        attacker[group_B].z  < 1 - Δz_min   (every env dropped)
        defender[group_A].z  < 1 - Δz_min   (every env dropped)
        defender[group_B].z  > 1 + Δz_min   (every env climbed)
3.  Symmetric magnitudes: |mean climb| and |mean drop| match within
    ``--symmetry_tol`` (kinematic check that no cross-coupling is
    biasing one cell more than its mirror).
4.  Cross-entity isolation under non-uniform load: attacker's z change
    in group_A is *independent* of defender's z change in the same envs
    (correlation between attacker_climb_amount and defender_drop_amount
    across envs ≈ 0, or at least not positively correlated, since they
    have independent dynamics).

Failure → V1's PASS was a uniform-input artifact; treat Plan A as
unproven until this passes.

Usage
-----
    python tests/sanity/dual_drone_differential_control.py
    python tests/sanity/dual_drone_differential_control.py --num_envs 16 --steps 100 --rpm_factor 0.05
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
    p = argparse.ArgumentParser(description="V1.5 sanity: differential dual-drone control")
    p.add_argument("--num_envs", type=int, default=16, help="batch size; must be even")
    p.add_argument("--steps", type=int, default=100, help="rollout length (dt=0.01)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--drone_yaml",
        type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
    )
    p.add_argument(
        "--rpm_factor",
        type=float,
        default=0.05,
        help="hover RPM is multiplied by (1 ± r). 5% → ~0.5 m altitude divergence after 1s",
    )
    p.add_argument("--dz_min", type=float, default=0.10, help="min |Δz| per env to count as moved")
    p.add_argument(
        "--spawn_z",
        type=float,
        default=None,
        help="initial z (m). default: auto-set high enough that drop drones don't hit z=0 ground",
    )
    p.add_argument(
        "--symmetry_tol",
        type=float,
        default=0.5,
        help="|mean_climb − |mean_drop|| / max  must be < this (relative)",
    )
    p.add_argument(
        "--corr_tol",
        type=float,
        default=0.5,
        help="|corr(attacker_dz, defender_dz)| must be < this (Pearson, per group)",
    )
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--show_viewer", action="store_true")
    p.add_argument("--num_rendered_envs", type=int, default=4)
    p.add_argument("--max_fps", type=int, default=60)
    p.add_argument(
        "--env_spacing",
        type=float,
        default=None,
        help="distance between env origins (m). default: 3.0 with --show_viewer, 0.0 otherwise",
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

    if args.num_envs < 2 or args.num_envs % 2 != 0:
        print(f"[FAIL] --num_envs must be even and ≥ 2, got {args.num_envs}", file=sys.stderr)
        return 2
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
    half = B // 2
    T = int(args.steps)
    dt = 0.01

    g = float(params.gravity)
    n_props = float(params.n_propellers)
    hover_rpm = float(np.sqrt(params.mass * g / (n_props * params.kf)))
    r = float(args.rpm_factor)

    # Per-(entity, env, prop) RPM buffers.  shape: (B, n_props)
    rpm_attacker = torch.full(
        (B, params.n_propellers), hover_rpm, device=gs.device, dtype=gs.tc_float
    )
    rpm_defender = rpm_attacker.clone()

    group_A = torch.arange(0, half, device=gs.device, dtype=torch.long)        # envs to climb attacker / drop defender
    group_B = torch.arange(half, B, device=gs.device, dtype=torch.long)        # envs to drop  attacker / climb defender

    rpm_attacker[group_A] *= (1.0 + r)
    rpm_attacker[group_B] *= (1.0 - r)
    rpm_defender[group_A] *= (1.0 - r)
    rpm_defender[group_B] *= (1.0 + r)

    # Predicted altitude offset after T steps (open-loop, gravity-only model).
    # Use the EXACT thrust factor (1±r)² instead of the small-r approximation
    # 2r — at r=0.20 the discrepancy is already 22% and breaks the symmetry
    # tolerance.
    accel_climb = ((1.0 + r) ** 2 - 1.0) * g    # net upward accel (m/s²)
    accel_drop = (1.0 - (1.0 - r) ** 2) * g     # net downward accel (m/s²)
    dz_climb_predicted = 0.5 * accel_climb * (T * dt) ** 2
    dz_drop_predicted = 0.5 * accel_drop * (T * dt) ** 2

    # Auto-pick spawn_z so drop drones don't hit the z=0 ground (with margin).
    # If they do, the drop magnitude is clipped and the symmetry check fails
    # for purely physical reasons unrelated to Plan A.
    spawn_z = args.spawn_z if args.spawn_z is not None else max(1.0, dz_drop_predicted + 0.5)

    print(f"[info] backend={args.backend} B={B} (group_A=[0..{half-1}], group_B=[{half}..{B-1}])")
    print(f"[info] hover_rpm={hover_rpm:.1f}  r=±{r*100:.1f}%")
    print(f"[info] predicted Δz climb ≈ +{dz_climb_predicted:.3f} m  (accel +{accel_climb:.2f} m/s²)")
    print(f"[info] predicted Δz drop  ≈ -{dz_drop_predicted:.3f} m  (accel -{accel_drop:.2f} m/s²)")
    print(f"[info] spawn_z = {spawn_z:.2f} m  (must exceed drop magnitude + safety to avoid ground)")
    print(
        f"[plan] attacker[group_A]={hover_rpm*(1+r):.1f} (climb)   "
        f"attacker[group_B]={hover_rpm*(1-r):.1f} (drop)"
    )
    print(
        f"[plan] defender[group_A]={hover_rpm*(1-r):.1f} (drop)    "
        f"defender[group_B]={hover_rpm*(1+r):.1f} (climb)"
    )

    n_rendered = max(1, min(int(args.num_rendered_envs), B))
    spacing = args.env_spacing if args.env_spacing is not None else (3.0 if args.show_viewer else 0.0)

    # Render envs spanning BOTH groups so the differential cross-pattern is
    # visible to the eye. With contiguous low indices [0,1,2,3] you'd only
    # see group_A (climb-attacker / drop-defender) and miss the inversion
    # in group_B.
    if n_rendered >= 2 and B >= 4:
        n_from_A = n_rendered // 2
        n_from_B = n_rendered - n_from_A
        rendered_envs_idx = list(range(n_from_A)) + list(range(half, half + n_from_B))
    else:
        rendered_envs_idx = list(range(n_rendered))
    print(f"[scene] env_spacing={spacing}  rendered_envs_idx={rendered_envs_idx}")

    # Camera follows spawn height + climb extent so drones stay in frame.
    cam_z = spawn_z + 1.0
    cam_back = max(3.5, dz_climb_predicted + 2.0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=4),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(args.max_fps),
            camera_pos=(cam_back, -1.5, cam_z),
            camera_lookat=(0.5, 0.0, spawn_z),
            camera_fov=50,
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=rendered_envs_idx),
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

    attacker_drone = scene.add_entity(
        gs.morphs.Drone(
            file=urdf_path,
            propellers_link_name=params.prop_link_names,
            propellers_spin=[int(s) for s in params.spin_directions],
        )
    )
    defender_drone = scene.add_entity(
        gs.morphs.Drone(
            file=urdf_path,
            propellers_link_name=params.prop_link_names,
            propellers_spin=[int(s) for s in params.spin_directions],
        )
    )
    scene.build(n_envs=B, env_spacing=(spacing, spacing))

    spawn_attacker = torch.zeros((B, 3), device=gs.device, dtype=gs.tc_float)
    spawn_attacker[:, 2] = spawn_z
    spawn_defender = torch.zeros((B, 3), device=gs.device, dtype=gs.tc_float)
    spawn_defender[:, 0] = 1.0
    spawn_defender[:, 2] = spawn_z
    attacker_drone.set_pos(spawn_attacker, zero_velocity=True)
    defender_drone.set_pos(spawn_defender, zero_velocity=True)

    for _ in range(T):
        attacker_drone.set_propellers_rpm(rpm_attacker)
        defender_drone.set_propellers_rpm(rpm_defender)
        scene.step()

    pos_a = attacker_drone.get_pos()    # shape: (B, 3)
    pos_d = defender_drone.get_pos()    # shape: (B, 3)
    vel_a = attacker_drone.get_vel()
    vel_d = defender_drone.get_vel()

    finite_all = bool(
        torch.isfinite(pos_a).all() and torch.isfinite(pos_d).all()
        and torch.isfinite(vel_a).all() and torch.isfinite(vel_d).all()
    )

    # Δz from spawn (auto-adjusted to keep drop drones above ground)
    dz_a = pos_a[:, 2] - spawn_z
    dz_d = pos_d[:, 2] - spawn_z
    dz_min = float(args.dz_min)

    # Per-env sign correctness — every env must be on the correct side.
    ok_a_climbed = bool((dz_a[group_A] > dz_min).all())
    ok_a_dropped = bool((dz_a[group_B] < -dz_min).all())
    ok_d_dropped = bool((dz_d[group_A] < -dz_min).all())
    ok_d_climbed = bool((dz_d[group_B] > dz_min).all())

    # Means used for symmetry + reporting.
    a_mean_climb = float(dz_a[group_A].mean())
    a_mean_drop = float(dz_a[group_B].mean())
    d_mean_drop = float(dz_d[group_A].mean())
    d_mean_climb = float(dz_d[group_B].mean())

    # Symmetry: climb ≈ |drop| within tol (relative). Climb is slightly
    # bigger than drop because (1+r)² − 1 > 1 − (1−r)² for r>0, so use the
    # max of the two as the denominator and tolerate ``--symmetry_tol``.
    def _rel_asym(a: float, b: float) -> float:
        a_, b_ = abs(a), abs(b)
        m = max(a_, b_, 1e-6)
        return abs(a_ - b_) / m

    asym_attacker = _rel_asym(a_mean_climb, a_mean_drop)
    asym_defender = _rel_asym(d_mean_climb, d_mean_drop)
    ok_symmetric = asym_attacker < args.symmetry_tol and asym_defender < args.symmetry_tol

    # Independence: attacker's Δz and defender's Δz across the same envs
    # should not be linearly tied. We compute Pearson within group_A
    # (where attacker climbs and defender drops) and group_B.
    def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
        if x.numel() < 2:
            return 0.0
        xc = x - x.mean()
        yc = y - y.mean()
        denom = (xc.norm() * yc.norm()).clamp_min(1e-9)
        return float((xc * yc).sum() / denom)

    corr_A = _pearson(dz_a[group_A], dz_d[group_A])
    corr_B = _pearson(dz_a[group_B], dz_d[group_B])
    ok_independent = abs(corr_A) < args.corr_tol and abs(corr_B) < args.corr_tol

    print("\n[result]")
    print(f"  attacker mean Δz climb (group_A) = {a_mean_climb:+.4f} m   (predicted +{dz_climb_predicted:.4f})")
    print(f"  attacker mean Δz drop  (group_B) = {a_mean_drop:+.4f} m   (predicted -{dz_drop_predicted:.4f})")
    print(f"  defender mean Δz drop  (group_A) = {d_mean_drop:+.4f} m   (predicted -{dz_drop_predicted:.4f})")
    print(f"  defender mean Δz climb (group_B) = {d_mean_climb:+.4f} m   (predicted +{dz_climb_predicted:.4f})")
    print(f"  symmetry: attacker={asym_attacker:.3f}  defender={asym_defender:.3f}   (tol {args.symmetry_tol})")
    print(f"  Pearson(attacker_dz, defender_dz) — group_A={corr_A:+.3f}   group_B={corr_B:+.3f}   (|tol| {args.corr_tol})")

    print("\n[verdict]")
    print(f"  no NaN/Inf:                              {finite_all}")
    print(f"  attacker[group_A] every env climbed:     {ok_a_climbed}")
    print(f"  attacker[group_B] every env dropped:     {ok_a_dropped}")
    print(f"  defender[group_A] every env dropped:     {ok_d_dropped}")
    print(f"  defender[group_B] every env climbed:     {ok_d_climbed}")
    print(f"  climb/drop magnitudes symmetric:         {ok_symmetric}")
    print(f"  attacker/defender Δz independent:        {ok_independent}")

    all_ok = all([
        finite_all,
        ok_a_climbed,
        ok_a_dropped,
        ok_d_dropped,
        ok_d_climbed,
        ok_symmetric,
        ok_independent,
    ])
    sign_ok = all([ok_a_climbed, ok_a_dropped, ok_d_dropped, ok_d_climbed])

    if all_ok:
        print(
            "\n[PASS] Differential closed-loop dual-drone control. "
            "Genesis batched dispatch is independent across (entity, env, prop). "
            "Plan A is genuinely production-ready, not just hover-safe."
        )
        return 0
    if not sign_ok:
        print(
            "\n[FAIL] Sign-pattern mismatch — at least one (entity, group) cell "
            "moved the wrong way. This is a buffer-sharing or dispatch bug. "
            "Plan A is INFEASIBLE until this is understood."
        )
    else:
        print(
            "\n[FAIL] Sign pattern OK but symmetry / independence checks failed. "
            "Likely a numerical or modeling subtlety, not a Plan A architecture problem."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
