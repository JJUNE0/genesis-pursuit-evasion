"""V1 sanity — Genesis batched dual-drone hover.

Goal (TODO ⚠️ V1):
    Verify that one ``gs.Scene`` can hold two ``Drone`` entities (attacker +
    defender), built with ``n_envs=B``, and that giving each its hover RPM
    keeps both stably at z=1.0 across the whole batch.

Pass criteria
-------------
1.  ``get_pos()`` returns shape ``(B, 3)`` for both drones.
2.  After ``--steps`` simulation steps (default 300 = 3.0 s @ dt=0.01),
    every env has |z - 1.0| < ``--z_tol`` for both drones.
3.  Horizontal drift |xy - spawn_xy| < ``--xy_tol`` for both drones.
4.  No NaN / Inf anywhere in pos / vel.
5.  Selective per-env reset on a subset works (sanity for ``envs_idx`` API).

Failure of any of the above triggers Plan B in TODO.md (split into two
scenes + state synchronizer).

Usage
-----
    python tests/sanity/two_drones_in_one_env.py                # 8192 envs
    python tests/sanity/two_drones_in_one_env.py --num_envs 64  # quick check
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Allow direct invocation without ``pip install -e .``
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs  # noqa: E402

from utils.drone_params import DroneParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V1 sanity: dual-drone batched hover")
    p.add_argument("--num_envs", type=int, default=8192, help="batch size B")
    p.add_argument("--steps", type=int, default=300, help="rollout length (steps); dt=0.01s")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--drone_yaml",
        type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
        help="DroneParams YAML",
    )
    p.add_argument("--z_tol", type=float, default=0.10, help="max |z - 1.0| (m)")
    p.add_argument("--xy_tol", type=float, default=0.20, help="max horizontal drift (m)")
    p.add_argument("--vel_tol", type=float, default=0.50, help="max |vel| (m/s)")
    p.add_argument(
        "--backend",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Genesis backend",
    )
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
    """Best-effort GPU/scene cleanup. Each step is independent and may fail."""
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        gs.destroy()  # Available in some Genesis builds; ignored if absent.
    except Exception:
        pass


def main() -> int:
    args = parse_args()

    # --- Reproducibility (CLAUDE.md §10) ---
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
    except Exception as exc:  # noqa: BLE001 — sanity boundary
        import traceback

        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3
    finally:
        _cleanup()


def _run(args: argparse.Namespace) -> int:
    params = DroneParams(args.drone_yaml)

    B = int(args.num_envs)  # batch size
    T = int(args.steps)
    dt = 0.01  # s — CLAUDE.md §5

    # --- Scene with two Drone entities ---
    n_rendered = max(1, min(int(args.num_rendered_envs), B))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=4),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(args.max_fps),
            # Camera looks at the midpoint between attacker (0,0,1) and defender (1,0,1).
            camera_pos=(3.5, -1.5, 2.5),
            camera_lookat=(0.5, 0.0, 1.0),
            camera_fov=45,
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
    spacing = args.env_spacing if args.env_spacing is not None else (3.0 if args.show_viewer else 0.0)
    scene.build(n_envs=B, env_spacing=(spacing, spacing))
    print(f"[scene] env_spacing={spacing}  rendered_envs={n_rendered}")

    # --- Spawn poses: attacker at (0,0,1), defender at (1,0,1) ---
    spawn_attacker = torch.zeros((B, 3), device=gs.device, dtype=gs.tc_float)
    spawn_attacker[:, 2] = 1.0
    spawn_defender = torch.zeros((B, 3), device=gs.device, dtype=gs.tc_float)
    spawn_defender[:, 0] = 1.0
    spawn_defender[:, 2] = 1.0
    attacker_drone.set_pos(spawn_attacker, zero_velocity=True)
    defender_drone.set_pos(spawn_defender, zero_velocity=True)

    # --- Hover RPM from physics (CLAUDE.md §14: no hardcoded constants) ---
    # rpm_hover = sqrt(m·g / (n_props · k_f))
    n_props = float(params.n_propellers)
    hover_rpm_scalar = float(np.sqrt(params.mass * params.gravity / (n_props * params.kf)))
    hover_rpm_tensor = torch.full(
        (B, params.n_propellers),
        hover_rpm_scalar,
        device=gs.device,
        dtype=gs.tc_float,
    )  # shape: (B, n_props)

    print(f"[info] backend={args.backend} B={B} T={T} dt={dt}")
    print(f"[info] drone='{params.name}' mass={params.mass}kg g={params.gravity} kf={params.kf}")
    print(f"[info] hover RPM ≈ {hover_rpm_scalar:.1f} per propeller")
    print(
        f"[info] spawn: attacker[0]={attacker_drone.get_pos()[0].tolist()}  "
        f"defender[0]={defender_drone.get_pos()[0].tolist()}"
    )

    # --- Roll out T steps with hover RPM on both drones ---
    for _ in range(T):
        attacker_drone.set_propellers_rpm(hover_rpm_tensor)
        defender_drone.set_propellers_rpm(hover_rpm_tensor)
        scene.step()

    pos_attacker = attacker_drone.get_pos()  # shape: (B, 3)
    pos_defender = defender_drone.get_pos()  # shape: (B, 3)
    vel_attacker = attacker_drone.get_vel()  # shape: (B, 3)
    vel_defender = defender_drone.get_vel()  # shape: (B, 3)

    # --- Checks ---
    ok_shape = pos_attacker.shape == (B, 3) and pos_defender.shape == (B, 3) \
        and vel_attacker.shape == (B, 3) and vel_defender.shape == (B, 3)

    finite_all = bool(
        torch.isfinite(pos_attacker).all()
        and torch.isfinite(pos_defender).all()
        and torch.isfinite(vel_attacker).all()
        and torch.isfinite(vel_defender).all()
    )

    z_err_attacker = (pos_attacker[:, 2] - 1.0).abs()  # shape: (B,)
    z_err_defender = (pos_defender[:, 2] - 1.0).abs()  # shape: (B,)
    xy_drift_attacker = (pos_attacker[:, :2] - spawn_attacker[:, :2]).norm(dim=1)  # shape: (B,)
    xy_drift_defender = (pos_defender[:, :2] - spawn_defender[:, :2]).norm(dim=1)  # shape: (B,)
    speed_attacker = vel_attacker.norm(dim=1)  # shape: (B,)
    speed_defender = vel_defender.norm(dim=1)  # shape: (B,)

    z_max = float(torch.maximum(z_err_attacker.max(), z_err_defender.max()))
    xy_max = float(torch.maximum(xy_drift_attacker.max(), xy_drift_defender.max()))
    v_max = float(torch.maximum(speed_attacker.max(), speed_defender.max()))

    ok_z = z_max < args.z_tol
    ok_xy = xy_max < args.xy_tol
    ok_vel = v_max < args.vel_tol

    # --- Cross-entity isolation check — the *real* Plan A go/no-go gate ---
    #
    # The original block only asked "did the teleport land?", which is the
    # weakest possible question. The actual Plan A risk is that
    # ``set_pos(envs_idx=subset)`` on the attacker silently corrupts the
    # **defender** state in the same env slots, or other attacker env slots,
    # because both share the same scene-level buffer. If that leaks, the
    # whole dual-drone-in-one-scene architecture is broken regardless of
    # whether hover looks fine.
    #
    # We snapshot defender(subset) and attacker(non_subset) BEFORE teleport,
    # then verify neither moved more than ``isolation_tol`` after one step.
    if B >= 2:
        subset = torch.tensor([0, 1], device=gs.device, dtype=torch.int32)
        # Long indexer for `[non_subset]` slicing; 0/1 are reserved for the subset
        non_subset = torch.arange(2, B, device=gs.device, dtype=torch.long)

        # Pre-snapshots (the entities we want PROTECTED from the teleport)
        defender_pos_before = defender_drone.get_pos()[subset].clone()      # shape: (2, 3)
        defender_vel_before = defender_drone.get_vel()[subset].clone()      # shape: (2, 3)
        attacker_pos_before_other = attacker_drone.get_pos()[non_subset].clone() \
            if non_subset.numel() > 0 else None

        # Teleport ONLY attacker[subset] up to z=5
        teleport = torch.tensor(
            [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        attacker_drone.set_pos(teleport, zero_velocity=True, envs_idx=subset)
        attacker_drone.set_propellers_rpm(hover_rpm_tensor)
        defender_drone.set_propellers_rpm(hover_rpm_tensor)
        scene.step()

        attacker_pos_after = attacker_drone.get_pos()
        defender_pos_after = defender_drone.get_pos()
        defender_vel_after = defender_drone.get_vel()

        isolation_tol = 0.05  # m / m·s⁻¹  — anything above this is genuine corruption

        # 1) Subset teleport actually took effect (sanity, not isolation).
        teleport_z_min = float(attacker_pos_after[subset, 2].min())
        ok_subset_teleport = teleport_z_min > 4.95  # 1 step of hover ≈ 0 fall

        # 2) Defender at the same env slots untouched.
        defender_subset_drift = float(
            (defender_pos_after[subset] - defender_pos_before).norm(dim=1).max()
        )
        defender_subset_vel_drift = float(
            (defender_vel_after[subset] - defender_vel_before).norm(dim=1).max()
        )
        ok_defender_isolated = defender_subset_drift < isolation_tol
        ok_defender_vel_isolated = defender_subset_vel_drift < isolation_tol

        # 3) Attacker's OTHER envs not perturbed by the partial write.
        if attacker_pos_before_other is not None:
            attacker_other_drift = float(
                (attacker_pos_after[non_subset] - attacker_pos_before_other).norm(dim=1).max()
            )
            ok_attacker_other_envs = attacker_other_drift < isolation_tol
        else:
            attacker_other_drift = 0.0
            ok_attacker_other_envs = True
    else:
        ok_subset_teleport = True
        ok_defender_isolated = True
        ok_defender_vel_isolated = True
        ok_attacker_other_envs = True
        teleport_z_min = float("nan")
        defender_subset_drift = 0.0
        defender_subset_vel_drift = 0.0
        attacker_other_drift = 0.0

    # --- Report ---
    print("\n[result] after T steps:")
    print(f"  attacker pos[0]    = {pos_attacker[0].tolist()}  vel[0] = {vel_attacker[0].tolist()}")
    print(f"  defender pos[0]    = {pos_defender[0].tolist()}  vel[0] = {vel_defender[0].tolist()}")
    print(f"  max |z-1.0|        = {z_max:.4f} m   (tol {args.z_tol})")
    print(f"  max horizontal err = {xy_max:.4f} m   (tol {args.xy_tol})")
    print(f"  max |vel|          = {v_max:.4f} m/s (tol {args.vel_tol})")

    print("\n[isolation check] subset=[0,1], non_subset=[2..B-1]")
    print(f"  attacker[subset] teleport z_min   = {teleport_z_min:.4f}  (need > 4.95)")
    print(f"  defender[subset] pos drift  max   = {defender_subset_drift:.5f} m")
    print(f"  defender[subset] vel drift  max   = {defender_subset_vel_drift:.5f} m/s")
    print(f"  attacker[non_subset] pos drift max = {attacker_other_drift:.5f} m")

    print("\n[verdict]")
    print(f"  shape (B,3) for both:                 {ok_shape}")
    print(f"  no NaN/Inf:                           {finite_all}")
    print(f"  z stable at 1.0:                      {ok_z}")
    print(f"  horizontal drift bounded:             {ok_xy}")
    print(f"  velocity bounded:                     {ok_vel}")
    print(f"  attacker[subset] teleport landed:     {ok_subset_teleport}")
    print(f"  defender[subset] pos NOT corrupted:   {ok_defender_isolated}")
    print(f"  defender[subset] vel NOT corrupted:   {ok_defender_vel_isolated}")
    print(f"  attacker[non_subset] NOT perturbed:   {ok_attacker_other_envs}")

    all_ok = all([
        ok_shape,
        finite_all,
        ok_z,
        ok_xy,
        ok_vel,
        ok_subset_teleport,
        ok_defender_isolated,
        ok_defender_vel_isolated,
        ok_attacker_other_envs,
    ])
    isolation_ok = all([
        ok_subset_teleport,
        ok_defender_isolated,
        ok_defender_vel_isolated,
        ok_attacker_other_envs,
    ])
    if all_ok:
        print(
            "\n[PASS] V1 — Genesis supports batched dual-drone hover AND "
            "envs_idx writes do not leak across entities or env slots. "
            "Phase 1 plan A is feasible."
        )
        return 0
    if not isolation_ok:
        print(
            "\n[FAIL] V1 — envs_idx isolation BROKEN. Plan A is INFEASIBLE: "
            "partial writes corrupt the other entity or other env slots in "
            "the same scene buffer. Switch to Plan B (split scenes + state "
            "synchronizer)."
        )
    else:
        print(
            "\n[FAIL] V1 — hover-stability check failed but isolation is OK. "
            "This is a tuning / numerical issue (kf, substeps, tolerances), "
            "NOT a Plan A architecture problem."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
