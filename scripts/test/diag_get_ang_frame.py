"""Smoking-gun test — Genesis ``drone.get_ang()`` returns body or world frame?

Setup:
  1. Spawn drone at high altitude, identity quat.
  2. Force 90° pitch via ``set_quat`` (nose down, body Z = world -X).
  3. Apply motor pattern that produces a known SINGLE-AXIS body torque.
  4. step() once → drone has nonzero ang_vel.
  5. Read raw ``drone.get_ang()`` + compute ``R^T · get_ang()``.
  6. Compare with the analytical expectation per frame hypothesis.

Decision logic — after motor pattern produces a body-axis torque:
  - If raw ang_vel ALONG body axis (i.e. raw and R^T·raw differ when quat≠I):
    → raw is WORLD frame  → pe_1v1 ``rotate_to_body_frame`` is CORRECT.
  - If raw ang_vel ALONG body axis WITHOUT applying R^T:
    → raw is BODY frame → pe_1v1 ``rotate_to_body_frame`` is **DOUBLE ROTATION** (BUG).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs   # noqa: E402

from envs.base_pe_env import quat_to_rotmat_flat, rotate_to_body_frame  # noqa: E402


def main() -> int:
    gs.init(backend=gs.cpu, seed=0, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, substeps=16),
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.005,
            enable_collision=False,
            enable_joint_limit=False,
        ),
    )
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        gs.morphs.Drone(file="urdf/drones/cf2x.urdf")
    )
    scene.build(n_envs=1)
    dev = gs.device

    # ------------------------------------------------------------------
    # Test 1 — identity quat, asymmetric motors → measure raw get_ang().
    # ------------------------------------------------------------------
    print("\n==== Test 1: identity quat (R = I; world == body) ====")
    drone.set_pos(torch.tensor([[0.0, 0.0, 5.0]], device=dev),
                  zero_velocity=True, envs_idx=torch.tensor([0], device=dev))
    drone.set_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev),
                   zero_velocity=True, envs_idx=torch.tensor([0], device=dev))
    # Asymmetric RPM — motor 0,2 high, 1,3 low (typical X-quad: 0,2 cw, 1,3 ccw
    # → unequal spins → net yaw torque + maybe roll). Magnitude small to keep
    # rotation < 90° in 1 step.
    hover_rpm = math.sqrt(0.8013 * 9.81 / 4 / 3.06e-8)  # ≈ 8014
    rpms_yaw = torch.tensor(
        [[hover_rpm * 1.4, hover_rpm * 0.6, hover_rpm * 1.4, hover_rpm * 0.6]],
        device=dev, dtype=gs.tc_float,
    )
    drone.set_propellers_rpm(rpms_yaw)
    for _ in range(5):
        scene.step()
    q1 = drone.get_quat().clone()                      # (1, 4)
    raw_ang_1 = drone.get_ang().clone()                # (1, 3)
    body_ang_1 = rotate_to_body_frame(raw_ang_1, q1)
    print(f"  quat (w,x,y,z): {q1[0].tolist()}")
    print(f"  raw get_ang() : {raw_ang_1[0].tolist()}")
    print(f"  R^T · raw     : {body_ang_1[0].tolist()}")
    print(f"  ‖raw − body‖ : {(raw_ang_1 - body_ang_1).abs().max().item():.6f}")
    print("  (At identity quat both should agree — sanity check only.)")

    # ------------------------------------------------------------------
    # Test 2 — 90° pitch quat, then asymmetric motors. R ≠ I.
    # ------------------------------------------------------------------
    print("\n==== Test 2: 90° pitch (R ≠ I) — frame discriminator ====")
    drone.set_pos(torch.tensor([[0.0, 0.0, 5.0]], device=dev),
                  zero_velocity=True, envs_idx=torch.tensor([0], device=dev))
    # 90° pitch about Y: q = (cos45°, 0, sin45°, 0)
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    pitch90 = torch.tensor([[c, 0.0, s, 0.0]], device=dev, dtype=gs.tc_float)
    drone.set_quat(pitch90, zero_velocity=True, envs_idx=torch.tensor([0], device=dev))
    drone.set_propellers_rpm(rpms_yaw)
    for _ in range(5):
        scene.step()
    q2 = drone.get_quat().clone()
    raw_ang_2 = drone.get_ang().clone()
    body_ang_2 = rotate_to_body_frame(raw_ang_2, q2)
    print(f"  quat (w,x,y,z): {q2[0].tolist()}")
    print(f"  raw get_ang() : {raw_ang_2[0].tolist()}")
    print(f"  R^T · raw     : {body_ang_2[0].tolist()}")
    diff = (raw_ang_2 - body_ang_2).abs().max().item()
    print(f"  ‖raw − body‖ : {diff:.6f}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n==== VERDICT ====")
    # The motor pattern generates body-axis torque (motor mixer is body-fixed).
    # In Test 1 (R=I) the resulting ω is the same in either frame.
    # In Test 2 (90° pitch) body and world differ. The body Z axis points
    # along world -X. So a body-yaw rotation (about body Z) appears in:
    #   - body frame as (0, 0, ω_body)
    #   - world frame as (-ω, 0, 0)
    # (similarly body-roll appears world as (0, 0, ω) etc.)
    raw = raw_ang_2[0]
    body = body_ang_2[0]
    print(f"  raw component max |xyz|: x={raw[0].abs().item():.3f} "
          f"y={raw[1].abs().item():.3f} z={raw[2].abs().item():.3f}")
    print(f"  body component max |xyz|: x={body[0].abs().item():.3f} "
          f"y={body[1].abs().item():.3f} z={body[2].abs().item():.3f}")
    print(f"  diff norm = {diff:.6f}")
    print()
    if diff < 1e-4:
        print("  ⚠ raw ≈ body even at 90° pitch.")
        print("  → Genesis ``get_ang()`` returns BODY-FRAME values.")
        print("  → pe_1v1 base_pe_env.py:562-563 ``rotate_to_body_frame`` "
              "is DOUBLE ROTATION (BUG).")
        print("  → hover env (no transform) is correct; pe_1v1 must drop the transform.")
    else:
        print(f"  raw ≠ body (diff {diff:.3f}).")
        print("  → Genesis ``get_ang()`` returns WORLD-FRAME values.")
        print("  → pe_1v1 ``rotate_to_body_frame`` is CORRECT.")
        print("  → hover env (no transform) might have been incidentally lucky")
        print("    (or hover never tilts much and z<floor crash hides errors).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
