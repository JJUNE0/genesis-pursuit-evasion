"""PR-N — wrench saturation handler effectiveness.

Verifies:
  enable=False — bang-bang loses ~65% wrench (regression of pre-fix state).
  enable=True  — bang-bang wrench_loss = 0%, motor thrusts in [0, max].
  Hover sanity (action=0): both modes give 4 motors at m·g/4.

Wall-clock < 1 s. Pure-numerical, no Genesis sim.
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

from utils.drone_params import DroneParams   # noqa: E402
from utils.rate_controller import (   # noqa: E402
    BodyRateController, build_mixer_matrix,
)


def main() -> int:
    gs.init(backend=gs.cpu, seed=0, logging_level="warning")
    params = DroneParams(str(ROOT / "configs" / "drones" / "nova.yaml"))

    M = build_mixer_matrix(params)
    M_t = torch.tensor(M, dtype=torch.float32)   # (4, n_props)

    cases = [
        ("hover",     [ 0.0,  0.0,  0.0,  0.0]),
        ("bang_+all", [+1.0, +1.0, +1.0, +1.0]),
        ("bang_-all", [-1.0, -1.0, -1.0, -1.0]),
        ("bang_T0",   [-1.0, +1.0, +1.0, +1.0]),
        ("bang_yaw",  [ 0.0,  0.0,  0.0, +1.0]),
    ]

    for enable in (False, True):
        ctrl = BodyRateController(
            num_envs=1, dt=0.005, params=params, device=torch.device("cpu"),
            kp_rate=(0.05, 0.10, 0.10), kd_rate=(1e-4, 5e-5, 1e-4),
            max_body_rate=math.pi, enable_wrench_saturation=enable,
        )
        print(f"\n=== enable_wrench_saturation = {enable} ===")
        print(f"{'case':>10s} | {'T_cmd':>6s} {'|τ|_cmd':>8s} "
              f"{'|τ|_act':>8s} {'loss%':>6s} | "
              f"{'sat_max':>7s} {'sat_zero':>8s} | rpms")
        print("-" * 96)
        for name, action in cases:
            ctrl.reset()
            action_t = torch.tensor([action], dtype=torch.float32)
            ang_vel = torch.zeros((1, 3))
            T_norm = action_t[:, 0:1]
            rate_ref = action_t[:, 1:4] * ctrl.max_body_rate
            T_total = (T_norm + 1.0) * ctrl.hover_thrust_total
            torque_cmd = ctrl.kp * (rate_ref - ang_vel)   # ang_acc=0 (last_ang_vel=0)
            rpms = ctrl.step(action_t, ang_vel)
            thrusts_act = (rpms ** 2) * ctrl.kf                                     # (1, n)
            wrench_act = thrusts_act @ M_t.T                                        # (1, 4)
            torque_act = wrench_act[:, 1:]
            tcmd_n = torque_cmd.norm(dim=1).clamp_min(1e-9)
            tact_n = torque_act.norm(dim=1)
            loss = (1.0 - tact_n / tcmd_n).clamp(0, 1).item() if torque_cmd.norm() > 1e-6 else 0.0
            sat_max = (rpms >= ctrl.max_rpm - 1.0).float().mean().item()
            sat_zero = (rpms <= 1.0).float().mean().item()
            rpms_s = "[" + ", ".join(f"{x:.0f}" for x in rpms[0].tolist()) + "]"
            print(f"{name:>10s} | {T_total.item():>6.2f} {torque_cmd.norm().item():>8.4f} "
                  f"{tact_n.item():>8.4f} {loss*100:>5.1f}% | "
                  f"{sat_max:>7.2f} {sat_zero:>8.2f} | {rpms_s}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
