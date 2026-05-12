"""scripts/diagnose_hover.py — PR-J 진단 스크립트.

BodyRateController에 action=[0,0,0,0] (hover 명령)을 5초간 입력하고
z, ang_vel, rpm trajectory를 측정. PR-I 튜닝이 자유낙하 중 측정인지,
정말 hover 안정한 상태에서 step response 보였는지 확인.

또한 mixer matrix sanity check (build_mixer_matrix가 quad 형상에 맞는지).

Usage
-----
    python scripts/diagnose_hover.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs                                # noqa: E402

from utils.drone_params import DroneParams          # noqa: E402
from utils.rate_controller import (                 # noqa: E402
    BodyRateController, build_mixer_matrix,
)


def main() -> int:
    gs.init(backend=gs.cpu, seed=42, logging_level="warning")
    try:
        return _run()
    finally:
        try:
            gs.destroy()
        except Exception:
            pass


def _run() -> int:
    params = DroneParams(str(ROOT / "configs" / "drones" / "nova.yaml"))
    dt = 0.005
    T_total_seconds = 5.0
    n_step = int(T_total_seconds / dt)
    B = 1

    # ===== 1-3. Mixer sanity check =====
    print("=" * 70)
    print("[mixer sanity check]")
    print("=" * 70)
    M = build_mixer_matrix(params)
    M_inv = np.linalg.inv(M) if M.shape[0] == M.shape[1] else np.linalg.pinv(M)
    print(f"M shape: {M.shape}")
    print(f"M =\n{M}")
    print(f"M_inv =\n{M_inv}")
    hover_wrench = np.array([params.mass * params.gravity, 0.0, 0.0, 0.0])
    thrusts_at_hover = M_inv @ hover_wrench
    expected_per_motor = params.mass * params.gravity / params.n_propellers
    print(f"\nhover wrench = {hover_wrench}")
    print(f"thrusts_at_hover (per motor) = {thrusts_at_hover}")
    print(f"expected per motor (m·g/n) = {expected_per_motor:.6f}")
    max_dev = float(np.abs(thrusts_at_hover - expected_per_motor).max())
    print(f"max deviation from uniform: {max_dev:.6e}")
    hover_rpm_from_mixer = float(np.sqrt(max(thrusts_at_hover[0], 1e-9) / params.kf))
    hover_rpm_formula = float(np.sqrt(expected_per_motor / params.kf))
    print(f"hover_rpm (from mixer thrust)   = {hover_rpm_from_mixer:.2f}")
    print(f"hover_rpm (formula √(m·g/(n·kf))) = {hover_rpm_formula:.2f}")
    print(f"prop_positions =\n{params.prop_positions}")
    print(f"spin_directions = {params.spin_directions.tolist()}")

    # ===== Build scene =====
    print("\n" + "=" * 70)
    print("[build scene]")
    print("=" * 70)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=16),
        rigid_options=gs.options.RigidOptions(
            dt=dt, constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True, enable_joint_limit=True,
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

    spawn = torch.zeros((B, 3), device=gs.device, dtype=gs.tc_float)
    spawn[:, 2] = 1.0
    drone.set_pos(spawn, zero_velocity=True)
    identity_q = torch.zeros((B, 4), device=gs.device, dtype=gs.tc_float)
    identity_q[:, 0] = 1.0
    drone.set_quat(identity_q, zero_velocity=True)

    # ===== Test A: BodyRateController with hover action =====
    print("\n" + "=" * 70)
    print("[Test A] BodyRateController, action = [0,0,0,0] for 5s")
    print("=" * 70)
    ctrl = BodyRateController(
        num_envs=B, dt=dt, params=params, device=gs.device,
        kp_rate=(0.05, 0.10, 0.10),
        kd_rate=(1.0e-4, 5.0e-5, 1.0e-4),
    )
    print(f"hover_thrust_total = {ctrl.hover_thrust_total:.4f} N")
    print(f"max_thrust_per_motor = {ctrl.max_thrust_per_motor:.4f} N")
    print(f"max_rpm = {ctrl.max_rpm:.1f}")

    a_zero = torch.zeros((B, 4), device=gs.device, dtype=gs.tc_float)
    z_hist = np.zeros(n_step)
    ang_hist = np.zeros((n_step, 3))
    rpm_hist = np.zeros((n_step, 4))
    for k in range(n_step):
        ang = drone.get_ang()
        rpms = ctrl.step(a_zero, ang)
        drone.set_propellers_rpm(rpms)
        scene.step()
        pos = drone.get_pos()
        z_hist[k] = float(pos[0, 2])
        ang_hist[k] = drone.get_ang()[0].cpu().numpy()
        rpm_hist[k] = rpms[0].cpu().numpy()

        if k < 10 or k in (50, 100, 200, 500, 999):
            print(
                f"  k={k:4d} t={(k+1)*dt:.3f}s  z={z_hist[k]:.4f}m  "
                f"ang_vel={ang_hist[k]}  rpm={rpm_hist[k]}"
            )

    print(f"\n  z_initial = 1.0000m, z_final = {z_hist[-1]:.4f}m")
    print(f"  z_drop over 5s = {1.0 - z_hist[-1]:.4f}m")
    print(f"  max |ang_vel| = {np.abs(ang_hist).max():.4f} rad/s")
    z_drop_03 = 1.0 - z_hist[int(0.3/dt) - 1]
    print(f"  z_drop @ 0.3s (PR-I metric window) = {z_drop_03:.4f}m")
    z_drop_10 = 1.0 - z_hist[int(1.0/dt) - 1]
    print(f"  z_drop @ 1.0s (full step phase) = {z_drop_10:.4f}m")

    # ===== Test B: 직접 hover_rpm 명령 (BodyRateController bypass) =====
    drone.set_pos(spawn, zero_velocity=True)
    drone.set_quat(identity_q, zero_velocity=True)
    print("\n" + "=" * 70)
    print("[Test B] Direct hover_rpm (bypass controller) for 5s")
    print("=" * 70)
    hover_rpm = float(np.sqrt(params.mass * params.gravity / (params.n_propellers * params.kf)))
    print(f"hover_rpm (formula) = {hover_rpm:.2f}")
    rpm_const = torch.full((B, 4), hover_rpm, device=gs.device, dtype=gs.tc_float)
    z_hist_B = np.zeros(n_step)
    for k in range(n_step):
        drone.set_propellers_rpm(rpm_const)
        scene.step()
        pos = drone.get_pos()
        z_hist_B[k] = float(pos[0, 2])
        if k < 5 or k in (100, 500, 999):
            print(f"  k={k:4d} t={(k+1)*dt:.3f}s  z={z_hist_B[k]:.4f}m")
    print(f"\n  z_drop over 5s (Test B, direct RPM) = {1.0 - z_hist_B[-1]:.4f}m")

    # ===== Compare =====
    print("\n" + "=" * 70)
    print("[Verdict]")
    print("=" * 70)
    z_drop_A = 1.0 - z_hist[-1]
    z_drop_B = 1.0 - z_hist_B[-1]
    print(f"BodyRateController hover (Test A): z_drop = {z_drop_A:.4f}m  ang_vel_max = {np.abs(ang_hist).max():.4f}")
    print(f"Direct hover_rpm     (Test B): z_drop = {z_drop_B:.4f}m")
    if abs(z_drop_A - z_drop_B) < 0.01 and abs(z_drop_A) < 0.05:
        print("→ 시나리오: HOVER OK. 두 방식 모두 안정. PR-I 튜닝 valid.")
    elif abs(z_drop_B) < 0.05 and abs(z_drop_A) > 0.1:
        print("→ 시나리오 D: Direct RPM은 호버 OK, Controller는 실패. Controller 버그.")
    elif abs(z_drop_A) > 0.1 and abs(z_drop_B) > 0.1:
        print("→ 시나리오 A/B: Direct RPM도 실패. Mixer/단위 문제 또는 Genesis sim issue.")
    else:
        print("→ 모호. 데이터 더 필요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
