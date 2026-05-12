"""Minimal prop visualization diagnostic — set RPM 직접 + viewer.

학습 정책 무관, drone에 상수 RPM 명령 + scene.step만 호출.
RPM을 1/10로 낮춰 viewer FPS 한계인지 batched mode 한계인지 판별.

Usage:
  # 1. Hover RPM (default 8014) — 우리 PE env와 동일 빠르기
  .venv/bin/python scripts/test/diag_prop_visual.py
  # 2. 1/10 RPM (800) — 시각화 한계 검증
  .venv/bin/python scripts/test/diag_prop_visual.py --rpm 800
  # 3. 매우 천천히 (RPM 80) — 확실히 회전 보여야 정상
  .venv/bin/python scripts/test/diag_prop_visual.py --rpm 80
  # 4. batched mode (n_envs=4) — batched가 prop 시각화 한계인지 검증
  .venv/bin/python scripts/test/diag_prop_visual.py --rpm 800 --num_envs 4
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs
from utils.drone_params import DroneParams


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rpm", type=float, default=8014.0,
                   help="constant RPM applied to all 4 motors (hover ≈ 8014)")
    p.add_argument("--num_envs", type=int, default=1,
                   help="0 = single env (unbatched), 1+ = batched")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--substeps", type=int, default=16)
    p.add_argument("--no_viewer", action="store_true")
    p.add_argument("--pin_pos", action="store_true",
                   help="pin drone at spawn pos every step (prevents gravity fall)")
    args = p.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    params = DroneParams(str(ROOT / "configs/drones/nova.yaml"))

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt, substeps=args.substeps),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=50,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(show_world_frame=True),
        show_viewer=not args.no_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        gs.morphs.Drone(
            file=params.urdf_path,
            pos=(0.0, 0.0, 0.5),
            propellers_link_name=params.prop_link_names,
            propellers_spin=[int(s) for s in params.spin_directions],
        )
    )
    scene.build(n_envs=args.num_envs)

    # Determine batch size for RPM tensor
    B = max(args.num_envs, 1)
    rpms = torch.full((B, 4), args.rpm, dtype=torch.float32, device=gs.device)

    if args.pin_pos:
        if args.num_envs > 0:
            spawn_pos = torch.tensor([[0.0, 0.0, 0.5]] * B, device=gs.device)
            envs_idx = torch.arange(B, device=gs.device)
        else:
            spawn_pos = torch.tensor([[0.0, 0.0, 0.5]], device=gs.device)
            envs_idx = None

    rev_per_frame = args.rpm / 60.0 * (1.0 / 60.0)   # 60 RPM → 1 rev/sec → 1/60 rev/frame
    print(f"[diag] RPM={args.rpm}  num_envs={args.num_envs}  dt={args.dt}  substeps={args.substeps}")
    print(f"[diag] viewer FPS=60 → expected revs/frame ≈ {rev_per_frame:.4f}")
    print(f"[diag] revs/frame interpretation:")
    print(f"         ≥ 0.5 → 시각적 blur 또는 random rotation (hover 8014은 2.23)")
    print(f"         0.05-0.5 → 회전 슬쩍 보일 수 있음 (RPM 200-2000)")
    print(f"         < 0.05 → 명확히 회전 보여야 (RPM < 200)")
    print(f"[diag] running {args.steps} steps")

    for k in range(args.steps):
        if args.pin_pos:
            drone.set_pos(spawn_pos, zero_velocity=True, envs_idx=envs_idx)
        drone.set_propellers_rpm(rpms)
        scene.step()
        if (k + 1) % 200 == 0:
            print(f"  step {k+1}/{args.steps}")
    print("[diag] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
