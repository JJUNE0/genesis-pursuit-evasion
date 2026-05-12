"""Live-policy RPM trace — viewer에서 propeller 안 도는 현상 진단.

eval.py와 동일 setup (c14 head-to-head, Phase C, 4 envs)으로 1 episode
rollout. 매 step:
  - attacker / defender action (4D, [-1,1])
  - rate_ctrl 출력 RPM (4 motors)
  - thrust per motor (N)
  - body ang_vel norm

판정 기준:
  - 정상 chase: RPM std > 500, sat_zero/max < 0.2, |ω| 1-5 rad/s
  - hover-only: RPM ≈ 8014 ± 100 정적, |ω| < 0.3
  - bang-bang: sat_zero/max > 0.4
  - controller dead: RPM std < 50 (정책이 변동 명령 못 함)

Wall-clock < 5s.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs   # noqa: E402

from envs.pe_1v1.env import PursuitEvasion1v1Env   # noqa: E402
from envs.pe_1v1.scripted_attacker import PretrainedAttacker   # noqa: E402
from envs.pe_1v1.scripted_defender import PretrainedDefender   # noqa: E402
from utils.drone_params import DroneParams   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attacker_ckpt", required=True)
    p.add_argument("--defender_ckpt", required=True)
    p.add_argument("--cfgs_pkl", required=True)
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample from policy distribution (μ + σ·ε) instead of mean.")
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")

    with open(args.cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)
    env_cfg = dict(cfgs["env_cfg"])
    env_cfg["ego"] = "attacker"

    rcfg = env_cfg.get("rate_controller", {}) or {}
    print(f"[diag] enable_wrench_saturation = {rcfg.get('enable_wrench_saturation', False)}")
    print(f"[diag] max_body_rate = {rcfg.get('max_body_rate', 'default π')}")

    params = DroneParams(cfgs.get("drone_yaml", str(ROOT / "configs/drones/nova.yaml")))
    det = not args.stochastic
    print(f"[diag] policy mode = {'deterministic (mean only)' if det else 'STOCHASTIC (μ + σ·ε)'}")
    defender = PretrainedDefender(num_envs=args.num_envs, ckpt_path=args.defender_ckpt, device=gs.device, deterministic=det)
    attacker = PretrainedAttacker(num_envs=args.num_envs, ckpt_path=args.attacker_ckpt, device=gs.device, deterministic=det)
    env = PursuitEvasion1v1Env(
        num_envs=args.num_envs, params=params,
        env_cfg=env_cfg, obs_cfg=cfgs["obs_cfg"],
        reward_cfg=cfgs["reward_cfg"], command_cfg=cfgs["command_cfg"],
        defender_policy=defender, show_viewer=False,
    )

    a_ctrl = env._attacker_rate_ctrl
    d_ctrl = env._defender_rate_ctrl
    print(f"[diag] hover RPM = {(8.013e-1*9.81/4/a_ctrl.kf)**0.5:.0f}, max RPM = {a_ctrl.max_rpm:.0f}")
    print()

    env.reset()
    log = defaultdict(list)
    t0 = time.time()
    print(f"{'step':>4} | {'a_act|':>8} {'d_act|':>8} | "
          f"{'a_rpm_μ':>8} {'a_rpm_σ':>8} {'a_sat0':>6} {'a_satM':>6} | "
          f"{'d_rpm_μ':>8} {'d_rpm_σ':>8} {'d_sat0':>6} {'d_satM':>6} | "
          f"{'a_|ω|':>5} {'d_|ω|':>5}")
    for step in range(args.max_steps):
        a_act = attacker.step(
            ego_state=env._attacker_state_dict(),
            opponent_state=env._defender_state_dict(),
            g_mission=env.g_mission,
        )
        # snapshot ang_vel BEFORE step (controller uses pre-step ang_vel)
        a_av_pre = env.attacker_ang_vel.detach().clone()
        d_av_pre = env.defender_ang_vel.detach().clone()

        # External attacker drives ego, defender internal — env step computes both RPMs.
        _, _, done, _ = env.step(a_act)

        # Recompute the RPMs the controllers produced (state-pure).
        # Note: env.step has already advanced last_ang_vel; for diagnostic we
        # re-run controller.step on a clone. Cheap.
        a_clone = a_ctrl.last_ang_vel.detach().clone()
        d_clone = d_ctrl.last_ang_vel.detach().clone()
        # Defender action: env stored it.
        d_act = env.last_defender_action.detach()
        a_rpms = a_ctrl.step(a_act, a_av_pre)   # mutates last_ang_vel
        a_ctrl.last_ang_vel[:] = a_clone        # restore
        d_rpms = d_ctrl.step(d_act, d_av_pre)
        d_ctrl.last_ang_vel[:] = d_clone

        a_sat0 = (a_rpms <= 1.0).float().mean().item()
        a_satM = (a_rpms >= a_ctrl.max_rpm - 1.0).float().mean().item()
        d_sat0 = (d_rpms <= 1.0).float().mean().item()
        d_satM = (d_rpms >= d_ctrl.max_rpm - 1.0).float().mean().item()
        a_w = env.attacker_ang_vel.norm(dim=1).mean().item()
        d_w = env.defender_ang_vel.norm(dim=1).mean().item()

        log["a_rpm_mean"].append(float(a_rpms.mean().item()))
        log["a_rpm_std"].append(float(a_rpms.std().item()))
        log["d_rpm_mean"].append(float(d_rpms.mean().item()))
        log["d_rpm_std"].append(float(d_rpms.std().item()))
        log["a_sat0"].append(a_sat0); log["a_satM"].append(a_satM)
        log["d_sat0"].append(d_sat0); log["d_satM"].append(d_satM)
        log["a_omega"].append(a_w); log["d_omega"].append(d_w)
        log["a_act_max"].append(float(a_act.abs().max().item()))
        log["d_act_max"].append(float(d_act.abs().max().item()))

        if step % args.print_every == 0:
            print(f"{step:>4} | "
                  f"{a_act.abs().max().item():>7.2f} {d_act.abs().max().item():>7.2f} | "
                  f"{a_rpms.mean().item():>8.0f} {a_rpms.std().item():>8.0f} "
                  f"{a_sat0:>5.2f} {a_satM:>5.2f} | "
                  f"{d_rpms.mean().item():>8.0f} {d_rpms.std().item():>8.0f} "
                  f"{d_sat0:>5.2f} {d_satM:>5.2f} | "
                  f"{a_w:>5.2f} {d_w:>5.2f}")
        if done.all():
            print(f"  (all envs done at step {step})")
            break

    print()
    print("=== Aggregate (across all steps, all envs) ===")
    for who in ("a", "d"):
        rpm_m = np.array(log[f"{who}_rpm_mean"])
        rpm_s = np.array(log[f"{who}_rpm_std"])
        sat0 = np.array(log[f"{who}_sat0"])
        satM = np.array(log[f"{who}_satM"])
        omega = np.array(log[f"{who}_omega"])
        amax = np.array(log[f"{who}_act_max"])
        name = "attacker" if who == "a" else "defender"
        print(f"\n{name}:")
        print(f"  RPM mean across-step:  μ={rpm_m.mean():.0f}  range=[{rpm_m.min():.0f}, {rpm_m.max():.0f}]")
        print(f"  RPM std  per-step:     median={np.median(rpm_s):.0f}  p99={np.percentile(rpm_s,99):.0f}")
        print(f"  sat_zero (RPM≈0):      mean={sat0.mean():.3f}  max={sat0.max():.3f}")
        print(f"  sat_max  (RPM≈cap):    mean={satM.mean():.3f}  max={satM.max():.3f}")
        print(f"  |ω|:                   mean={omega.mean():.2f}  p99={np.percentile(omega,99):.2f}")
        print(f"  |action|_max per step: mean={amax.mean():.2f}  saturation_frac={(amax>0.95).mean():.3f}")

    # Verdict
    a_rpm_s = np.array(log["a_rpm_std"])
    d_rpm_s = np.array(log["d_rpm_std"])
    print(f"\n=== Verdict ===")
    if np.median(a_rpm_s) < 50 and np.median(d_rpm_s) < 50:
        print("  ⚠ RPM 변동 거의 없음 (std median < 50) — 두 정책 모두 hover-only.")
        print("    학습이 chase/evade를 못 했거나, controller가 토크 명령을 죽이고 있음.")
    elif np.median(a_rpm_s) < 200 or np.median(d_rpm_s) < 200:
        print("  ~ RPM 변동 작음 — 학습 부분적, viewer에서 회전 식별 어려움.")
    else:
        print("  ✓ RPM 변동 정상 — viewer에서 회전 안 보이는 건 Genesis prop mesh 시각화 문제.")

    print(f"\nwall_sec = {time.time()-t0:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
