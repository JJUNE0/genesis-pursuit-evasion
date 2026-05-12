"""Phase 1 deploy-mode eval — reconstruct env from cfgs.pkl, roll out a frozen
actor, report winrate / mission / capture / crash / oob / timeout breakdown.

CLAUDE.md §7 / §10:

- Eval reads ``cfgs.pkl`` only; never depends on default values that may have
  drifted in the source tree.
- Phase 1's env builds ``obs = {"policy": ...}`` exclusively — no
  ``critic_priv`` key. Eval verifies this contract at startup so accidental
  Phase 4 leakage shows up here, not in production.

Usage::

    # attacker eval against the same defender it was trained against
    python scripts/eval.py \\
        --cfgs_pkl logs/pe1v1_v0/attacker_v0/<run>/cfgs.pkl \\
        --actor_ckpt logs/pe1v1_v0/attacker_v0/<run>/actor_v0.pt \\
        --defender stationary \\
        --num_envs 256 --episodes 4 --seed 0
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs   # noqa: E402

from envs.pe_1v1.env import PursuitEvasion1v1Env   # noqa: E402
from envs.pe_1v1.scripted_attacker import (   # noqa: E402
    RandomWalkAttacker,
    StationaryAttacker,
)
from envs.pe_1v1.scripted_defender import (   # noqa: E402
    PretrainedDefender,
    RandomWalkDefender,
    StationaryDefender,
)
from utils.drone_params import DroneParams   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 deploy eval")
    p.add_argument("--actor_ckpt", type=str, required=True,
                   help="Frozen actor — file (defender_v0.pt) or directory "
                        "(latest defender_v*.pt/attacker_v*.pt auto-picked).")
    p.add_argument("--cfgs_pkl", type=str, default=None,
                   help="Optional override. Default = cfgs.pkl in the same "
                        "directory as the resolved actor file.")
    p.add_argument("--defender", type=str, default="stationary",
                   choices=("stationary", "random_walk", "pretrained"))
    p.add_argument("--defender_ckpt", type=str, default="",
                   help="When --defender=pretrained: path to defender actor_v0.pt.")
    p.add_argument("--attacker_mode", type=str, default="stationary",
                   choices=("stationary", "random_walk"),
                   help="When ego=defender, choose attacker policy for eval. "
                        "Phase 1.0 default = stationary (matches train_defender.py).")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--episodes", type=int, default=4,
                   help="Episodes per env (rolled sequentially with auto-reset).")
    p.add_argument("--max_steps", type=int, default=0,
                   help="0 = use env.max_episode_length. Otherwise hard cap per episode.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--backend", type=str, default="cpu", choices=["gpu", "cpu"])
    p.add_argument("--show_viewer", action="store_true",
                   help="Open Genesis viewer with opponent (red) and g_mission (green) "
                        "markers. Renders first 4 envs.")
    p.add_argument("--plot_actions", type=str, default="",
                   help="Directory path. If set, dump action distribution: per-dim "
                        "histogram (T_norm, wx, wy, wz), saturation_frac stats, and "
                        "actions.npz raw trace. PR-O squashing diagnostic.")
    p.add_argument("--substeps", type=int, default=2,
                   help="Override env_cfg.substeps (default 2 = visualizer-friendly "
                        "prop animation). Set 0 to keep cfgs.pkl value (학습 시와 "
                        "동등 sim 정확도, 그러나 prop 시각화 한계). 학습 시는 16 권장.")
    p.add_argument(
        "--drone_yaml", type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
    )
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfgs(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def resolve_eval_paths(actor_ckpt: str, cfgs_pkl: str | None) -> tuple[Path, Path]:
    """``actor_ckpt`` (file or directory) → resolved actor file + cfgs.pkl.

    - directory: tries ``defender_v*.pt`` then ``attacker_v*.pt``, mtime-latest
    - file/symlink: resolve symlink, take parent for cfgs.pkl
    """
    p = Path(actor_ckpt)
    if p.is_dir():
        for prefix in ("defender", "attacker"):
            cands = sorted(p.glob(f"**/{prefix}_v*.pt"), key=lambda x: x.stat().st_mtime)
            if cands:
                actor_p = cands[-1]
                break
        else:
            raise FileNotFoundError(
                f"No defender_v*.pt or attacker_v*.pt under {p}"
            )
    elif p.is_file() or p.is_symlink():
        actor_p = p.resolve()
    else:
        raise FileNotFoundError(f"actor_ckpt not found: {p}")

    if cfgs_pkl:
        cfgs_p = Path(cfgs_pkl)
    else:
        cfgs_p = actor_p.parent / "cfgs.pkl"
        if not cfgs_p.is_file():
            raise FileNotFoundError(
                f"No cfgs.pkl next to actor ({actor_p}). Pass --cfgs_pkl explicitly."
            )
    return actor_p, cfgs_p


def build_attacker(mode: str, num_envs: int, device):
    if mode == "stationary":
        return StationaryAttacker(num_envs=num_envs, device=device)
    if mode == "random_walk":
        return RandomWalkAttacker(num_envs=num_envs, device=device)
    raise ValueError(f"attacker_mode={mode!r} not supported in eval")


def build_defender(mode: str, num_envs: int, defender_ckpt: str, device):
    if mode == "stationary":
        return StationaryDefender(num_envs=num_envs, device=device)
    if mode == "random_walk":
        return RandomWalkDefender(num_envs=num_envs, device=device)
    if mode == "pretrained":
        if not defender_ckpt:
            raise SystemExit("--defender pretrained requires --defender_ckpt.")
        return PretrainedDefender(num_envs=num_envs, ckpt_path=defender_ckpt, device=device)
    raise ValueError(f"Unknown defender mode: {mode!r}")


def _assert_no_critic_priv(env) -> None:
    """CLAUDE.md §7 — deploy must not build privileged critic obs."""
    obs = env.get_observations()
    keys = set(obs.keys())
    if "critic_priv" in keys:
        raise RuntimeError(
            "Deploy contract violated: env.get_observations() exposes 'critic_priv' "
            f"key (keys={sorted(keys)}). Phase 1 must build only 'policy'."
        )
    if env.obs_groups != {"actor": ["policy"], "critic": ["policy"]}:
        raise RuntimeError(
            f"Deploy contract violated: env.obs_groups={env.obs_groups}. "
            "Phase 1 expects actor=critic=['policy']."
        )


@torch.no_grad()
def main() -> int:
    args = parse_args()
    _seed_all(args.seed)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")
    device = "cuda" if (args.backend == "gpu" and torch.cuda.is_available()) else "cpu"

    actor_path, cfgs_path = resolve_eval_paths(args.actor_ckpt, args.cfgs_pkl)
    print(f"[eval] actor_ckpt resolved → {actor_path}")
    print(f"[eval] cfgs_pkl  resolved → {cfgs_path}")
    cfgs = load_cfgs(cfgs_path)
    env_cfg = dict(cfgs["env_cfg"])
    obs_cfg = cfgs["obs_cfg"]
    reward_cfg = cfgs["reward_cfg"]
    command_cfg = cfgs["command_cfg"]
    drone_yaml = cfgs.get("drone_yaml", args.drone_yaml)
    if args.substeps > 0 and args.substeps != int(env_cfg.get("substeps", -1)):
        print(f"[eval] override substeps {env_cfg.get('substeps')} → {args.substeps} "
              f"(visualizer-friendly; sim 정확도 저하 가능. "
              f"--substeps 0 또는 학습값 명시로 override 해제)")
        env_cfg["substeps"] = int(args.substeps)

    ego = env_cfg["ego"]
    print(f"[eval] cfgs.pkl ego={ego} ams_stage={cfgs.get('ams_stage')}")

    params = DroneParams(drone_yaml)

    # Pair the right scripted/loaded counterpart.
    if ego == "attacker":
        defender = build_defender(args.defender, args.num_envs, args.defender_ckpt, gs.device)
        attacker = None
    else:
        attacker = build_attacker(args.attacker_mode, args.num_envs, gs.device)
        defender = None

    env = PursuitEvasion1v1Env(
        num_envs=args.num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        attacker_policy=attacker,
        defender_policy=defender,
        show_viewer=args.show_viewer,
    )

    _assert_no_critic_priv(env)

    try:
        import rsl_rl  # noqa: F401
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "Eval requires rsl-rl to unpickle the actor module."
        ) from e

    actor = torch.load(actor_path, map_location=device, weights_only=False)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)

    max_steps = int(args.max_steps) if args.max_steps > 0 else env.max_episode_length
    target_episodes = int(args.episodes) * env.num_envs

    # Counters (per-episode terminal classifications).
    n_done = 0
    counts = {
        "mission": 0,
        "captured": 0,
        "crash": 0,
        "oob": 0,
        "timeout": 0,
    }
    rew_sum_attacker = 0.0
    rew_sum_defender = 0.0
    step_count = 0

    # PR-O — action distribution trace (opt-in via --plot_actions).
    record_actions = bool(args.plot_actions)
    action_buf: list[np.ndarray] = []
    raw_mean_buf: list[np.ndarray] = []   # actor(td) 결과 (squashed: tanh-bounded)

    obs = env.reset()
    while n_done < target_episodes and step_count < max_steps * args.episodes * 2:
        td = TensorDict(
            {"policy": obs["policy"]}, batch_size=[env.num_envs], device=env.device,
        )
        raw = actor(td)                                                 # (B, 4) — squashed: ∈ [-1,1]; vanilla: unbounded
        action = raw.clamp(-1.0, 1.0)
        if record_actions:
            action_buf.append(action.detach().cpu().numpy().copy())
            raw_mean_buf.append(raw.detach().cpu().numpy().copy())
        obs, _rew, _done, _extras = env.step(action)
        rew_sum_attacker += float(env.rew_attacker_buf.sum().item())
        rew_sum_defender += float(env.rew_defender_buf.sum().item())
        # Aggregate priority-resolved buffers as episode terminations.
        # env.step 후 reset_idx가 done env들의 *_buf를 False로 wipe하기 때문에
        # buf를 직접 읽으면 항상 0. last_term snapshot은 reset_idx에 영향 받지 않음.
        counts["mission"]  += int(env.last_term["mission"].sum().item())
        counts["captured"] += int(env.last_term["captured"].sum().item())
        counts["crash"]    += int(env.last_term["crash"].sum().item())
        counts["oob"]      += int(env.last_term["oob"].sum().item())
        counts["timeout"]  += int(env.last_term["timeout"].sum().item())
        n_done += int(env.reset_buf.sum().item())
        step_count += 1

    total_eps = max(n_done, 1)
    print(f"\n[eval] episodes={n_done} steps={step_count} num_envs={env.num_envs}")
    print(f"  attacker_winrate (mission/total) = {counts['mission'] / total_eps:.4f}")
    print(f"  captured_rate                    = {counts['captured'] / total_eps:.4f}")
    print(f"  crash_rate                       = {counts['crash'] / total_eps:.4f}")
    print(f"  oob_rate                         = {counts['oob'] / total_eps:.4f}")
    print(f"  timeout_rate                     = {counts['timeout'] / total_eps:.4f}")
    print(f"  mean rew_attacker / step / env   = {rew_sum_attacker / (step_count * env.num_envs):.4f}")
    print(f"  mean rew_defender / step / env   = {rew_sum_defender / (step_count * env.num_envs):.4f}")
    print("[eval] critic_priv contract: OK (deploy mode never built privileged obs)")

    # PR-O — action distribution dump
    if record_actions and action_buf:
        out_dir = Path(args.plot_actions).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        actions_np = np.concatenate(action_buf, axis=0)                  # (T*B, 4)
        raw_np = np.concatenate(raw_mean_buf, axis=0)                    # (T*B, 4)
        labels = ["T_norm", "wx_ref", "wy_ref", "wz_ref"]
        sat_frac = (np.abs(actions_np) > 0.95).mean(axis=0)
        raw_max = np.abs(raw_np).max(axis=0)
        print(f"\n[eval] action distribution ({actions_np.shape[0]} samples)")
        print(f"{'dim':>8s} | {'mean':>7s} {'std':>7s} {'min':>7s} {'max':>7s} "
              f"{'|sat>.95|':>9s} {'|raw|max':>9s}")
        for i, lbl in enumerate(labels):
            print(f"{lbl:>8s} | {actions_np[:,i].mean():>7.3f} {actions_np[:,i].std():>7.3f} "
                  f"{actions_np[:,i].min():>7.3f} {actions_np[:,i].max():>7.3f} "
                  f"{sat_frac[i]:>9.3f} {raw_max[i]:>9.2f}")
        print(f"  any_dim_saturated_per_step = "
              f"{((np.abs(actions_np) > 0.95).any(axis=1)).mean():.3f}")

        np.savez_compressed(
            out_dir / "actions.npz",
            actions=actions_np, raw=raw_np, labels=labels,
        )
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 4, figsize=(15, 6))
            for i, lbl in enumerate(labels):
                axes[0, i].hist(actions_np[:, i], bins=50, range=(-1.05, 1.05),
                                color="tab:blue", alpha=0.75)
                axes[0, i].axvline(-1, ls=":", color="r"); axes[0, i].axvline(1, ls=":", color="r")
                axes[0, i].set_title(f"{lbl}  sat>.95={sat_frac[i]:.2f}")
                axes[0, i].set_xlim(-1.1, 1.1)
                axes[1, i].hist(raw_np[:, i], bins=50, color="tab:orange", alpha=0.75)
                axes[1, i].axvline(-1, ls=":", color="r"); axes[1, i].axvline(1, ls=":", color="r")
                axes[1, i].set_title(f"{lbl} raw  |max|={raw_max[i]:.2f}")
            fig.suptitle(f"action distribution — {Path(args.actor_ckpt).name}", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(out_dir / "action_dist.png", dpi=120)
            print(f"[eval] saved {out_dir / 'action_dist.png'}")
        except Exception as e:
            print(f"[eval] plot skipped: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
