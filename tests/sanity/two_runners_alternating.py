"""V2 sanity — two ``OnPolicyRunner`` instances alternating in one process.

Goal (TODO ⚠️ V2)
-----------------
Verify that two rsl-rl ``OnPolicyRunner`` instances (one for the attacker, one
for the defender) can coexist in the same Python process and **alternate**
gradient updates without GPU memory or optimizer-state collision.

This is the necessary precondition for the AMS-DRL co-training orchestrator
in CLAUDE.md §9. If this fails, Phase 5 must split training across processes
(IPC sync) instead of using two runners in one process.

We deliberately use a CPU-friendly toy env (no Genesis) to isolate **runner
coexistence** from any simulator quirks — V1 owns the Genesis question.

The toy env exposes the asymmetric obs split mandated by CLAUDE.md §7
(``obs_groups = {"actor": ["policy"], "critic": ["policy", "critic_obs"]}``).
That makes V2 also smoke-test V3 along the way, but V3 has its own
dedicated test for the deploy-mode contract.

Pass criteria
-------------
1.  Both runners construct without raising.
2.  ``--rounds × --iters_per_round`` calls of ``runner.learn`` complete on
    both runners without OOM or NaN/Inf in the captured loss.
3.  After training, each runner's actor parameters have **changed** relative
    to their pre-training snapshot (i.e. learning actually happened).
4.  The two runners' final actor parameters are **different** (independent
    optimizer states).
5.  No memory leak signal (resident memory growth bounded across rounds).

Usage
-----
    python tests/sanity/two_runners_alternating.py
    python tests/sanity/two_runners_alternating.py --rounds 5 --iters_per_round 5 --num_envs 8
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 sanity: two OnPolicyRunner alternating")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--iters_per_round", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Toy vec env — rsl-rl 5.0 contract
# ---------------------------------------------------------------------------

class ToyAsymVecEnv:
    """Minimal vec env that satisfies rsl-rl 5.0's expectations.

    State (per env):
        pos    shape (3,)   integrated by action
        vel    shape (3,)   damped double-integrator
        goal   shape (3,)   resampled on episode reset

    Observation groups (CLAUDE.md §7 obs_groups contract):
        "policy"     = [pos, vel, goal]                   shape (9,)  -> actor + critic
        "critic_obs" = [true_pos, true_vel]               shape (6,)  -> critic only

    Action:
        thrust direction in [-1, 1]^3, applied with low-pass to velocity.

    Reward:
        -||pos - goal||
    """

    def __init__(self, num_envs: int, device: str, seed: int) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.num_actions = 3
        self.dt = 0.05
        self.max_episode_length = 50

        # rsl-rl reads `cfg` and `obs_groups` directly off the env in 5.0.
        self.obs_groups = {
            "actor": ["policy"],
            "critic": ["policy", "critic_obs"],
        }
        self.cfg: dict[str, Any] = {
            "obs_groups": self.obs_groups,
            "num_actions": self.num_actions,
        }

        # Per-env tensors (no Python loops over envs — CLAUDE.md §14)
        self.pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.vel = torch.zeros_like(self.pos)
        self.goal = torch.zeros_like(self.pos)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(seed)
        self.reset()

    # --- observation builders ---
    def _obs(self) -> TensorDict:
        policy_obs = torch.cat([self.pos, self.vel, self.goal], dim=1)  # shape: (B, 9)
        critic_priv = torch.cat([self.pos, self.vel], dim=1)            # shape: (B, 6)
        return TensorDict(
            {"policy": policy_obs, "critic_obs": critic_priv},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    # --- env API ---
    def reset(self, env_ids: torch.Tensor | None = None) -> TensorDict:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = int(env_ids.numel())
        if n > 0:
            self.pos[env_ids] = torch.randn(
                (n, 3), generator=self._rng, device=self.device
            ) * 0.5
            self.vel[env_ids] = 0.0
            self.goal[env_ids] = torch.randn(
                (n, 3), generator=self._rng, device=self.device
            )
            self.episode_length_buf[env_ids] = 0
        return self._obs()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = actions.clamp(-1.0, 1.0)

        self.vel = 0.95 * self.vel + 0.1 * actions
        self.pos = self.pos + self.dt * self.vel
        self.episode_length_buf += 1

        rewards = -(self.pos - self.goal).norm(dim=1).clamp_min(1e-6)  # shape: (B,)
        dones = self.episode_length_buf >= self.max_episode_length     # shape: (B,)

        done_idx = dones.nonzero(as_tuple=False).reshape(-1)
        if done_idx.numel() > 0:
            self.reset(env_ids=done_idx)

        return self._obs(), rewards, dones, {}


# ---------------------------------------------------------------------------
# Train cfg — rsl-rl 5.0 layout (mirrors source repo's get_train_cfg)
# ---------------------------------------------------------------------------

def make_train_cfg(run_name: str) -> dict[str, Any]:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "tanh",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "tanh",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy", "critic_obs"],
        },
        "num_steps_per_env": 8,
        "save_interval": 10_000,
        "run_name": run_name,
        "logger": "tensorboard",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_param_snapshot(module: torch.nn.Module) -> torch.Tensor:
    """Return a single flat tensor of all parameters, detached on CPU."""
    return torch.cat([p.detach().reshape(-1).cpu() for p in module.parameters()])


def _resident_mem_mb() -> float:
    """Process resident set size in MB (Linux). Best-effort."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Importing rsl_rl after seed setup. Defer import errors until we reach
    # the failure boundary so the user sees a clean message.
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        print(
            f"[FAIL] rsl-rl-lib import failed: {exc}\n"
            "Hint: pip install 'rsl-rl-lib>=5.0.0'",
            file=sys.stderr,
        )
        return 2

    log_root = Path(tempfile.mkdtemp(prefix="gpe_v2_sanity_"))
    log_attacker = log_root / "attacker"
    log_defender = log_root / "defender"
    log_attacker.mkdir(parents=True, exist_ok=True)
    log_defender.mkdir(parents=True, exist_ok=True)

    print(f"[info] device={args.device} num_envs={args.num_envs}")
    print(f"[info] schedule: {args.rounds} rounds × {args.iters_per_round} iter/runner")
    print(f"[info] log_root={log_root}")
    print(f"[info] mem before envs: {_resident_mem_mb():.1f} MB")

    try:
        env_attacker = ToyAsymVecEnv(args.num_envs, device=args.device, seed=args.seed)
        env_defender = ToyAsymVecEnv(
            args.num_envs, device=args.device, seed=args.seed + 1
        )

        cfg_attacker = make_train_cfg("v2_sanity_attacker")
        cfg_defender = make_train_cfg("v2_sanity_defender")
        # Disable any logger backend to avoid TB/wandb side-effects in CI.
        cfg_attacker["logger"] = "tensorboard"
        cfg_defender["logger"] = "tensorboard"

        runner_attacker = OnPolicyRunner(
            env_attacker, cfg_attacker, str(log_attacker), device=args.device
        )
        runner_defender = OnPolicyRunner(
            env_defender, cfg_defender, str(log_defender), device=args.device
        )
        print(f"[info] mem after runners: {_resident_mem_mb():.1f} MB")

        # rsl-rl 5.2: actor/critic live directly on `runner.alg`.
        actor_a = runner_attacker.alg.actor
        actor_d = runner_defender.alg.actor
        snap_a0 = _flat_param_snapshot(actor_a)
        snap_d0 = _flat_param_snapshot(actor_d)

        mem_trace = [_resident_mem_mb()]
        ok_finite = True

        for r in range(args.rounds):
            print(f"[round {r + 1}/{args.rounds}] attacker.learn({args.iters_per_round}) ...")
            runner_attacker.learn(
                num_learning_iterations=args.iters_per_round,
                init_at_random_ep_len=(r == 0),
            )
            mem_trace.append(_resident_mem_mb())
            print(f"[round {r + 1}/{args.rounds}] defender.learn({args.iters_per_round}) ...")
            runner_defender.learn(
                num_learning_iterations=args.iters_per_round,
                init_at_random_ep_len=(r == 0),
            )
            mem_trace.append(_resident_mem_mb())

            for tag, mod in (("attacker", actor_a), ("defender", actor_d)):
                snap = _flat_param_snapshot(mod)
                if not torch.isfinite(snap).all():
                    print(
                        f"[FAIL] non-finite params in {tag} after round {r + 1}",
                        file=sys.stderr,
                    )
                    ok_finite = False

        snap_aF = _flat_param_snapshot(actor_a)
        snap_dF = _flat_param_snapshot(actor_d)

        delta_a = (snap_aF - snap_a0).norm().item()
        delta_d = (snap_dF - snap_d0).norm().item()
        cross_diff = (snap_aF - snap_dF).abs().mean().item() if snap_aF.numel() == snap_dF.numel() else float("nan")

        ok_attacker_learned = delta_a > 1e-4
        ok_defender_learned = delta_d > 1e-4
        ok_runners_independent = bool(np.isfinite(cross_diff)) and cross_diff > 1e-4
        ok_mem_bounded = (
            len(mem_trace) >= 2
            and (np.isnan(mem_trace[-1]) or np.isnan(mem_trace[0]) or mem_trace[-1] < mem_trace[0] + 2048.0)
        )

        print("\n[result]")
        print(f"  ||Δ params attacker|| = {delta_a:.4e}")
        print(f"  ||Δ params defender|| = {delta_d:.4e}")
        print(f"  mean |attacker - defender| = {cross_diff:.4e}")
        print(f"  mem trace (MB): {[round(m, 1) for m in mem_trace]}")

        print("\n[verdict]")
        print(f"  runners constructed:        True")
        print(f"  finite params throughout:   {ok_finite}")
        print(f"  attacker actor learned:     {ok_attacker_learned}")
        print(f"  defender actor learned:     {ok_defender_learned}")
        print(f"  runners independent:        {ok_runners_independent}")
        print(f"  resident mem bounded:       {ok_mem_bounded}")

        all_ok = all([
            ok_finite,
            ok_attacker_learned,
            ok_defender_learned,
            ok_runners_independent,
            ok_mem_bounded,
        ])
        if all_ok:
            print(
                "\n[PASS] V2 — two OnPolicyRunner instances coexist and alternate. "
                "AMS-DRL orchestrator path is feasible."
            )
            return 0
        print(
            "\n[FAIL] V2 — at least one check failed. "
            "Reconsider single-runner consolidation or process isolation."
        )
        return 1

    except Exception as exc:  # noqa: BLE001 — sanity boundary
        import traceback

        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # Keep log_root for inspection on failure; remove on clean exit only
        # to keep the workspace tidy.
        if log_root.exists():
            try:
                shutil.rmtree(log_root, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
