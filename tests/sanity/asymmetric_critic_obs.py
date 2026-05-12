"""V3 sanity — asymmetric critic ``obs_groups`` deploy contract.

Goal (TODO ⚠️ V3, CLAUDE.md §7)
-------------------------------
Verify that rsl-rl 5.2's ``obs_groups`` mechanism actually **routes the
privileged tensor only to the critic** and that the actor never reads it —
so deploy / inference can omit the ``critic_priv`` key entirely.

This is the contract that protects the asymmetric-critic theory: the critic
sees ground-truth defender state at training time, the actor only sees the
deployment-feasible (delayed + forecasted) view, and at deploy time
``critic_priv`` is never built. If this contract is wrong, every value
prediction at training would leak privileged info into the actor's gradients
through some shared pathway, undermining the whole approach.

Three contract tests
--------------------
1.  **Dimensionality split**:
        actor.obs_dim   == dim("policy")
        critic.obs_dim  == dim("policy") + dim("critic_priv")
    If wrong: the model literally connects the wrong wires.

2.  **Actor blindness**:
        Perturb only ``obs["critic_priv"]`` → actor(obs) output unchanged.
    If wrong: the actor is silently consuming privileged state.

3.  **Critic uses privileged**:
        Perturb only ``obs["critic_priv"]`` → critic(obs) output **does** change.
    If wrong: the asymmetric advantage is meaningless — privileged input
    is being dropped on the floor.

4.  **Deploy contract**:
        Build a TensorDict with **only** ``"policy"`` (no ``critic_priv``
        key at all) and call the actor. It must produce the same output as
        when called on a full TensorDict with the same policy values. This
        is the production deploy code path — environments must be allowed
        to skip building ``critic_priv`` altogether.

Pass criteria
-------------
All four contract tests pass. Failure → revisit obs_groups wiring or rsl-rl
version (e.g., a regression in 5.2.x).

Usage
-----
    python tests/sanity/asymmetric_critic_obs.py
    python tests/sanity/asymmetric_critic_obs.py --device cpu
"""

from __future__ import annotations

import argparse
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


# --- Group dims (CLAUDE.md §7 keys) ---
POLICY_DIM = 9          # [pos(3), vel(3), goal(3)]
CRITIC_PRIV_DIM = 6     # [true_pos(3), true_vel(3)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3 sanity: asymmetric critic obs_groups")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--iters", type=int, default=2, help="train iterations before contract probes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    p.add_argument("--rtol", type=float, default=1e-5, help="strict equality tolerance for actor blindness")
    p.add_argument("--priv_perturb_min", type=float, default=1e-2, help="min |Δ value| when critic_priv changes")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Toy vec env — exposes the CLAUDE.md §7 obs_groups split
# ---------------------------------------------------------------------------

class ToyAsymVecEnv:
    """Minimal vec env exposing two obs groups: ``policy`` (shared) and
    ``critic_priv`` (privileged, critic-only).

    The dynamics are intentionally simple: a 3D damped double-integrator
    chasing a goal. ``critic_priv`` carries the *true* (no-noise) pos/vel
    while ``policy`` carries the same pos/vel with optional perturbation
    semantics — but for V3 we just need the keys to exist with distinct
    information so the contract tests are meaningful.
    """

    def __init__(self, num_envs: int, device: str, seed: int) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.num_actions = 3
        self.dt = 0.05
        self.max_episode_length = 50

        self.obs_groups = {
            "actor": ["policy"],
            "critic": ["policy", "critic_priv"],
        }
        self.cfg: dict[str, Any] = {
            "obs_groups": self.obs_groups,
            "num_actions": self.num_actions,
        }

        self.pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.vel = torch.zeros_like(self.pos)
        self.goal = torch.zeros_like(self.pos)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(seed)
        self.reset()

    def _obs(self) -> TensorDict:
        policy = torch.cat([self.pos, self.vel, self.goal], dim=1)  # shape: (B, 9)
        critic_priv = torch.cat([self.pos, self.vel], dim=1)         # shape: (B, 6)
        return TensorDict(
            {"policy": policy, "critic_priv": critic_priv},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

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

        rewards = -(self.pos - self.goal).norm(dim=1).clamp_min(1e-6)
        dones = self.episode_length_buf >= self.max_episode_length

        done_idx = dones.nonzero(as_tuple=False).reshape(-1)
        if done_idx.numel() > 0:
            self.reset(env_ids=done_idx)

        return self._obs(), rewards, dones, {}


def make_train_cfg() -> dict[str, Any]:
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
            "critic": ["policy", "critic_priv"],
        },
        "num_steps_per_env": 8,
        "save_interval": 10_000,
        "run_name": "v3_sanity",
        "logger": "tensorboard",
    }


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        print(
            f"[FAIL] rsl-rl-lib import failed: {exc}\n"
            "Hint: pip install 'rsl-rl-lib>=5.0.0'",
            file=sys.stderr,
        )
        return 2

    log_root = Path(tempfile.mkdtemp(prefix="gpe_v3_sanity_"))

    print(f"[info] device={args.device} num_envs={args.num_envs}")
    print(f"[info] obs_groups: actor=['policy'], critic=['policy', 'critic_priv']")
    print(f"[info] expected dims: policy={POLICY_DIM}, critic_priv={CRITIC_PRIV_DIM}")

    try:
        env = ToyAsymVecEnv(args.num_envs, device=args.device, seed=args.seed)
        cfg = make_train_cfg()
        runner = OnPolicyRunner(env, cfg, str(log_root), device=args.device)

        actor = runner.alg.actor
        critic = runner.alg.critic

        # ------------------------------------------------------------------
        # Test 1 — Dimensionality split.
        # ------------------------------------------------------------------
        actor_dim = int(getattr(actor, "obs_dim", -1))
        critic_dim = int(getattr(critic, "obs_dim", -1))
        ok_actor_dim = actor_dim == POLICY_DIM
        ok_critic_dim = critic_dim == POLICY_DIM + CRITIC_PRIV_DIM
        print(f"\n[test 1] dimensionality split")
        print(f"  actor.obs_dim  = {actor_dim}  (expect {POLICY_DIM})           -> {'OK' if ok_actor_dim else 'FAIL'}")
        print(f"  critic.obs_dim = {critic_dim}  (expect {POLICY_DIM + CRITIC_PRIV_DIM})  -> {'OK' if ok_critic_dim else 'FAIL'}")

        # ------------------------------------------------------------------
        # Train a couple of iterations so the network is past its identity-init
        # and small numerical changes from critic_priv perturbation are visible.
        # ------------------------------------------------------------------
        print(f"\n[info] training {args.iters} iter to settle weights ...")
        runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)

        # Switch to eval-style call: deterministic forward, no grad.
        actor.eval()
        critic.eval()

        # ------------------------------------------------------------------
        # Test 2 — Actor blindness.
        # ------------------------------------------------------------------
        with torch.no_grad():
            obs_full = env.get_observations()
            obs_perturbed = obs_full.clone()
            # Replace critic_priv with garbage of the SAME shape; keep policy intact
            obs_perturbed["critic_priv"] = torch.randn_like(obs_full["critic_priv"]) * 1e3

            action_full = actor(obs_full)
            action_perturbed = actor(obs_perturbed)

        act_diff = (action_full - action_perturbed).abs().max().item()
        ok_actor_blind = act_diff < args.rtol
        print(f"\n[test 2] actor blindness to critic_priv perturbation")
        print(f"  max |Δ action| = {act_diff:.3e}  (rtol {args.rtol})  -> {'OK' if ok_actor_blind else 'FAIL'}")

        # ------------------------------------------------------------------
        # Test 3 — Critic actually uses critic_priv.
        # ------------------------------------------------------------------
        with torch.no_grad():
            value_full = critic(obs_full)
            value_perturbed = critic(obs_perturbed)
        val_diff = (value_full - value_perturbed).abs().max().item()
        ok_critic_uses_priv = val_diff > args.priv_perturb_min
        print(f"\n[test 3] critic does use critic_priv (NOT silently dropped)")
        print(f"  max |Δ value| = {val_diff:.3e}  (need > {args.priv_perturb_min})  -> {'OK' if ok_critic_uses_priv else 'FAIL'}")

        # ------------------------------------------------------------------
        # Test 4 — Deploy contract: actor must work with policy-only obs.
        # ------------------------------------------------------------------
        with torch.no_grad():
            policy_only = TensorDict(
                {"policy": obs_full["policy"].clone()},
                batch_size=[env.num_envs],
                device=env.device,
            )
            try:
                action_deploy = actor(policy_only)
                deploy_built = True
            except Exception as exc:
                action_deploy = None
                deploy_built = False
                deploy_err = repr(exc)

        if deploy_built and action_deploy is not None:
            deploy_diff = (action_deploy - action_full).abs().max().item()
            ok_deploy = deploy_diff < args.rtol
            print(f"\n[test 4] deploy contract — actor on TensorDict with ONLY 'policy'")
            print(f"  forward succeeded: True")
            print(f"  max |Δ action vs full obs| = {deploy_diff:.3e}  -> {'OK' if ok_deploy else 'FAIL'}")
        else:
            ok_deploy = False
            print(f"\n[test 4] deploy contract — actor on TensorDict with ONLY 'policy'")
            print(f"  forward FAILED: {deploy_err}")
            print("  -> FAIL")

        # ------------------------------------------------------------------
        all_ok = all([ok_actor_dim, ok_critic_dim, ok_actor_blind, ok_critic_uses_priv, ok_deploy])
        print("\n[verdict]")
        print(f"  actor.obs_dim correct:        {ok_actor_dim}")
        print(f"  critic.obs_dim correct:       {ok_critic_dim}")
        print(f"  actor blind to critic_priv:   {ok_actor_blind}")
        print(f"  critic uses critic_priv:      {ok_critic_uses_priv}")
        print(f"  deploy mode (policy-only):    {ok_deploy}")

        if all_ok:
            print(
                "\n[PASS] V3 — asymmetric obs_groups contract holds. "
                "Critic gets privileged info; actor is blind. "
                "Deploy can omit critic_priv entirely. CLAUDE.md §7 satisfied."
            )
            return 0
        print(
            "\n[FAIL] V3 — at least one contract is violated. "
            "Do NOT proceed to Phase 1 until obs_groups wiring is fixed."
        )
        return 1

    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if log_root.exists():
            try:
                shutil.rmtree(log_root, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
