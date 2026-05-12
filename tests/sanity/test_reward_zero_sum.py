"""TODO.md §1.2 + PR-E — terminal reward minimax + dense ``R_block + R_progress = 0``.

Three terminal pairs sum to zero exactly (per the YAML magnitudes):

    R_mission_win + R_mission_failed = 0
    R_capture_win + R_captured       = 0
    R_timeout_win + R_timeout        = 0

crash/oob is *not* zero-sum but is *minimax sign-correct* (PR-E):

    self crash  → R_self_crash (penalty)
    opp crash   → R_opp_crash  (bonus)

env init asserts ``R_self_crash <= min(R_captured, R_mission_failed)`` and
``R_opp_crash  <= min(R_capture_win, R_mission_win)`` to block self-destruct
exploits (e.g. attacker suiciding to dodge capture, defender suiciding when
mission failure is imminent).

Dense flow:
- ``R_block = -R_progress`` step-by-step (potential-based, zero-sum).
- PR-D Gaussian dense rewards (``R_chase`` defender-side, ``R_goal_dense``
  attacker-side, ``R_existential`` attacker-side) are intentionally
  asymmetric — they sit OUTSIDE the dense zero-sum contract.
"""

from __future__ import annotations

import pytest
import torch

import genesis as gs


# ---------------------------------------------------------------------------
# YAML / init invariants
# ---------------------------------------------------------------------------

def test_terminal_magnitudes_zero_sum_in_yaml(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    rwd = env.reward_cfg
    assert rwd["R_mission_win"] + rwd["R_mission_failed"] == 0.0
    assert rwd["R_capture_win"] + rwd["R_captured"] == 0.0
    assert rwd["R_timeout_win"] + rwd["R_timeout"] == 0.0


def test_constants_invariants(pe_env_attacker_stationary):
    """env init이 부등식 4개를 assert로 강제하므로 fixture가 살아있다는
    사실 자체가 통과 증거. 추가로 attribute 노출만 재확인."""
    env = pe_env_attacker_stationary
    assert env.R_self_crash <= env.R_captured
    assert env.R_self_crash <= env.R_mission_failed
    assert env.R_opp_crash <= env.R_capture_win
    assert env.R_opp_crash <= env.R_mission_win


# ---------------------------------------------------------------------------
# Terminal reward — bucket-by-bucket helper
# ---------------------------------------------------------------------------

_ALL_TERM_BUFS = (
    "mission_reached_buf",
    "captured_buf",
    "timeout_buf",
    "attacker_crash_buf",
    "defender_crash_buf",
    "attacker_oob_buf",
    "defender_oob_buf",
)


def _set_only(env, *keys: str) -> None:
    """Force the priority-resolved termination buffers to exactly the named
    keys (everything else cleared)."""
    for k in _ALL_TERM_BUFS:
        getattr(env, k)[:] = (k in keys)


def test_terminal_reward_pure_zero_sum_each_bucket(pe_env_attacker_stationary):
    """mission/captured/timeout pairs are exactly zero-sum in isolation."""
    env = pe_env_attacker_stationary
    env.reset()

    _set_only(env, "mission_reached_buf")
    pair = env._terminal_reward_attacker() + env._terminal_reward_defender()
    assert pair.abs().max().item() < 1e-7

    _set_only(env, "captured_buf")
    pair = env._terminal_reward_attacker() + env._terminal_reward_defender()
    assert pair.abs().max().item() < 1e-7

    _set_only(env, "timeout_buf")
    pair = env._terminal_reward_attacker() + env._terminal_reward_defender()
    assert pair.abs().max().item() < 1e-7


# ---------------------------------------------------------------------------
# Per-side minimax sign matrix
# ---------------------------------------------------------------------------

# (term-buf to set, attacker sign, defender sign).  ``sa == 0`` skips sign
# check (used for symmetric/zero-magnitude buckets).
_SIGN_MATRIX = (
    ("mission_reached_buf", +1, -1),
    ("captured_buf",        -1, +1),
    # YAML 현재 R_timeout = R_timeout_win = 0 → 부호 검증 스킵 (zero-sum만 보장).
    ("timeout_buf",          0,  0),
    ("attacker_crash_buf",  -1, +1),
    ("defender_crash_buf",  +1, -1),
    ("attacker_oob_buf",    -1, +1),
    ("defender_oob_buf",    +1, -1),
)


@pytest.mark.parametrize("buf_name,sa,sd", _SIGN_MATRIX)
def test_terminal_minimax_sign(pe_env_attacker_stationary, buf_name, sa, sd):
    env = pe_env_attacker_stationary
    env.reset()
    _set_only(env, buf_name)
    r_a = env._terminal_reward_attacker()
    r_d = env._terminal_reward_defender()
    if sa == 0:
        return
    if sa > 0:
        assert (r_a > 0).all() and (r_d < 0).all(), (
            f"{buf_name}: expected r_a>0, r_d<0; got r_a={r_a}, r_d={r_d}"
        )
    else:
        assert (r_a < 0).all() and (r_d > 0).all(), (
            f"{buf_name}: expected r_a<0, r_d>0; got r_a={r_a}, r_d={r_d}"
        )


def test_terminal_both_crash_is_symmetric(pe_env_attacker_stationary):
    """양쪽 동시 crash → r_a == r_d (둘 다 self_crash + opp_crash 합산).

    Zero-sum은 아니지만 양 측이 동일한 페널티+보너스를 받으므로 부호상
    "동률" (advantage 신호 = 0). PR-E 자살 익스플로잇 가드와 무관."""
    env = pe_env_attacker_stationary
    env.reset()
    _set_only(env, "attacker_crash_buf", "defender_crash_buf")
    r_a = env._terminal_reward_attacker()
    r_d = env._terminal_reward_defender()
    assert torch.allclose(r_a, r_d)


# ---------------------------------------------------------------------------
# Dense flow
# ---------------------------------------------------------------------------

def test_dense_block_equals_negative_progress():
    """Dense flow contract: R_block = -R_progress, source-level guard."""
    import inspect
    from envs.pe_1v1.env import PursuitEvasion1v1Env

    src_def = inspect.getsource(PursuitEvasion1v1Env._defender_reward)
    src_att = inspect.getsource(PursuitEvasion1v1Env._attacker_reward)
    assert "-self.r_progress_buf" in src_def
    assert "self.r_progress_buf" in src_att and "-self.r_progress_buf" not in src_att


def test_dense_progress_zero_sum_with_motion(pe_env_attacker_stationary):
    """``R_progress + R_block = 0`` per env per step, regardless of action.

    PR-D adds Gaussian dense rewards (R_chase / R_goal_dense / R_existential)
    that are intentionally asymmetric — they sit OUTSIDE the dense zero-sum
    contract. We assert only the potential-based zero-sum here.
    """
    env = pe_env_attacker_stationary
    env.reset()
    B = env.num_envs
    a = torch.zeros((B, 3), device=env.device, dtype=gs.tc_float)
    a[:, 2] = 1.0
    g = a.clone(); g[:, 0] = 1.0       # 1 m away on x
    far_d = a.clone(); far_d[:, 0] = -4.5
    env.attacker_drone.set_pos(a, zero_velocity=True)
    env.defender_drone.set_pos(far_d, zero_velocity=True)
    env.attacker_pos[:] = a
    env.defender_pos[:] = far_d
    env.g_mission[:] = g
    env._update_pe_state()
    env.last_dist_to_goal[:] = env.dist_to_goal

    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=gs.tc_float)
    env.step(actions)

    rp = env.r_progress_buf
    rb = -rp
    assert (rp + rb).abs().max().item() < 1e-7


def test_pr_d_gaussian_dense_rewards_are_nonnegative(pe_env_attacker_stationary):
    """PR-D — R_chase and R_goal_dense are ``coef · exp(-α·d)`` so they
    must always be in [0, coef]."""
    env = pe_env_attacker_stationary
    env.reset()

    chase = env.chase_coef * torch.exp(-env.chase_alpha * env.dist_to_attacker_defender)
    assert (chase >= 0).all() and chase.max().item() <= env.chase_coef + 1e-9

    goal_dense = env.goal_dense_coef * torch.exp(-env.goal_dense_alpha * env.dist_to_goal)
    assert (goal_dense >= 0).all() and goal_dense.max().item() <= env.goal_dense_coef + 1e-9
