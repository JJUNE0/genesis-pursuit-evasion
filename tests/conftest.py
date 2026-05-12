"""Shared pytest fixtures.

- ``_gs_init`` initialises Genesis once per test session on CPU so the suite
  is runnable in CI / on dev machines without a GPU.
- ``pe_env_attacker_stationary`` builds a small ``PursuitEvasion1v1Env`` with
  ego=attacker vs a stationary scripted defender; reused across termination /
  reward / shape tests so we only pay one Scene-build per session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import genesis as gs


# ---------------------------------------------------------------------------
# Genesis init (session-scoped, CPU backend)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _gs_init():
    try:
        gs.init(backend=gs.cpu, seed=42, logging_level="warning")
    except Exception:   # noqa: BLE001 — second init is fine to ignore
        pass
    yield


# ---------------------------------------------------------------------------
# PE env fixture (session-scoped — Scene build is expensive)
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def pe_env_attacker_stationary(_gs_init):
    """ego=attacker, defender=Stationary; B=4 on CPU.

    Each test must call ``env.reset()`` to start from a clean slate before
    forcing positions / g_mission, since the fixture is session-scoped.
    """
    from envs.pe_1v1.cfgs import (
        load_pe_1v1_cfg,
        make_command_cfg,
        make_env_cfg,
        make_obs_cfg,
        make_reward_cfg,
    )
    from envs.pe_1v1.env import PursuitEvasion1v1Env
    from envs.pe_1v1.scripted_defender import StationaryDefender
    from utils.drone_params import DroneParams

    raw = load_pe_1v1_cfg()
    env_cfg = make_env_cfg(raw, ego="attacker")
    obs_cfg = make_obs_cfg(raw)
    reward_cfg = make_reward_cfg(raw)
    command_cfg = make_command_cfg(raw)

    drone_yaml = _project_root() / "configs" / "drones" / "nova.yaml"
    params = DroneParams(str(drone_yaml))

    num_envs = 4
    defender = StationaryDefender(num_envs=num_envs, device=gs.device)

    env = PursuitEvasion1v1Env(
        num_envs=num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        defender_policy=defender,
        show_viewer=False,
    )
    return env


@pytest.fixture(scope="session")
def pe_env_defender_stationary(_gs_init):
    """ego=defender, attacker=Stationary; B=4 on CPU. PR-N obs parity test용.

    Mirror of ``pe_env_attacker_stationary`` with sides swapped — defender
    is ego, frozen-at-spawn StationaryAttacker drives the opponent. Each test
    must call ``env.reset()`` before forcing positions / g_mission.
    """
    from envs.pe_1v1.cfgs import (
        load_pe_1v1_cfg,
        make_command_cfg,
        make_env_cfg,
        make_obs_cfg,
        make_reward_cfg,
    )
    from envs.pe_1v1.env import PursuitEvasion1v1Env
    from envs.pe_1v1.scripted_attacker import StationaryAttacker
    from utils.drone_params import DroneParams

    raw = load_pe_1v1_cfg()
    env_cfg = make_env_cfg(raw, ego="defender")
    obs_cfg = make_obs_cfg(raw)
    reward_cfg = make_reward_cfg(raw)
    command_cfg = make_command_cfg(raw)

    drone_yaml = _project_root() / "configs" / "drones" / "nova.yaml"
    params = DroneParams(str(drone_yaml))

    num_envs = 4
    attacker = StationaryAttacker(num_envs=num_envs, device=gs.device)

    env = PursuitEvasion1v1Env(
        num_envs=num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        attacker_policy=attacker,
        show_viewer=False,
    )
    return env
