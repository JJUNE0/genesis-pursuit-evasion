"""HoverEnv config builder. Returns 4 dicts for HoverEnv.__init__.

Defaults target Nova drone + body-rate action mode (PR-K validation 후).
"""

from __future__ import annotations

import math
from typing import Any


DEFAULT_DRONE_CONFIG = "configs/drones/nova.yaml"


def get_cfgs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    env_cfg = {
        "dt": 0.005,                # 200 Hz inner loop
        "substeps": 16,
        "episode_length_s": 5.0,    # max_steps = 1000
        "clip_actions": 1.0,
        "action_mode": "rate",      # "rate" (default) | "rpm" (legacy fallback)
        # Termination
        "termination_if_roll_greater_than":  60.0,
        "termination_if_pitch_greater_than": 60.0,
        "termination_if_close_to_ground":    0.10,
        "termination_if_xy_greater_than":    3.0,
        "termination_if_z_greater_than":     3.0,
        # Visualization
        "visualize_target":   False,
        "max_visualize_FPS":  60,
        # Body-rate PD controller (PR-I 튜닝값 그대로)
        "rate_controller": {
            "kp_rate": [0.05, 0.10, 0.10],
            "kd_rate": [1.0e-4, 5.0e-5, 1.0e-4],
            "max_body_rate": math.pi,
        },
    }

    obs_cfg = {
        "obs_scales": {
            "rel_pos": 1.0 / 3.0,    # ±3m → ±1
            "lin_vel": 1.0 / 3.0,    # ±3 m/s → ±1
            "ang_vel": 1.0 / math.pi,  # ±π rad/s → ±1
        },
    }

    reward_cfg = {
        "yaw_lambda": -10.0,
        "target_alpha": 5.0,
        "hover_alpha_dist": 2.0,
        "hover_alpha_vel": 5.0,
        "reward_scales": {
            "target":  5.0,
            "hover":   2.0,
            "lin_vel": -1.0e-3,
            "smooth":  -5.0e-2,
            "yaw":      0.01,
            "angular": -1.0e-3,
            "crash":   -10.0,
        },
    }

    command_cfg = {
        # spawn 위치 — env reset 시 random init
        "spawn_pos_range": {
            "x": [-0.5, 0.5],
            "y": [-0.5, 0.5],
            "z": [0.8, 1.2],
        },
        # target hover 위치
        "target_pos_range": {
            "x": [-1.0, 1.0],
            "y": [-1.0, 1.0],
            "z": [0.8, 1.5],
        },
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg
