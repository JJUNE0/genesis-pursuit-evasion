"""CF2X HoverEnv 설정"""

DRONE_CONFIG = "configs/cf2x/drone_params.yaml"


def get_cfgs() -> tuple:
    env_cfg = {
        "num_actions": 4,
        "actuator_mode": "rpm",
        # 종료 조건
        "termination_if_roll_greater_than":  180,   # deg
        "termination_if_pitch_greater_than": 180,
        "termination_if_close_to_ground":    0.1,   # m
        "termination_if_x_greater_than":     3.0,
        "termination_if_y_greater_than":     3.0,
        "termination_if_z_greater_than":     2.0,
        # 초기 자세
        "base_init_pos":  [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s":    15.0,
        "at_target_threshold":  0.1,
        "resampling_time_s":    3.0,
        "simulate_action_latency": False,
        "action_latency_steps": 1,       # latency=True 시 몇 스텝 지연할지
        "clip_actions": 1.0,
        "visualize_target":  False,
        "visualize_camera":  False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "target":   10.0,
            "smooth":  -1e-4,
            "yaw":      0.01,
            "angular": -2e-4,
            "crash":   -10.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [ 1.0, 1.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg
