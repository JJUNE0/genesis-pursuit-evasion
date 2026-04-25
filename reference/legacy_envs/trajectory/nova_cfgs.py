"""Nova TrajectoryEnv 설정"""

DRONE_CONFIG = "configs/nova/drone_params.yaml"


def get_cfgs() -> tuple:
    env_cfg = {
        "num_actions": 4,
        "actuator_mode": "rpm",
        "termination_if_roll_greater_than": 150,
        "termination_if_pitch_greater_than": 150,
        "termination_if_close_to_ground": 0.15,
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "at_target_threshold": 0.15,
        "simulate_action_latency": False,
        "action_latency_steps": 1,
        "clip_actions": 1.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        "workspace_margin": 1.0,
        # Agile Flight: Adaptive Temporal Scaling (ATS)
        "ats_enabled": False,       # set True to enable ATS
        "ats_easy_dist": 0.10,      # dist < this → speed up (alpha * 1+rate)
        "ats_hard_dist": 0.30,      # dist > this → slow down (alpha * 1-rate)
        "ats_rate": 0.001,          # alpha change per step
        "ats_alpha_min": 0.5,
        "ats_alpha_max": 2.0,
        # Agile Flight: Residual Dynamics online learning
        "use_residual": False,      # set True to enable
        "residual_hidden_dim": 64,
        "residual_buffer_size": 100_000,
        "residual_batch_size": 256,
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
            "target": 5.0,
            "waypoint": 2.0,
            "completion": 10.0,
            "smooth": -5e-2,
            "yaw": 0.01,
            "angular": -1e-3,
            "crash": -10.0,
        },
    }
    command_cfg = {
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [1.0, 1.0],
        "num_waypoints": 4,
        "min_waypoints": 4,
        "max_waypoints": 4,
        "obs_waypoint_count": 3,
        "min_waypoint_distance": 0.4,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg
