"""Nova TrackingEnv configuration — Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5"""

DRONE_CONFIG = "configs/nova/drone_params.yaml"


def get_cfgs() -> tuple:
    env_cfg = {
        "num_actions": 4,           # overwritten below if action chunking enabled
        "actuator_mode": "rpm",
        "raw_rpm_output": True,

        # Body-rate controller
        "max_body_rate": 3.0,
        "rate_kp": [2.0, 2.0, 0.5],

        # Episode timing
        "episode_length_s": 20.0,
        "dt": 0.02,                     # 50 Hz (matches Phase 3 comm-delay spec)

        # Crash / termination (BaseDroneEnv)
        "termination_if_roll_greater_than":  150,   # deg
        "termination_if_pitch_greater_than": 150,
        "termination_if_close_to_ground":    0.15,  # m

        # Workspace termination (TrackingEnv._compute_termination)
        "workspace_margin": 12.0,   # m, symmetric XY
        "max_altitude":     12.0,   # m, Z upper bound

        # Reward shaping
        "tracking_sigma": 2.0,     # σ in exp(-d/σ) proximity reward

        # Ego spawn
        "base_init_pos":  [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],

        # Battery model (Phase 2a)
        "use_battery": True,
        "battery_cfg": {
            "V_nom": 16.8,
            "V_min": 13.2,
            "decay_rate_range": (0.001, 0.01),
            "kappa": 0.02,
            "tau_0": 0.02,
            "kappa_range": (0.5, 2.0),
            "randomize": True,
        },

        # Dynamics encoder (Phase 2b)
        "use_dynamics_encoder": True,
        "residual_history_len": 16,

        # Communication delay (Phase 3a)
        "use_comm_delay": True,
        "comm_delay_cfg": {
            "tau_max": 20,                          # max delay steps (0.4s at 50Hz)
            "blackout_prob": 0.1,                   # per-episode blackout probability
            "blackout_duration_range": (50, 150),   # steps (1-3s at 50Hz)
        },

        # Action chunking (Phase 3b)
        "use_action_chunking": True,
        "action_chunk_K": 8,                        # steps per chunk

        # Lead pursuit curriculum (Phase 4)
        "use_curriculum": True,
        "curriculum_ema_alpha": 0.05,
        "curriculum_advance_threshold": 0.7,
        "intercept_threshold": 1.0,        # meters, success criterion
        "intercept_sigma": 2.0,            # sigma for intercept reward shaping

        # Visual observation (Phase 5)
        "use_visual_obs": False,
        "visual_cfg": {
            "res": (64, 64),
            "fov": 90.0,
            "frame_stack": 2,
            "camera_offset_pos": [0.05, 0.0, 0.02],
            "camera_offset_lookat": [1.0, 0.0, 0.0],
            "render_every_n": 1,
        },

        # Training
        "simulate_action_latency": False,
        "action_latency_steps":    1,
        "clip_actions":            1.0,
        "domain_rand":             True,
        "dr_kf_scale":             0.3,

        # Visualisation
        "visualize_target":    False,   # sphere target unused; evader is a drone entity
        "visualize_camera":    False,
        "max_visualize_FPS":   60,
        "substeps":            2,
        "num_rendered_envs":   10,
    }

    obs_cfg = {
        "obs_scales": {
            "rel_pos": 1.0 / 3.0,
            "rel_vel": 1.0 / 3.0,
            "lin_vel": 1.0 / 3.0,
            "ang_vel": 1.0 / 3.14159,
        },
    }

    reward_cfg = {
        "yaw_lambda": -10.0,
        "reward_scales": {
            "tracking":  0.0,    # disabled in Phase 4 (replaced by intercept)
            "intercept": 1.0,    # exp(-d_intercept/σ) lead pursuit (Phase 4)
            "closing":   0.3,    # delta-distance bonus
            "energy":   -0.01,   # acceleration penalty (Phase 4)
            "smooth":   -1e-1,   # action smoothness penalty
            "angular":  -1e-3,   # body-rate penalty
            "crash":    -10.0,   # terminal crash penalty
        },
    }

    command_cfg = {
        # Evader initial spawn range (world frame, relative to origin)
        "tgt_init_x_range": [-2.0, 2.0],
        "tgt_init_y_range": [-2.0, 2.0],
        "tgt_init_z_range": [ 1.0, 3.0],

        # Waypoint sampling range
        "wp_x_range": [-3.0, 3.0],
        "wp_y_range": [-3.0, 3.0],
        "wp_z_range": [ 0.5, 4.0],

        # Evader motion dynamics
        "tgt_max_speed": 2.0,    # m/s
        "tgt_max_acc":   2.0,    # m/s²
        "tgt_max_jerk":  5.0,    # m/s³

        # Waypoint hold duration (steps)
        "wp_min_steps": 100,     # ~2 s at 50 Hz
        "wp_max_steps": 300,     # ~6 s at 50 Hz

        # Curriculum target behavior (Phase 4)
        "heading_change_min_steps": 150,   # 3 s at 50 Hz
        "heading_change_max_steps": 250,   # 5 s at 50 Hz
        "evasion_gain": 1.5,               # m/s², reactive evasion strength
        "evasion_blend": 0.5,              # blend ratio: evasion vs waypoint
        "curriculum_speed_scales": [1.0, 1.2, 1.5],  # per-stage speed multiplier
    }

    # Dynamic num_actions: 4*K when action chunking enabled, else 4
    if env_cfg.get("use_action_chunking", False):
        env_cfg["num_actions"] = 4 * env_cfg["action_chunk_K"]

    return env_cfg, obs_cfg, reward_cfg, command_cfg
