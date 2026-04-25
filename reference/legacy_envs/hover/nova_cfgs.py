"""Nova HoverEnv 설정"""

DRONE_CONFIG = "configs/nova/drone_params.yaml"


def get_cfgs() -> tuple:
    env_cfg = {
        "num_actions": 4,
        "actuator_mode": "rpm",
        # 종료 조건 — Nova는 무겁고 커서 범위를 넓게
        "termination_if_roll_greater_than":  150,   # deg
        "termination_if_pitch_greater_than": 150,
        "termination_if_close_to_ground":    0.15,  # m
        "termination_if_x_greater_than":     3.0,
        "termination_if_y_greater_than":     3.0,
        "termination_if_z_greater_than":     3.0,
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
            "target":   5.0,    # 10.0 → 5.0: Gaussian 방식으로 바뀌므로 스케일 조정
            "smooth":  -5e-2,   # -5e-3 → -5e-2: 10x 추가 강화 (액션 변화량 여전히 큼)
            "yaw":      0.01,
            "angular": -1e-3,   # -2e-4 → -1e-3: 5x 강화
            "crash":   -10.0,
            "lin_vel": -1e-3,   # 신규: 선속도 페널티
            "hover":    2.0,    # 신규: 목표 근처 정지 보너스
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [ 1.0, 1.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg
