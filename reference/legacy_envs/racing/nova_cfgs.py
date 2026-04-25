"""Nova RacingEnv 설정"""

DRONE_CONFIG = "configs/nova/drone_params.yaml"


def get_cfgs() -> tuple:
    env_cfg = {
        "num_actions": 4,
        "actuator_mode": "rpm",
        "termination_if_roll_greater_than": 80,
        "termination_if_pitch_greater_than": 80,
        "termination_if_close_to_ground": 0.1,
        "base_init_pos": [0.0, 0.0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 30.0,
        "simulate_action_latency": False,
        "action_latency_steps": 1,
        "clip_actions": 1.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        # 물리 안정성: substeps 증가 → 유효 timestep 감소
        "substeps": 8,
        # 게이트 설정
        "gate_half_width": 0.6,    # 게이트 통과 판정 폭 [m]
        "gate_half_height": 0.6,   # 게이트 통과 판정 높이 [m]
        "gate_pass_margin": 0.3,   # 법선 방향 통과 판정 여유 [m]
        "workspace_margin": 1.5,
        "progress_sigma": 2.0,   # exp(-dist/σ) 거리 척도 [m]
    }
    obs_cfg = {
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "gate_pass":  10.0,   # 게이트 통과 보상
            "progress":    2.0,   # exp(-dist/σ) 근접 보상 (항상 양수, 충돌전략 차단)
            "crash":     -10.0,
            "smooth":     -1e-2,  # -5e-2 → -1e-2: 페널티 완화 (생존 유인 강화)
            "angular":    -1e-4,  # -1e-3 → -1e-4: 페널티 완화
        },
    }
    command_cfg = {
        # 게이트 위치 (world frame, 에피소드별 랜덤 생성 또는 고정)
        # fixed_gates: 고정 코스, 없으면 랜덤 생성
        "num_gates": 4,
        "gate_x_range": [-2.0, 2.0],
        "gate_y_range": [-2.0, 2.0],
        "gate_z_range": [0.8, 1.8],
        "min_gate_distance": 1.5,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg
