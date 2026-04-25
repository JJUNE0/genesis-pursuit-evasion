"""Nova AttitudeRatesEnv 설정"""

from copy import deepcopy

from envs.hover.nova_cfgs import DRONE_CONFIG, get_cfgs as get_hover_cfgs


def get_cfgs() -> tuple:
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_hover_cfgs()
    env_cfg = deepcopy(env_cfg)
    obs_cfg = deepcopy(obs_cfg)
    reward_cfg = deepcopy(reward_cfg)
    command_cfg = deepcopy(command_cfg)

    env_cfg["num_actions"] = 4
    env_cfg["actuator_mode"] = "rpm"
    env_cfg["raw_rpm_output"] = True
    env_cfg["max_body_rate"] = 3.0
    env_cfg["rate_kp"] = [2.0, 2.0, 0.5]
    env_cfg["domain_rand"] = True
    env_cfg["dr_kf_scale"] = 0.3

    reward_cfg["reward_scales"]["smooth"] = -1e-1  # hover -5e-2 → rates -1e-1

    return env_cfg, obs_cfg, reward_cfg, command_cfg
