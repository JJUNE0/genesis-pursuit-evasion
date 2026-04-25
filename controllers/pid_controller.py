"""
Cascade PID Controller for Nova Drone (Genesis AI)

Reference: drone/quadcopter_controller.py (CF2x RPM 기반)
변경사항: 출력을 RPM → PWM [µs] 로 변환, DroneParams 기반 hover_pwm 사용

Cascade 구조:
    Position error → 목표 속도 (pos PID)
        → Velocity error → thrust/roll/pitch 델타 (vel PID)
            → Attitude error → yaw 델타 (att PID)
                → Mixer → 모터별 PWM [µs]
"""

import numpy as np
import torch
from genesis.utils.geom import quat_to_xyz
from utils.drone_params import DroneParams


class _PID:
    def __init__(self, kp: float, ki: float, kd: float, windup_limit: float = np.inf):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._windup_limit = windup_limit
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self._windup_limit, self._windup_limit)
        derivative = (error - self._prev_error) / dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class DronePIDController:
    """
    X-frame Nova 드론용 Cascade PID 제어기.
    출력: 모터별 PWM [µs], shape (4,)

    Motor order (X-frame):
        0: front_left  (CCW)
        1: front_right (CW)
        2: back_left   (CW)
        3: back_right  (CCW)
    """

    # 기본 PID 게인 (CF2x 기반, Nova 튜닝 필요)
    DEFAULT_PID_PARAMS = [
        # [kp,    ki,    kd,    windup_limit]
        [2.0,   0.0,   1.0,   np.inf],   # pos_x   (kd↑ 오버슈트 감소)
        [2.0,   0.0,   1.0,   np.inf],   # pos_y
        [2.0,   0.0,   3.0,   np.inf],   # pos_z
        [8.0,   0.0,   1.0,   30.0  ],   # vel_x → deg: 1m/s → ~8° tilt (windup±30°)
        [8.0,   0.0,   1.0,   30.0  ],   # vel_y
        [60.0,  0.0,   12.0,  np.inf],   # vel_z: 1m/s → 60µs → ~0.8m/s² 가속
        [8.0,   0.0,   1.5,   np.inf],   # att_roll
        [8.0,   0.0,   1.5,   np.inf],   # att_pitch
        [3.0,   0.0,   0.5,   np.inf],   # att_yaw
    ]

    def __init__(self, drone, dt: float, params: DroneParams, pid_params: list = None):
        """
        Parameters
        ----------
        drone   : genesis DroneEntity
        dt      : float, 제어 주기 [s]
        params  : DroneParams, 드론 물리 파라미터
        pid_params : list of [kp, ki, kd] x 9, None이면 DEFAULT 사용
        """
        self.drone = drone
        self.dt = dt
        self.params = params
        self.hover_pwm = params.hover_pwm

        gains = pid_params if pid_params is not None else self.DEFAULT_PID_PARAMS

        self._pid_pos_x   = _PID(*gains[0])
        self._pid_pos_y   = _PID(*gains[1])
        self._pid_pos_z   = _PID(*gains[2])
        self._pid_vel_x   = _PID(*gains[3])
        self._pid_vel_y   = _PID(*gains[4])
        self._pid_vel_z   = _PID(*gains[5])
        self._pid_att_roll  = _PID(*gains[6])
        self._pid_att_pitch = _PID(*gains[7])
        self._pid_att_yaw   = _PID(*gains[8])

    def update(self, target: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        target : np.ndarray shape (3,) or (4,)  [x, y, z] or [x, y, z, yaw_deg]

        Returns
        -------
        pwm : np.ndarray shape (4,)  [µs]
        """
        # obs: pos [3], vel [3], att [3]
        curr_pos = self.drone.get_pos().cpu().numpy().flatten()   # (3,)
        curr_vel = self.drone.get_vel().cpu().numpy().flatten()   # (3,)
        curr_att = quat_to_xyz(
            self.drone.get_quat(), rpy=True, degrees=True
        ).cpu().numpy().flatten()                                  # (roll, pitch, yaw) deg

        # --- Position → desired velocity ---
        vel_des_x = self._pid_pos_x.update(target[0] - curr_pos[0], self.dt)
        vel_des_y = self._pid_pos_y.update(target[1] - curr_pos[1], self.dt)
        vel_des_z = self._pid_pos_z.update(target[2] - curr_pos[2], self.dt)

        # --- Velocity → desired attitude angles ---
        # ENU pitch: positive pitch_del = back motors more = nose down = moves +X
        # ENU roll:  positive roll = left(Y>0) side up = moves -Y → need negative roll for +Y
        pitch_des =  self._pid_vel_x.update(vel_des_x - curr_vel[0], self.dt)
        roll_des  = -self._pid_vel_y.update(vel_des_y - curr_vel[1], self.dt)
        thrust_del =  self._pid_vel_z.update(vel_des_z - curr_vel[2], self.dt)

        # --- Attitude → roll/pitch/yaw delta ---
        target_yaw = target[3] if len(target) > 3 else 0.0
        roll_del  = self._pid_att_roll.update(roll_des  - curr_att[0], self.dt)
        pitch_del = self._pid_att_pitch.update(pitch_des - curr_att[1], self.dt)
        yaw_del   = self._pid_att_yaw.update(target_yaw - curr_att[2], self.dt)

        # --- Mixer: delta → PWM ---
        pwm = self._mixer(thrust_del, roll_del, pitch_del, yaw_del)
        return np.clip(pwm, self.params.pwm_min, self.params.pwm_max).astype(np.float32)

    def _mixer(self, thrust, roll, pitch, yaw) -> np.ndarray:
        """
        # pwm: (4,) [µs]
        YAML prop_positions / spin_directions 기반 자동 mixer.
        signs shape: (4, 4) — [thrust, roll, pitch, yaw] per motor
        """
        s = self.params.mixer_signs                     # (4, 4)
        deltas = np.array([thrust, roll, pitch, yaw], dtype=np.float32)
        return (self.hover_pwm + s @ deltas).astype(np.float32)

    def reset(self):
        for pid in [
            self._pid_pos_x, self._pid_pos_y, self._pid_pos_z,
            self._pid_vel_x, self._pid_vel_y, self._pid_vel_z,
            self._pid_att_roll, self._pid_att_pitch, self._pid_att_yaw,
        ]:
            pid.reset()
