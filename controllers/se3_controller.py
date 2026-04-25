"""
SE3 Geometric Controller for Nova Drone (Genesis AI)

Reference:
    Lee, Leok, McClamroch (2010),
    "Geometric Tracking Control of a Quadrotor UAV on SE(3)"
    https://arxiv.org/abs/1003.2005

    IsaacSimulator-devel/scripts/Nova_isaaclab_standalone.py (인터페이스 참고)

입력: 현재 pose/twist, 목표 pose
출력: 모터별 추력 [N], shape (4,)  → DroneParams.pwm_to_thrust 역산으로 PWM 변환 가능

Motor order (X-frame):
    0: front_left  ( x= L,  y= L, CCW, spin=+1)
    1: front_right ( x= L,  y=-L, CW,  spin=-1)
    2: back_left   ( x=-L,  y= L, CW,  spin=-1)
    3: back_right  ( x=-L,  y=-L, CCW, spin=+1)
"""

import numpy as np
from utils.drone_params import DroneParams


class SE3Controller:
    """
    SE(3) 기하학적 제어기.

    Usage:
        ctrl = SE3Controller(params, dt=0.01)
        ctrl.set_pose(pos, quat)        # 현재 위치/자세
        ctrl.set_twist(lin_vel, ang_vel)  # 현재 속도/각속도
        ctrl.set_target(pos, yaw=0.0)   # 목표 위치/yaw
        thrusts = ctrl.compute()        # shape (4,) [N]
    """

    # 기본 게인 (CF2X 기준, Nova 튜닝 필요)
    # CF2X: Ixx=1.4e-5 kg·m², arm=0.0323m, F_max=0.160N
    # 최대 pitch 토크 ≈ 2 × 0.160 × 0.0323 ≈ 0.010 Nm
    # k_R * e_R_max(≈0.3rad) ≤ 0.010 Nm → k_R ≤ 0.033
    # 임계 제동: k_Omega ≈ 2*sqrt(k_R * Ixx) = 2*sqrt(0.015 * 1.4e-5) ≈ 0.003
    DEFAULT_GAINS = dict(
        k_pos  = np.array([4.0, 4.0, 6.0]),    # 위치 비례 게인
        k_vel  = np.array([2.0, 2.0, 4.0]),    # 속도 미분 게인
        k_R    = 0.015,                         # 자세 오차 게인 (3.0 → 0.015)
        k_Omega= 0.003,                         # 각속도 오차 게인 (0.3 → 0.003)
    )

    def __init__(self, params: DroneParams, dt: float, gains: dict = None):
        """
        Parameters
        ----------
        params : DroneParams
        dt     : float, 제어 주기 [s]
        gains  : dict, 게인 오버라이드 (None이면 DEFAULT 사용)
        """
        self.params = params
        self.dt = dt
        self.mass = params.mass
        self.gravity = params.gravity
        self.J = np.diag(params.inertia.astype(np.float64))  # (3,3) inertia tensor

        g = gains if gains is not None else self.DEFAULT_GAINS
        self.k_pos   = np.array(g["k_pos"],   dtype=np.float64)
        self.k_vel   = np.array(g["k_vel"],   dtype=np.float64)
        self.k_R     = float(g["k_R"])
        self.k_Omega = float(g["k_Omega"])

        # 추력 배분 행렬 구성 (4x4)
        self._B = self._build_allocation_matrix()
        self._B_inv = np.linalg.inv(self._B)

        # 상태 초기화
        self._pos   = np.zeros(3)
        self._vel   = np.zeros(3)
        self._R     = np.eye(3)
        self._Omega = np.zeros(3)

        # 목표
        self._pos_des = np.zeros(3)
        self._vel_des = np.zeros(3)
        self._acc_des = np.zeros(3)
        self._yaw_des = 0.0

    # ------------------------------------------------------------------
    # Public: 상태 / 목표 설정
    # ------------------------------------------------------------------

    def set_pose(self, pos: np.ndarray, quat: np.ndarray):
        """
        Parameters
        ----------
        pos  : np.ndarray (3,)  [m]  ENU 월드 프레임
        quat : np.ndarray (4,)  [w, x, y, z]
        """
        self._pos = np.array(pos, dtype=np.float64)
        self._R   = self._quat_to_rot(np.array(quat, dtype=np.float64))

    def set_twist(self, lin_vel: np.ndarray, ang_vel: np.ndarray):
        """
        Parameters
        ----------
        lin_vel : np.ndarray (3,)  [m/s]  ENU 월드 프레임
        ang_vel : np.ndarray (3,)  [rad/s] 바디 프레임
        """
        self._vel   = np.array(lin_vel, dtype=np.float64)
        self._Omega = np.array(ang_vel, dtype=np.float64)

    def set_target(self, pos_des: np.ndarray, yaw_des: float = 0.0,
                   vel_des: np.ndarray = None, acc_des: np.ndarray = None):
        """
        Parameters
        ----------
        pos_des : np.ndarray (3,)  목표 위치 [m]
        yaw_des : float  목표 yaw [rad]
        vel_des : np.ndarray (3,)  목표 속도 (기본 0)
        acc_des : np.ndarray (3,)  feedforward 가속도 (기본 0)
        """
        self._pos_des = np.array(pos_des, dtype=np.float64)
        self._yaw_des = float(yaw_des)
        self._vel_des = np.zeros(3) if vel_des is None else np.array(vel_des, dtype=np.float64)
        self._acc_des = np.zeros(3) if acc_des is None else np.array(acc_des, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public: 제어 계산
    # ------------------------------------------------------------------

    def compute(self) -> np.ndarray:
        """
        SE(3) 제어 계산.

        Returns
        -------
        thrusts : np.ndarray shape (4,) [N]  모터별 추력 (클램프 적용)
        """
        e3 = np.array([0.0, 0.0, 1.0])
        g_vec = np.array([0.0, 0.0, self.gravity])

        # --- 1. 위치/속도 오차 ---
        e_pos = self._pos - self._pos_des
        e_vel = self._vel - self._vel_des

        # --- 2. 목표 가속도 (피드백 + feedforward) ---
        a_des = (
            - self.k_pos * e_pos
            - self.k_vel * e_vel
            + g_vec
            + self._acc_des
        )

        # --- 3. 총 추력 (바디 Z축 성분) ---
        F_total = self.mass * np.dot(a_des, self._R @ e3)
        F_total = np.clip(F_total, 0.0, self.params.max_thrust_per_motor * 4)

        # --- 4. 목표 자세 계산 ---
        R_des = self._compute_desired_rotation(a_des, self._yaw_des)

        # --- 5. 자세/각속도 오차 (Lee 2010 eq. 10-11) ---
        e_R = 0.5 * self._vee(R_des.T @ self._R - self._R.T @ R_des)
        e_Omega = self._Omega  # 목표 각속도 = 0 (호버링)

        # --- 6. 모멘트 (Lee 2010 Eq. 12 — gyroscopic term 포함) ---
        gyro = np.cross(self._Omega, self.J @ self._Omega)
        M = - self.k_R * e_R - self.k_Omega * e_Omega + gyro

        # --- 7. 추력 배분 ---
        wrench = np.array([F_total, M[0], M[1], M[2]])
        thrusts = self._B_inv @ wrench

        # 클램프 (물리적 한계)
        thrusts = np.clip(thrusts, 0.0, self.params.max_thrust_per_motor)
        return thrusts.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_allocation_matrix(self) -> np.ndarray:
        """
        추력 배분 행렬 B (4x4).

        B @ [F1, F2, F3, F4]^T = [F_total, tau_x, tau_y, tau_z]^T

        Row 정의 (ENU body frame, x=forward, y=left, z=up):
            Row 0: F_total = Σ F_i                        (총 추력)
            Row 1: tau_x (roll)  = Σ y_i * F_i            (Y 위치 × 추력)
            Row 2: tau_y (pitch) = Σ -x_i * F_i           (X 위치 × 추력, 부호 반전)
            Row 3: tau_z (yaw)   = Σ spin_i * c * F_i     (프로펠러 반토크)

        Note: Isaac SE3Controller는 x_f→Row1, y_f→Row2 순서를 사용하는데,
              이는 다른 body frame 정의(x_f가 roll arm을 의미)로 인한 차이임.
              본 구현은 ENU 물리 정의에 따라 y 위치가 roll에 기여함. (올바름)

        # B: shape (4, 4)
        """
        pos = self.params.prop_positions   # (4, 3)
        spin = self.params.spin_directions  # (4,)
        c = self.params.torque_thrust_ratio

        B = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            x, y, _ = pos[i]
            B[0, i] = 1.0           # 총 추력
            B[1, i] = y             # roll  토크: tau_x = Σ y_i * F_i
            B[2, i] = -x            # pitch 토크: tau_y = Σ -x_i * F_i
            B[3, i] = spin[i] * c   # yaw   토크: 프로펠러 반토크
        return B

    @staticmethod
    def _compute_desired_rotation(a_des: np.ndarray, yaw_des: float) -> np.ndarray:
        """
        목표 가속도 벡터와 yaw로부터 목표 회전 행렬 계산.
        (Lee 2010 Eq. 기반)
        """
        z_b = a_des / (np.linalg.norm(a_des) + 1e-8)  # 정규화된 추력 방향

        yaw_vec = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        x_c = yaw_vec - np.dot(yaw_vec, z_b) * z_b
        x_b = x_c / (np.linalg.norm(x_c) + 1e-8)
        y_b = np.cross(z_b, x_b)

        return np.column_stack([x_b, y_b, z_b])  # (3, 3)

    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        """
        쿼터니언 [w, x, y, z] → 회전 행렬 (3x3).
        # R: shape (3, 3)
        """
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)  ],
            [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)  ],
            [2*(x*z - w*y),       2*(y*z + w*x),      1 - 2*(x*x + y*y)],
        ])

    @staticmethod
    def _vee(skew: np.ndarray) -> np.ndarray:
        """
        Skew-symmetric 행렬 → 벡터 (vee operator).
        # skew: (3, 3), return: (3,)
        """
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
