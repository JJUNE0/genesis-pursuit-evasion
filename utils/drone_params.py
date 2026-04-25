"""
Drone parameter loader and PWM-to-Thrust interpolator.

Reference: IsaacSimulator-devel/CocelDroneSimulation/Utils/MotorPropellerProfiler.py
"""

import numpy as np
import yaml
from pathlib import Path
from scipy.interpolate import interp1d


class DroneParams:
    """
    Load drone physical parameters from YAML and provide PWM→Thrust/Torque interpolation.

    Usage:
        params = DroneParams("configs/nova/drone_params.yaml")
        thrust = params.pwm_to_thrust(np.array([1400, 1500, 1600, 1700]))  # shape: (4,)
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._cfg = self._load_yaml(self.config_path)

        drone = self._cfg["drone"]
        motor = drone["motor"]

        # --- 메타데이터 ---
        self.name: str = drone["name"]
        self.urdf_path: str = drone.get("urdf", "")
        self.frame_type: str = drone.get("frame_type", "x_frame")   # "x_frame" | "+_frame"
        self.coord_system: str = drone.get("coord_system", "enu")   # "enu" | "ned"
        self.kf: float = float(drone.get("kf", 3.16e-10))           # [N/RPM²]

        # --- 물리 파라미터 ---
        self.mass: float = drone["mass"]                        # [kg]
        self.gravity: float = self._cfg.get("gravity", 9.81)   # [m/s²]
        self.arm_length: float = drone["arm_length"]           # [m]

        inertia = drone["inertia"]
        self.inertia: np.ndarray = np.array([                  # [kg·m²]
            inertia["ixx"],
            inertia["iyy"],
            inertia["izz"],
        ], dtype=np.float32)

        # --- 프로펠러 ---
        props = drone["propellers"]
        self.prop_positions: np.ndarray = np.array(            # shape: (4, 3) [m]
            props["positions"], dtype=np.float32
        )
        self.spin_directions: np.ndarray = np.array(           # shape: (4,) +1=CCW, -1=CW
            props["spin_directions"], dtype=np.float32
        )
        self.n_propellers: int = len(self.spin_directions)
        self.prop_link_names: list = props.get(                # Genesis 링크명 리스트
            "link_names",
            [f"prop{i}_link" for i in range(self.n_propellers)]
        )

        # --- 모터 스펙 ---
        self.pwm_min: float = motor["pwm_min"]                 # [µs]
        self.pwm_max: float = motor["pwm_max"]                 # [µs]
        self.max_thrust_per_motor: float = motor["max_thrust"] # [N]
        self.torque_thrust_ratio: float = motor["torque_thrust_ratio"]  # KM/KF [m]

        # --- PWM→Thrust/Torque 보간기 구성 ---
        table = np.array(drone["pwm_thrust_table"], dtype=np.float32)  # shape: (N, 2)
        pwm_vals    = table[:, 0]   # [µs]
        thrust_vals = table[:, 1]   # [N]
        torque_vals = thrust_vals * self.torque_thrust_ratio            # [N·m]

        self._thrust_interp = interp1d(
            pwm_vals, thrust_vals,
            kind="linear",
            bounds_error=False,
            fill_value=(thrust_vals[0], thrust_vals[-1]),  # 범위 밖은 min/max로 클램프
        )
        self._torque_interp = interp1d(
            pwm_vals, torque_vals,
            kind="linear",
            bounds_error=False,
            fill_value=(torque_vals[0], torque_vals[-1]),
        )

        # --- 드론별 PID 게인 (선택적) ---
        raw = drone.get("pid_gains")
        if raw is not None:
            self.pid_gains: list | None = [
                [raw["pos_x"][0],   raw["pos_x"][1],   raw["pos_x"][2],   raw["pos_x"][3]  ],
                [raw["pos_y"][0],   raw["pos_y"][1],   raw["pos_y"][2],   raw["pos_y"][3]  ],
                [raw["pos_z"][0],   raw["pos_z"][1],   raw["pos_z"][2],   raw["pos_z"][3]  ],
                [raw["vel_x"][0],   raw["vel_x"][1],   raw["vel_x"][2],   raw["vel_x"][3]  ],
                [raw["vel_y"][0],   raw["vel_y"][1],   raw["vel_y"][2],   raw["vel_y"][3]  ],
                [raw["vel_z"][0],   raw["vel_z"][1],   raw["vel_z"][2],   raw["vel_z"][3]  ],
                [raw["att_roll"][0],  raw["att_roll"][1],  raw["att_roll"][2],  raw["att_roll"][3] ],
                [raw["att_pitch"][0], raw["att_pitch"][1], raw["att_pitch"][2], raw["att_pitch"][3]],
                [raw["att_yaw"][0],   raw["att_yaw"][1],   raw["att_yaw"][2],   raw["att_yaw"][3]  ],
            ]
        else:
            self.pid_gains = None

        # --- 호버 PWM 계산 ---
        self.hover_thrust_per_motor: float = (self.mass * self.gravity) / self.n_propellers
        self.hover_pwm: float = self._compute_hover_pwm(pwm_vals, thrust_vals)

        # --- Thrust → PWM 역산 보간기 (SE3 출력 [N] → PWM [µs]) ---
        self._thrust_to_pwm_interp = interp1d(
            thrust_vals, pwm_vals,
            kind="linear",
            bounds_error=False,
            fill_value=(pwm_vals[0], pwm_vals[-1]),  # 범위 밖은 min/max PWM 클램프
        )

        self._print_summary()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pwm_to_thrust(self, pwm: np.ndarray) -> np.ndarray:
        """
        PWM → 추력 변환 (선형 보간).

        Parameters
        ----------
        pwm : np.ndarray  shape: (n_motors,) or (n_envs, n_motors)  [µs]

        Returns
        -------
        thrust : np.ndarray  same shape as pwm  [N]
        """
        return self._thrust_interp(pwm).astype(np.float32)

    def pwm_to_torque(self, pwm: np.ndarray) -> np.ndarray:
        """
        PWM → 반토크 변환 (선형 보간).

        Parameters
        ----------
        pwm : np.ndarray  shape: (n_motors,) or (n_envs, n_motors)  [µs]

        Returns
        -------
        torque : np.ndarray  same shape as pwm  [N·m]
        """
        return self._torque_interp(pwm).astype(np.float32)

    def pwm_to_thrust_torque(self, pwm: np.ndarray):
        """
        PWM → (추력, 반토크) 동시 변환.

        Returns
        -------
        thrust : np.ndarray  [N]
        torque : np.ndarray  [N·m]
        """
        return self.pwm_to_thrust(pwm), self.pwm_to_torque(pwm)

    def action_to_pwm(self, action: np.ndarray) -> np.ndarray:
        """
        RL 액션 [-1, 1] → PWM [pwm_min, pwm_max] 선형 변환.

        Parameters
        ----------
        action : np.ndarray  shape: (n_motors,) or (n_envs, n_motors)  [-1, 1]

        Returns
        -------
        pwm : np.ndarray  same shape  [µs]
        """
        # action=0 → hover_pwm, action=-1 → pwm_min, action=+1 → pwm_max
        return (self.hover_pwm + action * (self.pwm_max - self.hover_pwm)).astype(np.float32)

    def thrust_to_pwm(self, thrust: np.ndarray) -> np.ndarray:
        """
        추력 → PWM 역산 변환 (SE3 컨트롤러 출력 [N] → ESC 입력 [µs]).

        Parameters
        ----------
        thrust : np.ndarray  shape: (n_motors,) or (n_envs, n_motors)  [N]

        Returns
        -------
        pwm : np.ndarray  same shape as thrust  [µs]
        """
        return self._thrust_to_pwm_interp(thrust).astype(np.float32)

    @property
    def mixer_signs(self) -> np.ndarray:
        """
        YAML prop_positions / spin_directions로부터 자동 계산한 mixer 부호 행렬.

        Returns
        -------
        signs : np.ndarray shape (n_propellers, 4)  [thrust, roll, pitch, yaw]
            - thrust : always +1
            - roll   : sign(y_i)    — y>0(좌)이면 +roll 기여
            - pitch  : -sign(x_i)  — x>0(전방)이면 -pitch 기여 (노즈다운 시 전방모터 감소)
            - yaw    : spin_i       — CCW(+1) 프로펠러가 +yaw 기여
        """
        signs = np.zeros((self.n_propellers, 4), dtype=np.float32)
        signs[:, 0] = 1.0                                         # thrust
        signs[:, 1] = np.sign(self.prop_positions[:, 1])          # roll
        signs[:, 2] = -np.sign(self.prop_positions[:, 0])         # pitch
        signs[:, 3] = self.spin_directions                        # yaw
        return signs

    def get_thrust_range(self):
        """(min_thrust, max_thrust) per motor [N]"""
        return 0.0, self.max_thrust_per_motor

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_hover_pwm(self, pwm_vals: np.ndarray, thrust_vals: np.ndarray) -> float:
        """hover_thrust_per_motor에 해당하는 PWM 역산."""
        hover_interp = interp1d(
            thrust_vals, pwm_vals,
            kind="linear",
            bounds_error=False,
            fill_value=(pwm_vals[0], pwm_vals[-1]),
        )
        return float(hover_interp(self.hover_thrust_per_motor))

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _print_summary(self):
        print("=" * 50)
        print(f"[DroneParams] {self._cfg['drone']['name']}")
        print(f"  mass          : {self.mass:.4f} kg")
        print(f"  inertia (diag): {self.inertia}")
        print(f"  arm_length    : {self.arm_length:.4f} m")
        print(f"  hover thrust  : {self.hover_thrust_per_motor:.4f} N/motor")
        print(f"  hover PWM     : {self.hover_pwm:.1f} µs")
        print(f"  PWM range     : [{self.pwm_min}, {self.pwm_max}] µs")
        print("=" * 50)
