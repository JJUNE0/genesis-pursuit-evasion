"""Body-rate PD inner-loop controller.

RL action ``[T_norm, wx_ref, wy_ref, wz_ref]`` (4D, ∈ [-1, 1]) → motor RPMs.

Pipeline (per env step, 200 Hz at env dt = 0.005 s):

    [T_norm, w*_ref]  ─ scaling ─►  T_total (N), w_ref (rad/s)
                       ─ PD     ─►  body torque (N·m)
                       ─ mixer  ─►  per-motor thrust (N)
                       ─ √(f/kf)─►  motor RPM

Single-loop body-rate PD only (no cascade). Tuned externally — initial gains
are placeholders (CLAUDE.md §10: tuning lives in stage 2 of the plan).
"""

from __future__ import annotations

import math

import numpy as np
import torch

import genesis as gs

from utils.drone_params import DroneParams


def build_mixer_matrix(params: DroneParams) -> np.ndarray:
    """Forward mixer ``M`` (4 × n_props): wrench = M @ f.

    Convention (matches ``DroneParams.mixer_signs``):
        thrust : 1
        roll   : y_i        (y>0 motor produces +roll)
        pitch  : -x_i       (x>0 motor produces -pitch)
        yaw    : spin_i · c   c = torque_thrust_ratio = km/kf

    Returns
    -------
    M : np.ndarray  shape: (4, n_props), float64
    """
    n = int(params.n_propellers)
    pos = np.asarray(params.prop_positions, dtype=np.float64)        # (n, 3)
    spin = np.asarray(params.spin_directions, dtype=np.float64)      # (n,)
    c = float(params.torque_thrust_ratio)
    M = np.zeros((4, n), dtype=np.float64)
    M[0, :] = 1.0
    M[1, :] = pos[:, 1]
    M[2, :] = -pos[:, 0]
    M[3, :] = spin * c
    return M


class BodyRateController:
    """Batched body-rate PD inner-loop controller.

    Notes
    -----
    - All tensor ops are batched on ``num_envs`` (CLAUDE.md §14: no for-loop).
    - The PD derivative term uses a backward-difference of the measured body
      angular velocity (``(ω_t - ω_{t-1}) / dt``) — sufficient as initial
      gain; finer derivatives (filtered, command-feedforward) are tuning-time
      concerns.
    - ``last_ang_vel`` per-env state survives ``step()``; ``reset(envs_idx)``
      zeroes the named slots after env reset.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        params: DroneParams,
        device: torch.device | str,
        kp_rate: tuple[float, float, float] = (0.05, 0.05, 0.05),
        kd_rate: tuple[float, float, float] = (0.001, 0.001, 0.001),
        max_body_rate: float = math.pi,
        enable_wrench_saturation: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = device
        self.params = params

        self.kf = float(params.kf)
        self.n_props = int(params.n_propellers)
        self.max_body_rate = float(max_body_rate)
        self.enable_wrench_saturation = bool(enable_wrench_saturation)
        # T_norm = 0 → hover; hover_thrust = m·g (CLAUDE.md §5: N).
        self.hover_thrust_total = float(params.mass * params.gravity)
        self.max_thrust_per_motor = float(params.max_thrust_per_motor)
        self.max_rpm = float(math.sqrt(self.max_thrust_per_motor / self.kf))

        # Forward mixer M (4, n) and inverse used for wrench → per-motor thrust.
        # f = wrench @ M_inv_T   (B, 4) × (4, n) → (B, n)
        M = build_mixer_matrix(params)
        if M.shape[0] == M.shape[1]:
            M_inv = np.linalg.inv(M)
        else:
            M_inv = np.linalg.pinv(M)
        self.M_inv_T = torch.tensor(
            M_inv.T, device=device, dtype=gs.tc_float
        )                                                       # shape: (4, n_props)

        f = gs.tc_float
        self.kp = torch.tensor(kp_rate, device=device, dtype=f).view(1, 3)   # (1, 3)
        self.kd = torch.tensor(kd_rate, device=device, dtype=f).view(1, 3)   # (1, 3)

        # Backward-difference state for d(ω)/dt (CLAUDE.md §4: shape comments).
        self.last_ang_vel = torch.zeros(
            (self.num_envs, 3), device=device, dtype=f
        )                                                       # shape: (B, 3)

    # ------------------------------------------------------------------
    # Per-env reset (called by env.reset_idx after spawn).
    # ------------------------------------------------------------------

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        """Zero ``last_ang_vel`` for the named slots (or all if ``None``)."""
        if envs_idx is None:
            self.last_ang_vel.zero_()
            return
        if hasattr(envs_idx, "numel") and envs_idx.numel() == 0:
            return
        self.last_ang_vel[envs_idx] = 0.0

    # ------------------------------------------------------------------
    # One inner-loop tick.
    # ------------------------------------------------------------------

    def step(
        self, action_normalized: torch.Tensor, ang_vel: torch.Tensor
    ) -> torch.Tensor:
        """Compute motor RPMs for one env step.

        Parameters
        ----------
        action_normalized : torch.Tensor  shape: (B, 4)  ∈ [-1, 1]
            ``[T_norm, wx_ref, wy_ref, wz_ref]``.
        ang_vel : torch.Tensor  shape: (B, 3)  rad/s, body frame
            Current measured body angular velocity (Genesis ``drone.get_ang()``).

        Returns
        -------
        rpms : torch.Tensor  shape: (B, n_props)  in [0, max_rpm]
        """
        # NaN/Inf guard at the boundary (CLAUDE.md §4).
        action_normalized = torch.nan_to_num(
            action_normalized, nan=0.0, posinf=1.0, neginf=-1.0
        )
        ang_vel = torch.nan_to_num(ang_vel, nan=0.0, posinf=0.0, neginf=0.0)

        T_norm = action_normalized[:, 0:1]                              # (B, 1)
        rate_ref = action_normalized[:, 1:4] * self.max_body_rate       # (B, 3)

        # T_norm ∈ [-1, 1] → T_total ∈ [0, 2·hover_thrust]
        # (T_norm=0 → hover, T_norm=+1 → 2g acceleration, T_norm=-1 → free fall).
        T_total = (T_norm + 1.0) * self.hover_thrust_total              # (B, 1)

        # Body-rate PD per axis: τ = Kp·(w_ref - w) - Kd·dω/dt.
        rate_err = rate_ref - ang_vel                                   # (B, 3)
        ang_acc = (ang_vel - self.last_ang_vel) / self.dt               # (B, 3)
        torque = self.kp * rate_err - self.kd * ang_acc                 # (B, 3)

        # Snapshot current ω for next step's backward difference. Detach in
        # case ang_vel ever carries grad (it shouldn't — Genesis is no-grad).
        self.last_ang_vel[:] = ang_vel.detach()

        # Mixer: [T, τx, τy, τz] → per-motor thrust.
        # PR-N (Phase C 2026-04-29) — opt-in PX4 control allocation. Without it,
        # bang-bang commands lose ~65% wrench (controller diag: scripts/test/
        # diag_rate_controller.py). With it, torque is scaled to fit motor
        # saturation; T_total preserved (priority: T > τ).
        if self.enable_wrench_saturation:
            wrench = self._saturate_wrench(T_total, torque)             # (B, 4)
        else:
            wrench = torch.cat([T_total, torque], dim=1)                # (B, 4)
        thrusts = wrench @ self.M_inv_T                                 # (B, n_props)

        # Motors cannot produce negative thrust; cap at per-motor max as well.
        thrusts = thrusts.clamp(min=0.0, max=self.max_thrust_per_motor)

        # RPM from thrust: f = kf·ω² → ω = √(f/kf). clamp_min(0) for safety
        # (already non-negative after clamp above; explicit 0 guard for sqrt).
        rpms = torch.sqrt(thrusts.clamp_min(0.0) / self.kf)             # (B, n_props)
        rpms = rpms.clamp(min=0.0, max=self.max_rpm)
        return rpms

    # ------------------------------------------------------------------
    # PR-N — Wrench saturation (opt-in PX4-style control allocation).
    # ------------------------------------------------------------------

    def _saturate_wrench(
        self, T_total: torch.Tensor, torque: torch.Tensor
    ) -> torch.Tensor:
        """Scale torque so per-motor thrust stays in ``[0, max_thrust_per_motor]``.

        Single global scale per env (priority: T_total preserved > τ scaled).
        Hover sanity (τ=0): scale = 1 (no-op). bang-bang (τ saturating): scale<1.

        Parameters
        ----------
        T_total : (B, 1) total thrust command (N).
        torque  : (B, 3) [τx, τy, τz] body torque (N·m).

        Returns
        -------
        (B, 4) wrench [T_total, τ_scaled].
        """
        # T-only contribution per motor (B, n_props).
        wrench_t = torch.cat([T_total, torch.zeros_like(torque)], dim=1)
        thrusts_t = wrench_t @ self.M_inv_T

        # Per-motor headroom around the T-only point. Clamp_min(1e-6) avoids
        # divide-by-zero when T_total≈0 or T_total≈4·max_thrust (degenerate).
        headroom_pos = (self.max_thrust_per_motor - thrusts_t).clamp_min(1e-6)
        headroom_neg = thrusts_t.clamp_min(1e-6)

        # Torque-only contribution per motor.
        thrusts_full = (torch.cat([T_total, torque], dim=1)) @ self.M_inv_T
        thrust_tau = thrusts_full - thrusts_t

        # Per-env max violation (>1 ⇒ saturation; ≤1 ⇒ feasible).
        pos_v = (thrust_tau / headroom_pos).clamp_min(0).max(dim=1, keepdim=True).values
        neg_v = (-thrust_tau / headroom_neg).clamp_min(0).max(dim=1, keepdim=True).values
        max_v = torch.maximum(pos_v, neg_v)                              # (B, 1)

        # Scale ∈ (0, 1] — only shrinks; never amplifies.
        scale = 1.0 / max_v.clamp_min(1.0)                               # (B, 1)
        torque_scaled = torque * scale                                   # (B, 3)
        return torch.cat([T_total, torque_scaled], dim=1)                # (B, 4)
