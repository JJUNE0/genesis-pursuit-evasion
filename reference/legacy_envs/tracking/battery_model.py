"""
Vectorized Battery Voltage Model — Phase 2a.

Models linear voltage discharge of a 4S LiPo battery pack and its
effects on motor thrust capacity and mechanical response time.

All tensors are shape (B,) operating over the batch dimension.
No Python for-loops over environments.

Physics:
    V_t = V_0 - decay_rate * step_count        (linear discharge)
    thrust_scale = V_t / V_nom                  (thrust reduction)
    tau_mech = tau_0 + kappa * (V_nom / V_t - 1) (motor lag increase)
"""

from __future__ import annotations

import torch


def _rand_float(lo: float, hi: float, shape: tuple, device: str) -> torch.Tensor:
    return (hi - lo) * torch.rand(size=shape, device=device) + lo


class BatteryModel:
    """Vectorized battery model with voltage-dependent thrust scaling and motor lag.

    Args:
        num_envs:  Batch size (number of parallel environments).
        device:    Torch device string.
        dt:        Simulation time-step [s].
        V_nom:     Nominal (fully-charged) voltage [V].  Default 16.8 (4S).
        V_min:     Cutoff voltage [V].  Default 13.2 (4S).
        decay_rate_range:  (lo, hi) voltage drop per step [V/step].
        kappa:     Motor-delay sensitivity coefficient.
        tau_0:     Base motor mechanical delay [s].
        kappa_range:  Multiplicative randomisation range for kappa (lo, hi).
        randomize: Whether to randomise battery parameters on reset.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        dt: float,
        V_nom: float = 16.8,
        V_min: float = 13.2,
        decay_rate_range: tuple[float, float] = (0.001, 0.01),
        kappa: float = 0.02,
        tau_0: float = 0.02,
        kappa_range: tuple[float, float] = (0.5, 2.0),
        randomize: bool = True,
    ):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.V_nom = V_nom
        self.V_min = V_min
        self.decay_rate_lo, self.decay_rate_hi = decay_rate_range
        self.kappa_base = kappa
        self.tau_0 = tau_0
        self.kappa_lo, self.kappa_hi = kappa_range
        self.randomize = randomize

        # Per-env buffers
        self.V_0 = torch.full((num_envs,), V_nom, device=device)
        self.V_t = torch.full((num_envs,), V_nom, device=device)
        self.decay_rate = torch.full(
            (num_envs,),
            (self.decay_rate_lo + self.decay_rate_hi) / 2.0,
            device=device,
        )
        self.kappa_val = torch.full((num_envs,), kappa, device=device)
        self.step_count = torch.zeros((num_envs,), device=device, dtype=torch.int32)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Randomise battery parameters for the specified environments."""
        n = len(envs_idx)
        if n == 0:
            return

        if self.randomize:
            self.V_0[envs_idx] = _rand_float(
                self.V_min, self.V_nom, (n,), self.device
            )
            self.decay_rate[envs_idx] = _rand_float(
                self.decay_rate_lo, self.decay_rate_hi, (n,), self.device
            )
            kappa_mult = _rand_float(self.kappa_lo, self.kappa_hi, (n,), self.device)
            self.kappa_val[envs_idx] = self.kappa_base * kappa_mult
        else:
            self.V_0[envs_idx] = self.V_nom
            self.decay_rate[envs_idx] = (self.decay_rate_lo + self.decay_rate_hi) / 2.0
            self.kappa_val[envs_idx] = self.kappa_base

        self.V_t[envs_idx] = self.V_0[envs_idx]
        self.step_count[envs_idx] = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance voltage by one time-step: V_t = V_0 - decay_rate * count."""
        self.step_count += 1
        self.V_t = (self.V_0 - self.decay_rate * self.step_count.float()).clamp_min(
            self.V_min
        )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def thrust_scale(self) -> torch.Tensor:
        """Thrust scaling factor (B,): V_t / V_nom ∈ [V_min/V_nom, 1.0]."""
        return self.V_t / self.V_nom

    def tau_mech(self) -> torch.Tensor:
        """Voltage-dependent motor time constant (B,) [s].

        tau = tau_0 + kappa * (V_nom / V_t - 1)
        Higher as voltage drops → sluggish motor response.
        """
        return self.tau_0 + self.kappa_val * (self.V_nom / self.V_t - 1.0)

    @property
    def voltage(self) -> torch.Tensor:
        """Current voltage per environment (B,)."""
        return self.V_t
