"""Minimal actor module for ``ams_drl_minimal.py`` smoke test.

**Why this is a separate file:** ``torch.save(actor)`` pickles the actor's
class qualname (``module.ClassName``). When the smoke test saves a
synthetic v0 defender at ``ams_drl_minimal.py`` and the orchestrator's
``train_attacker.py`` subprocess later ``torch.load``s it via
``PretrainedDefender(...)``, pickle imports the class by its stored qualname.
If the class is defined in the script's ``__main__`` the subprocess's
``__main__`` is ``train_attacker.py`` and the import fails (
``AttributeError: Can't get attribute 'SmokeActor' on <module '__main__' …>``).

Defining the class here, in an importable module under ``ROOT/tests/sanity/``,
gives it a stable qualname ``tests.sanity._smoke_actor.SmokeActor`` that any
subprocess with ROOT on ``sys.path`` can resolve.
"""

from __future__ import annotations

import torch
from torch import nn


class SmokeActor(nn.Module):
    """Minimal forward-pass actor for the AMS-DRL smoke test.

    Mirrors the rsl-rl-saved actor module's contract just enough that
    ``PretrainedAttacker`` / ``PretrainedDefender`` can load and call it:
      - has an integer ``obs_dim`` attribute (validated at load time)
      - ``forward(td)`` consumes ``TensorDict({"policy": ...})`` and returns
        an action tensor in [-1, 1].
    """

    def __init__(self, obs_dim: int, num_actions: int = 4):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.linear = nn.Linear(self.obs_dim, num_actions)

    def forward(self, td):  # type: ignore[no-untyped-def]
        return torch.tanh(self.linear(td["policy"]))
