# Genesis Pursuit-Evasion (1v1)

Attacker-PoV reinforcement learning for 1v1 quadrotor pursuit-evasion with three coupled contributions:

1. Goal-conditioned attacker policy (PETN-style) on a Genesis batched simulator.
2. Asymmetric τ-step communication delay enforced **inside the environment** (agent never sees the buffer).
3. Direct transformer forecaster `f_ψ` plus an asymmetric privileged critic `Q_θ` co-trained via AMS-DRL.

> Status: Phase 0 — repo bootstrap. See [TODO.md](./TODO.md) for the active roadmap and [CLAUDE.md](./CLAUDE.md) for fixed engineering rules.

## Layout

```
agents/        forecaster, critic, orchestrator, ppo runner
envs/          base_pe_env + 1v1 task + delay buffer + coord wrappers
controllers/   PID / SE3 (defender scripted policy material)
configs/       drones, envs, ams_drl stages, train hyperparams
assets/        URDF + meshes (Nova, cf2x)
scripts/       train_attacker, train_defender, train_ams_drl, eval, sanity
tests/         shape contract, asymmetry contract, sanity
reference/     legacy single-agent envs (read-only, not imported)
paper/, docs/  research notes (read-only for the agent)
```

## Quick start (placeholder until Phase 1 lands)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest -q
```

## Citation

```bibtex
@misc{genesis_pursuit_evasion_2026,
  title  = {Asymmetric Pursuit-Evasion under Communication Delay with Direct Forecasting and Privileged Critic},
  author = {TBD},
  year   = {2026},
  note   = {Preprint in preparation}
}
```
