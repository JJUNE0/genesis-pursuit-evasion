"""1v1 pursuit-evasion environments built on Genesis.

Phase 1 is a perfect-info baseline (no delay buffer, no forecaster).
Subpackages own their own imports — this top-level __init__ stays empty so
callers can import ``envs.pe_1v1.cfgs`` without dragging in Genesis.
See docs/plans/phase1_base_pe_env.md.
"""
