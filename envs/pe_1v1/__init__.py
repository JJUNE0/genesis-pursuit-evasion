"""PursuitEvasion 1v1 concrete environment package.

Avoid eager imports here so ``envs.pe_1v1.cfgs`` is importable without
loading the Genesis-dependent env module. Callers import directly:

    from envs.pe_1v1.cfgs import load_pe_1v1_cfg
    from envs.pe_1v1.env import PursuitEvasion1v1Env
"""
