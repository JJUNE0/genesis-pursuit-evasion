"""PR-N sanity — ``train_ams_drl.py --smoke`` end-to-end (2 cycles, CPU).

Standalone script (NOT pytest — wall time on CPU is minutes due to nested
Genesis subprocess inits). Invoke directly:

    python tests/sanity/ams_drl_minimal.py
    python tests/sanity/ams_drl_minimal.py --keep   # keep tmpdir for inspection

Pass criteria (CCG round 1+2+3 reviewed):
  1. orchestrator returncode == 0; both cycles run to completion.
  2. attacker_v1.pt, attacker_v2.pt, defender_v1.pt, defender_v2.pt all exist
     and are byte-distinct (cycle 1 vs cycle 2 produced different files).
  3. cycle_001/h2h.json + cycle_002/h2h.json valid JSON with
     att_winrate + def_winrate + draw_rate ≈ 1.0 (atol=0.02 — Wilson rounding).
  4. attacker actor parameters: cycle 1 vs cycle 2 not bit-identical.
     CCG round 3 F3 — exact ``torch.equal`` non-identity check, NOT
     ``allclose`` magnitude (PPO with switch_every=2 may produce small
     deltas inside float32 noise; allclose would false-pass on no-learning).
  5. cycle_001/attacker_train/<run>/cfgs.pkl exists (smoke produces a
     resumable model + cfgs).

Synthetic v0 actor seeds: cycle 1A trains attacker using only the
synthetic v0 defender as PretrainedDefender opponent, so that needs to be
loadable. The v0 attacker is only existence-checked at preflight (its file
is never actually loaded by the orchestrator's subprocess flow — cycle 1B
uses the freshly-trained attacker_v1.pt instead). We still synthesize both
for symmetry.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pe_1v1.cfgs import (  # noqa: E402
    ATTACKER_OBS_DIM,
    DEFENDER_OBS_DIM,
    NUM_ACTIONS,
)
from tests.sanity._smoke_actor import SmokeActor  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic v0 actor — class lives in tests/sanity/_smoke_actor.py so that
# torch.save's pickle qualname resolves in subprocess (train_attacker.py)
# whose __main__ is the train script, not this sanity script.
# ---------------------------------------------------------------------------


def _save_synthetic_v0(out_dir: Path, role: str, obs_dim: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    actor = SmokeActor(obs_dim=obs_dim, num_actions=NUM_ACTIONS)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    path = out_dir / f"{role}_v0.pt"
    torch.save(actor, path)
    return path


def _bytes_distinct(p1: Path, p2: Path) -> bool:
    return p1.read_bytes() != p2.read_bytes()


def _params_changed(p1: Path, p2: Path) -> bool:
    """CCG F3 — exact non-identity (torch.equal), NOT allclose."""
    a1 = torch.load(p1, map_location="cpu", weights_only=False)
    a2 = torch.load(p2, map_location="cpu", weights_only=False)
    sd1 = a1.state_dict()
    sd2 = a2.state_dict()
    common = set(sd1.keys()) & set(sd2.keys())
    if not common:
        return False
    return any(not torch.equal(sd1[k], sd2[k]) for k in common)


def main() -> int:   # noqa: PLR0911, PLR0912
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true",
                        help="Keep tmpdir for inspection on success.")
    args = parser.parse_args()

    tmpdir = Path(tempfile.mkdtemp(prefix="ams_drl_smoke_"))
    print(f"[smoke] workspace: {tmpdir}", flush=True)

    try:
        v0_dir = tmpdir / "v0"
        v0_attacker = _save_synthetic_v0(v0_dir, role="attacker", obs_dim=ATTACKER_OBS_DIM)
        v0_defender = _save_synthetic_v0(v0_dir, role="defender", obs_dim=DEFENDER_OBS_DIM)

        logs_root = tmpdir / "ams_logs"

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_ams_drl.py"),
            "--attacker_ckpt", str(v0_attacker),
            "--defender_ckpt", str(v0_defender),
            "--seed", "0",
            "--logger", "tensorboard",
            "--logs_root", str(logs_root),
            "--smoke",
        ]
        print(f"[smoke] $ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            print(f"[FAIL] orchestrator rc={rc}", file=sys.stderr)
            return 1

        # --- Criterion 2: actor + defender ckpts present and distinct ---
        a_v1 = logs_root / "attacker_v1.pt"
        a_v2 = logs_root / "attacker_v2.pt"
        d_v1 = logs_root / "defender_v1.pt"
        d_v2 = logs_root / "defender_v2.pt"
        for p in (a_v1, a_v2, d_v1, d_v2):
            if not p.is_file() or p.stat().st_size == 0:
                print(f"[FAIL] expected ckpt missing/empty: {p}", file=sys.stderr)
                return 1
        if not _bytes_distinct(a_v1, a_v2):
            print("[FAIL] attacker_v1.pt == attacker_v2.pt (byte-identical)", file=sys.stderr)
            return 1
        if not _bytes_distinct(d_v1, d_v2):
            print("[FAIL] defender_v1.pt == defender_v2.pt (byte-identical)", file=sys.stderr)
            return 1

        # --- Criterion 3: h2h.json valid + winrate sums ≈ 1.0 ---
        for cycle in (1, 2):
            jpath = logs_root / f"cycle_{cycle:03d}" / "h2h.json"
            if not jpath.is_file():
                print(f"[FAIL] missing h2h.json: {jpath}", file=sys.stderr)
                return 1
            with open(jpath) as f:
                h = json.load(f)
            wr_sum = h["attacker_winrate"] + h["defender_winrate"] + h["draw_rate"]
            if abs(wr_sum - 1.0) > 0.02:
                print(
                    f"[FAIL] cycle {cycle} winrate sum = {wr_sum:.4f} (expected ≈ 1.0)",
                    file=sys.stderr,
                )
                return 1
            for key in ("wilson_ci_95", "breakdown", "git_sha", "seed"):
                if key not in h:
                    print(f"[FAIL] cycle {cycle} h2h.json missing key: {key}",
                          file=sys.stderr)
                    return 1

        # --- Criterion 4: attacker actor params changed cycle 1 → cycle 2 ---
        # CCG F3 — torch.equal exact non-identity, no allclose tolerance.
        if not _params_changed(a_v1, a_v2):
            print(
                "[FAIL] attacker actor params bit-identical between cycle 1 and "
                "cycle 2 — no learning happened. Smoke training step is broken "
                "or runner.load() restored snapshot mid-training.",
                file=sys.stderr,
            )
            return 1

        # --- Criterion 5: cfgs.pkl produced ---
        cfgs_glob = list((logs_root / "cycle_001" / "attacker_train").rglob("cfgs.pkl"))
        if not cfgs_glob:
            print(
                f"[FAIL] no cfgs.pkl under {logs_root}/cycle_001/attacker_train",
                file=sys.stderr,
            )
            return 1

        print(
            "\n[PASS] ams_drl_minimal smoke — 2 cycles, distinct ckpts, valid h2h, "
            "params changed.",
            flush=True,
        )
        return 0

    finally:
        if not args.keep:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
        else:
            print(f"[smoke] kept workspace at {tmpdir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
