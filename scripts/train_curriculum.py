"""Phase C — Curriculum scratch wrapper.

Plan: docs/plans/phase_c_curriculum_scratch.md (v2).

각 curriculum stage를 ``scripts/train_ams_drl.py --max_cycles 1`` 호출로 실행한다.
Stage K 산출 ckpt를 stage K+1의 ``--*_ckpt`` / ``--*_resume`` 로 자동 인계한다.
Opponent curriculum은 ``train_ams_drl.py``의 alternating 구조가 자연스럽게 처리:
  - stage 0 (scratch): cycle 1 attacker = vs stationary, cycle 1 defender = vs new attacker
  - stage 1+ (pretrain): cycle 1 attacker = vs stage K-1 defender, vice versa

Per-stage 진급 게이트는 ``train_ams_drl.py`` 의 ``--attacker_min_winrate`` /
``--defender_min_winrate`` + ``--per_stage_max_rounds`` 로 위임한다. 이 wrapper는
stage 종료 후 cycle_001/h2h.json만 읽어 게이트 통과 여부를 alarm한다 (실패 시 abort).

Usage::

    python scripts/train_curriculum.py \\
        --logs_root logs/pe1v1_v0/curriculum_v1 \\
        --num_envs 1024 --seed 0 --logger wandb

    # smoke (cpu, 1 stage 1 cycle, switch_every=2):
    python scripts/train_curriculum.py --smoke --logs_root /tmp/curr_smoke --seed 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    id: int
    env_yaml: str               # relative to ROOT
    switch_every: int           # PPO iters per role per cycle
    attacker_min_winrate: float
    defender_min_winrate: float
    per_stage_max_rounds: int
    max_cycles: int = 1         # alternating cycles within stage
    use_fp: bool = False        # Phase D (2026-05-07) — Fictitious Play (stage 4 only)


# Plan v4 spec (사용자 2026-05-05). 5 stages — stage 4 = AMS 본 학습.
# stage 0: scratch + max_cycles=2 (cycle 1 vs stationary, cycle 2 vs def_v1)
# stage 1~3: pretrain + max_cycles=2 (alternating 안정성)
# stage 4: AMS env, max_cycles=5 (사용자 명시 — 본 학습)
STAGES: tuple[Stage, ...] = (
    # v9 (2026-05-05) — stage 0 둘 다 약하게 학습 (mission ~0.5, capture ~0.5),
    # stage 1~3 게이트 완화 (alternating 학습 시간 확보), stage 4는 AMS 엄격.
    # v10 (2026-05-05): stage 1~3 attacker 게이트 비활성. wrapper의 cycle h2h.json은
    # 마지막 b_check (= defender 학습 직후 attacker vs defender) 결과인데, attacker_vK.pt
    # 자체는 a_check max에서 mission 0.5+ 학습되어도 b_check에선 defender over-fit으로
    # 0%로 보임. ckpt 자체는 학습됐으므로 attacker 게이트 비활성, defender 균형만 검사.
    Stage(0, "configs/envs/pe_1v1_curriculum/stage_0.yaml", 200, 0.0, 0.3, 3, 1),
    Stage(1, "configs/envs/pe_1v1_curriculum/stage_1.yaml", 200, 0.0, 0.3, 3, 2),
    Stage(2, "configs/envs/pe_1v1_curriculum/stage_2.yaml", 200, 0.0, 0.3, 3, 2),
    # v15 (2026-05-06 사용자 결정): stage 4 본 학습으로 복귀. stage 3은 통과만, stage 4
    # max_cycles=10으로 alternating 길게 → Nash 수렴성 검증. v14 stage 3 결과 (cycle 3
    # peak 0.253 후 collapse) → stage 4 큰 환경에서 더 긴 alternating 시도.
    Stage(3, "configs/envs/pe_1v1_curriculum/stage_3.yaml", 200, 0.0, 0.3, 3, 2),   # v15: 통과만
    Stage(4, "configs/envs/pe_1v1_curriculum/stage_4.yaml", 200, 0.4, 0.4, 5, 10, use_fp=True),  # v15: max_cycles 10, max_rounds 5; Phase D FP 활성
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase C — curriculum scratch wrapper")
    p.add_argument("--logs_root", type=str, required=True,
                   help="<logs_root>/stage_<K>/ 아래에 각 stage 산출물 저장.")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--logger", type=str, default="wandb",
                   choices=["wandb", "tensorboard"])
    p.add_argument("--start_stage", type=int, default=0,
                   help="Resume from this stage. Stage K-1 ckpt가 logs_root/stage_<K-1>/ 에 "
                        "있어야 함. 0=scratch.")
    p.add_argument("--end_stage", type=int, default=4,
                   help="Inclusive. v15 (2026-05-06): 4=AMS 본 학습 + max_cycles 10으로 Nash 검증.")
    p.add_argument("--attacker_train_yaml", type=str,
                   default="configs/train/ppo_attacker_tanhm.yaml")
    p.add_argument("--defender_train_yaml", type=str,
                   default="configs/train/ppo_defender_tanhm.yaml")
    p.add_argument("--n_h2h_episodes", type=int, default=400)
    p.add_argument("--h2h_num_envs", type=int, default=256)
    p.add_argument("--use_asymmetric_critic", action="store_true",
                   help="Phase E.1 (CLAUDE.md §7, 2026-05-08). 자식 train_ams_drl + "
                        "train_attacker/defender에 --use_asymmetric_critic 전달.")
    p.add_argument("--smoke", action="store_true",
                   help="Override: 1 stage(=start_stage), switch_every=2, cpu, num_envs=16, "
                        "n_h2h=8, h2h_num_envs=8, logger=tensorboard.")
    return p.parse_args()


def _stage_dir(logs_root: Path, k: int) -> Path:
    return logs_root / f"stage_{k}"


def _prev_artifacts(logs_root: Path, k: int) -> dict[str, Path]:
    """Return ckpt + resume paths produced by stage K (last cycle's outputs).

    train_ams_drl writes ``attacker_v{cycle}.pt`` / ``model_attacker_v{cycle}.ckpt``
    per cycle into the stage logs_root. With ``max_cycles>1`` we want the last.
    """
    sd = _stage_dir(logs_root, k)
    att_cands = sorted(sd.glob("attacker_v*.pt"))
    def_cands = sorted(sd.glob("defender_v*.pt"))
    a_resume_cands = sorted(sd.glob("model_attacker_v*.ckpt"))
    d_resume_cands = sorted(sd.glob("model_defender_v*.ckpt"))
    if not (att_cands and def_cands and a_resume_cands and d_resume_cands):
        raise SystemExit(
            f"[curriculum] stage_{k} 산출물 누락 (attacker/defender ckpt or resume). dir={sd}"
        )
    return {
        "attacker_ckpt": att_cands[-1],
        "defender_ckpt": def_cands[-1],
        "attacker_resume": a_resume_cands[-1],
        "defender_resume": d_resume_cands[-1],
    }


def _build_stage_cmd(
    py: str, train_ams: str, st: Stage, args: argparse.Namespace,
    stage_logs_root: Path, prev: dict[str, Path] | None,
) -> list[str]:
    cmd = [
        py, train_ams,
        "--env_yaml", str(ROOT / st.env_yaml),
        "--attacker_train_yaml", str(ROOT / args.attacker_train_yaml),
        "--defender_train_yaml", str(ROOT / args.defender_train_yaml),
        "--num_envs", str(args.num_envs),
        "--switch_every", str(st.switch_every),
        "--max_cycles", str(st.max_cycles),
        "--n_h2h_episodes", str(args.n_h2h_episodes),
        "--h2h_num_envs", str(args.h2h_num_envs),
        "--attacker_min_winrate", str(st.attacker_min_winrate),
        "--defender_min_winrate", str(st.defender_min_winrate),
        "--per_stage_max_rounds", str(st.per_stage_max_rounds),
        # 2026-05-06: Nash 수렴 검출 활성화 — |att-def|<0.05 + draw<0.2 + last 3 cycle 연속.
        # stage 3에서 max_cycles=10 동안 진정 수렴 시 early stop (시간 절약).
        # 진동만 하면 max_cycles 끝까지 진행 → 시계열로 진동 폭 분석.
        "--convergence_window", "3",
        "--convergence_eps", "0.05",
        "--draw_max", "0.2",
        "--seed", str(args.seed),
        "--backend", args.backend,
        "--logger", args.logger,
        "--logs_root", str(stage_logs_root),
    ]
    if st.use_fp:
        cmd += ["--use_fp"]   # Phase D — Fictitious Play (stage 4 only by spec)
    if args.use_asymmetric_critic:
        cmd += ["--use_asymmetric_critic"]   # Phase E.1 — 모든 stage에 전달
    if prev is None:
        cmd += ["--train_mode", "scratch"]
    else:
        cmd += [
            "--train_mode", "pretrain",
            "--attacker_ckpt", str(prev["attacker_ckpt"]),
            "--defender_ckpt", str(prev["defender_ckpt"]),
            "--attacker_resume", str(prev["attacker_resume"]),
            "--defender_resume", str(prev["defender_resume"]),
        ]
    return cmd


def _check_promote_gate(stage_logs_root: Path, st: Stage) -> tuple[bool, dict]:
    """Read last cycle's h2h.json + verify both winrates ≥ stage thresholds.

    With ``max_cycles>1`` train_ams_drl may converge early or run all cycles —
    pick the highest cycle_NNN/h2h.json that exists.
    """
    cycle_dirs = sorted(p for p in stage_logs_root.glob("cycle_*") if p.is_dir())
    if not cycle_dirs:
        raise SystemExit(f"[curriculum] no cycle_* dirs under {stage_logs_root}")
    h2h_path = cycle_dirs[-1] / "h2h.json"
    if not h2h_path.is_file() or h2h_path.stat().st_size == 0:
        raise SystemExit(f"[curriculum] missing h2h.json: {h2h_path}")
    h = json.loads(h2h_path.read_text())
    att_ok = h["attacker_winrate"] >= st.attacker_min_winrate
    def_ok = h["defender_winrate"] >= st.defender_min_winrate
    return (att_ok and def_ok), h


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.backend = "cpu"
        args.num_envs = 16
        args.h2h_num_envs = 8
        args.n_h2h_episodes = 8
        args.logger = "tensorboard"   # wandb api key 없는 환경에서 멈추지 않게
        args.end_stage = args.start_stage   # one-stage smoke
        # smoke에선 switch_every / max_rounds도 강제로 줄임 (Stage frozen → 새 instance)
        global STAGES   # noqa: PLW0603 — module-level reassign for smoke override
        STAGES = tuple(
            Stage(s.id, s.env_yaml, 2, s.attacker_min_winrate,
                  s.defender_min_winrate, 1)
            for s in STAGES
        )

    logs_root = Path(args.logs_root).resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    if not (0 <= args.start_stage <= 4 and 0 <= args.end_stage <= 4):
        raise SystemExit("[curriculum] start_stage / end_stage must be in 0..4")
    if args.start_stage > args.end_stage:
        raise SystemExit("[curriculum] start_stage > end_stage")

    py = sys.executable
    train_ams = str(ROOT / "scripts" / "train_ams_drl.py")

    print(f"[curriculum] logs_root={logs_root}", flush=True)
    print(f"[curriculum] running stages {args.start_stage}..{args.end_stage}", flush=True)

    for st in STAGES:
        if st.id < args.start_stage or st.id > args.end_stage:
            continue
        sd = _stage_dir(logs_root, st.id)
        sd.mkdir(parents=True, exist_ok=True)
        prev = _prev_artifacts(logs_root, st.id - 1) if st.id > 0 else None

        cmd = _build_stage_cmd(py, train_ams, st, args, sd, prev)
        print(f"\n[curriculum] === STAGE {st.id} ({st.env_yaml}) ===", flush=True)
        print(f"[curriculum] $ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            print(f"[curriculum] ✗ stage_{st.id} train_ams_drl rc={rc}", file=sys.stderr)
            return rc

        passed, h = _check_promote_gate(sd, st)
        att = h["attacker_winrate"]
        dfn = h["defender_winrate"]
        drw = h["draw_rate"]
        verdict = "✓ PASS" if passed else "✗ FAIL"
        print(
            f"[curriculum] stage_{st.id} {verdict} — "
            f"att={att:.3f} (≥{st.attacker_min_winrate}) "
            f"def={dfn:.3f} (≥{st.defender_min_winrate}) draw={drw:.3f}",
            flush=True,
        )
        if not passed:
            print(
                f"[curriculum] ABORT — stage_{st.id} 진급 게이트 fail. "
                f"plan §7 rollback: per_stage_max_rounds 5로 늘리거나 보간 stage 추가 검토.",
                file=sys.stderr,
            )
            return 2

    print(f"\n[curriculum] ✓ stages {args.start_stage}..{args.end_stage} 졸업", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
