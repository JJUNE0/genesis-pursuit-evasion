"""PR-N — Phase 5 AMS-DRL alternating co-training orchestrator (S2 stage).

Subprocess-based scheduler. Each cycle k (1..max_cycles) runs three children
in **strict serial** (A→B→C; never parallel — eval needs the cycle's actors):

  A) ``train_attacker.py``  vs frozen ``defender_v{k-1}.pt``
  B) ``train_defender.py``  vs frozen ``attacker_v{k}.pt``  (just produced)
  C) ``eval_head_to_head.py``  on the new pair → cycle Nash metrics

After each cycle the produced artifacts are copied into the orchestrator-
managed ``logs_root``::

    <logs_root>/                         # args.logs_root
      attacker_v1.pt, attacker_v2.pt ... # frozen actor copies
      defender_v1.pt, ...
      model_attacker_v1.ckpt, ...        # resumable runner state copies
      model_defender_v1.ckpt, ...
      cycle_001/
        attacker_train/<run>/{attacker_v0.pt, cfgs.pkl, model_*.pt}
        defender_train/<run>/{defender_v0.pt, cfgs.pkl, model_*.pt}
        h2h.json
      cycle_002/...

Convergence (Nash gate) — fires when **both** of the following hold for the
last ``window`` cycles:
  * ``|attacker_winrate - defender_winrate| < epsilon``  (payoff balance)
  * ``draw_rate < draw_max``                              (non-degenerate)

CCG review — MVP scope. Operational features (mid-run --start_cycle resume,
summary.json, stderr capture, process-group SIGINT, fresh timestamped
logs_root, VRAM cleanup) deferred to PR-N.1 after first co-training run.

Usage::

    python scripts/train_ams_drl.py \\
        --attacker_ckpt logs/pe1v1_v0/attacker_v0/<run>/attacker_v0.pt \\
        --defender_ckpt logs/pe1v1_v0/defender_v0/<run>/defender_v0.pt \\
        --attacker_resume logs/pe1v1_v0/attacker_v0/<run>/model_<n>.pt \\
        --defender_resume logs/pe1v1_v0/defender_v0/<run>/model_<n>.pt \\
        --num_envs 1024 --switch_every 50 --max_cycles 20 \\
        --n_h2h_episodes 400 --convergence_window 5 --convergence_eps 0.05 \\
        --draw_max 0.2 --seed 0 --logger wandb

    # smoke (CI / local sanity) — forces backend=cpu, num_envs=16, etc:
    python scripts/train_ams_drl.py --smoke \\
        --attacker_ckpt <v0.pt> --defender_ckpt <v0.pt> --seed 0
        
        
    python scripts/train_ams_drl.py \
        --attacker_ckpt logs/pe1v1_v0/attacker_v0/phase1.5_attacker_stationary_seed0_1777450261_2e97bec-dirty/attacker_v0.pt \
        --defender_ckpt logs/pe1v1_v0/defender_v0/phase1.0_defender_seed0_1777450253_2e97bec-dirty/defender_v0.pt \
        --attacker_resume logs/pe1v1_v0/attacker_v0/phase1.5_attacker_stationary_seed0_1777450261_2e97bec-dirty/model_1999.pt \
        --defender_resume logs/pe1v1_v0/defender_v0/phase1.0_defender_seed0_1777450253_2e97bec-dirty/model_1999.pt \
        --seed 0
        
"""

from __future__ import annotations

import argparse
import yaml
import json
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PR-N — AMS-DRL S2 alternating orchestrator")
    p.add_argument("--train_mode", type=str, default="pretrain",
                   choices=("pretrain", "scratch"),
                   help="pretrain (default): Phase 1 산출물 ckpt에서 시작. "
                        "scratch: ckpt 없이 cycle 1 attacker는 stationary defender 상대로 "
                        "fresh PPO init 학습. cycle 1 defender는 cycle 1 attacker 산출 ckpt 상대.")
    p.add_argument("--attacker_ckpt", type=str, default="",
                   help="Phase 1 산출물 attacker_v0.pt. train_mode=pretrain 시 required.")
    p.add_argument("--defender_ckpt", type=str, default="",
                   help="Phase 1 산출물 defender_v0.pt. train_mode=pretrain 시 required.")
    p.add_argument("--attacker_resume", type=str, default="",
                   help="OnPolicyRunner.load() ckpt for cycle 1 attacker. "
                        "Empty = fresh PPO init. train_mode=scratch 시 강제 빈값.")
    p.add_argument("--defender_resume", type=str, default="",
                   help="OnPolicyRunner.load() ckpt for cycle 1 defender. "
                        "Empty = fresh PPO init. train_mode=scratch 시 강제 빈값.")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--switch_every", type=int, default=50,
                   help="PPO iterations per role per cycle.")
    p.add_argument("--max_cycles", type=int, default=20)
    p.add_argument("--n_h2h_episodes", type=int, default=400,
                   help="Absolute total episodes per cycle's H2H eval.")
    p.add_argument("--h2h_num_envs", type=int, default=256)
    p.add_argument("--convergence_window", type=int, default=5)
    p.add_argument("--convergence_eps", type=float, default=0.05)
    p.add_argument("--draw_max", type=float, default=0.2)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--logger", type=str, default="wandb",
                   choices=["wandb", "tensorboard"])
    p.add_argument(
        "--logs_root", type=str,
        default=str(ROOT / "logs" / "pe1v1_v0" / "ams_s2"),
        help="Output root. Cycle artifacts go here.",
    )
    p.add_argument(
        "--env_yaml", type=str,
        default=str(ROOT / "configs" / "envs" / "pe_1v1_default.yaml"),
        help="Env config forwarded to train_attacker/train_defender. "
             "Phase A: configs/envs/pe_1v1_phase_a.yaml.",
    )
    p.add_argument(
        "--attacker_train_yaml", type=str,
        default=str(ROOT / "configs" / "train" / "ppo_attacker.yaml"),
        help="PPO train cfg for attacker child. PR-O squashed: "
             "configs/train/ppo_attacker_squash.yaml.",
    )
    p.add_argument(
        "--defender_train_yaml", type=str,
        default=str(ROOT / "configs" / "train" / "ppo_defender.yaml"),
        help="PPO train cfg for defender child. PR-O squashed: "
             "configs/train/ppo_defender_squash.yaml.",
    )
    # 2026-05-04 — Per-stage adaptive (사용자 의도 설계).
    # A) train_attacker → A.adaptive while att_winrate < attacker_min_winrate
    #    (vs frozen def_{k-1}, attacker만 강화 — 양쪽 oscillation 차단)
    # B) train_defender → B.adaptive while def_winrate < defender_min_winrate
    #    (vs frozen att_v_k, defender만 강화)
    # C) cycle h2h.json = B.adaptive 마지막 eval (def vs att 둘 다 finalized)
    p.add_argument("--attacker_min_winrate", type=float, default=0.0,
                   help="A.adaptive trigger: attacker가 frozen def_{k-1} 상대 mission "
                        "rate < 이 값이면 추가 학습. 0=비활성. 권장 0.3-0.5.")
    p.add_argument("--defender_min_winrate", type=float, default=0.0,
                   help="B.adaptive trigger: defender가 frozen att_v_k 상대 capture "
                        "rate (defender_winrate) < 이 값이면 추가 학습. 0=비활성. "
                        "권장 0.4-0.6.")
    p.add_argument("--per_stage_max_rounds", type=int, default=1,
                   help="A.adaptive / B.adaptive 각각 최대 round 수.")
    p.add_argument("--use_fp", action="store_true",
                   help="Phase D Fictitious Play (2026-05-07). 매 cycle attacker 학습 시 "
                        "이전 cycles defender ckpts 모두를 pool로 사용 (vice versa). "
                        "단일 ckpt over-fit 방지 + Nash 안정 수렴. "
                        "cycle 1 scratch attacker는 stationary 그대로.")
    p.add_argument("--pold", type=float, default=1.0,
                   help="Phase F (2026-05-10, AMSPB style) — pool sampling probability. "
                        "1.0=all envs use pool (Phase D), 0.5=AMSPB mixing. "
                        "use_fp combo. cycle K: pool=past, current_best=v_K split.")
    p.add_argument("--use_asymmetric_critic", action="store_true",
                   help="Phase E.1 (2026-05-08). Critic obs에 opponent ground truth "
                        "(vel+ang_vel+quat=10D) 추가. Actor obs는 그대로. 자식 "
                        "train_attacker/defender에 전달.")
    p.add_argument("--deterministic", action="store_true",
                   help="GPU deterministic algorithms (paper-grade reproducibility). "
                        "Default OFF — 학습 빠름. 자식 train_attacker/defender에 forward.")
    p.add_argument("--minimal_log", action="store_true",
                   help="터미널 'Mean episode {*}' extras print skip. wandb 기록 그대로. "
                        "자식 train_attacker/defender에 forward.")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="Wandb project name (run_name prefix로도 사용). 자식 forward.")
    p.add_argument("--smoke", action="store_true",
                   help="Force MVP smoke params: backend=cpu, num_envs=16, "
                        "switch_every=2, max_cycles=2, n_h2h_episodes=8.")
    return p.parse_args()


def _apply_smoke(args: argparse.Namespace) -> None:
    """CCG round 3 F1 — smoke MUST force backend=cpu (else CPU CI runner dies)."""
    args.backend = "cpu"
    args.num_envs = 16
    args.h2h_num_envs = 8
    args.switch_every = 2
    args.max_cycles = 2
    args.n_h2h_episodes = 8
    args.convergence_window = 2


# ---------------------------------------------------------------------------
# Subprocess orchestration with SIGINT handling
# ---------------------------------------------------------------------------

_child_proc: subprocess.Popen | None = None


def _sigint_handler(signum, frame):    # noqa: ARG001 — handler signature
    """MVP — terminate the single in-flight child + propagate exit. Process-
    group kill (start_new_session + os.killpg) deferred to PR-N.1."""
    if _child_proc is not None and _child_proc.poll() is None:
        try:
            _child_proc.terminate()
            _child_proc.wait(timeout=10)
        except Exception:
            try:
                _child_proc.kill()
            except Exception:
                pass
    sys.exit(130)


def _run_child(cmd: list[str]) -> int:
    """Run subprocess with handle held in module-global so SIGINT can kill it.
    Returns subprocess returncode. Failure messaging happens at the caller —
    here we just ensure the child either completes or dies on Ctrl+C."""
    global _child_proc   # noqa: PLW0603 — SIGINT handler reads this
    print(f"[ams-s2] $ {' '.join(cmd)}", flush=True)
    _child_proc = subprocess.Popen(cmd)
    try:
        _child_proc.wait()
    finally:
        rc = _child_proc.returncode
        _child_proc = None
    return rc


def _fail(msg: str, *, cycle: int) -> None:
    """Three-way validation failure: print full context + exit nonzero."""
    print(f"\n[ams-s2] ✗ FAIL at cycle {cycle}: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Artifact discovery (cycle-scoped, NOT global)
# ---------------------------------------------------------------------------

def _find_actor_pt(scope_dir: Path, prefix: str) -> Path | None:
    """Find the latest ``{prefix}_v*.pt`` strictly under ``scope_dir``.

    CCG round 1 B5 — orchestrator must restrict glob to the cycle's own
    train log_dir, never the global tree. ``scope_dir`` MUST be cycle-local.
    """
    cands = sorted(scope_dir.rglob(f"{prefix}_v*.pt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _find_model_pt(scope_dir: Path) -> Path | None:
    cands = sorted(scope_dir.rglob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _find_cfgs_pkl(scope_dir: Path) -> Path | None:
    cands = sorted(scope_dir.rglob("cfgs.pkl"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


# ---------------------------------------------------------------------------
# Phase D — Fictitious Play helpers (2026-05-07)
# ---------------------------------------------------------------------------

def _defender_args_for_attacker_train(
    use_fp: bool, defender_pool: list, cur_d_actor, pold: float = 1.0,
) -> list[str]:
    """Build defender-related CLI args for train_attacker.py.

    Phase F (pold mixing): defender_pool의 마지막을 current_best로 분리.
      - pold=1.0 (default): 기존 Phase D 동작 (전체 pool, current_best 없음).
      - pold<1.0: pool[:-1]=past, pool[-1]=current_best, --pold 전달.
    """
    if use_fp and defender_pool:
        if pold < 1.0 and len(defender_pool) >= 1:
            past_pool = defender_pool[:-1]
            current_best = defender_pool[-1]
            args = ["--defender", "pretrained_pool",
                    "--current_best_defender_ckpt", str(current_best),
                    "--pold", str(pold)]
            if past_pool:
                args += ["--defender_ckpt_pool", *(str(p) for p in past_pool)]
            else:
                # past 비어있으면 pool 인자 빼고 current_best만 (effective pold=0)
                args += ["--defender_ckpt_pool", str(current_best)]   # placeholder
            return args
        # Phase D 기존 동작 (pold=1.0)
        return ["--defender", "pretrained_pool",
                "--defender_ckpt_pool", *(str(p) for p in defender_pool)]
    if cur_d_actor is None:
        return ["--defender", "stationary"]
    return ["--defender", "pretrained", "--defender_ckpt", str(cur_d_actor)]


def _attacker_args_for_defender_train(
    use_fp: bool, attacker_pool: list, cur_a_actor, pold: float = 1.0,
) -> list[str]:
    """Mirror — Phase F pold mixing for attacker pool."""
    if use_fp and attacker_pool:
        if pold < 1.0 and len(attacker_pool) >= 1:
            past_pool = attacker_pool[:-1]
            current_best = attacker_pool[-1]
            args = ["--attacker_mode", "pretrained_pool",
                    "--current_best_attacker_ckpt", str(current_best),
                    "--pold", str(pold)]
            if past_pool:
                args += ["--attacker_ckpt_pool", *(str(p) for p in past_pool)]
            else:
                args += ["--attacker_ckpt_pool", str(current_best)]   # placeholder
            return args
        return ["--attacker_mode", "pretrained_pool",
                "--attacker_ckpt_pool", *(str(p) for p in attacker_pool)]
    return ["--attacker_mode", "pretrained", "--attacker_ckpt", str(cur_a_actor)]


def _h2h_cmd(
    py: str, eval_h2h: str,
    attacker_ckpt: Path, defender_ckpt: Path,
    output_json: Path, args, cfgs_pkl: Path | None,
) -> list[str]:
    """Build eval_head_to_head command for use in main + per-stage adaptive."""
    cmd = [
        py, eval_h2h,
        "--attacker_ckpt", str(attacker_ckpt),
        "--defender_ckpt", str(defender_ckpt),
        "--n_episodes", str(args.n_h2h_episodes),
        "--num_envs", str(args.h2h_num_envs),
        "--seed", str(args.seed),
        "--backend", args.backend,
        "--output_json", str(output_json),
    ]
    if cfgs_pkl is not None:
        cmd += ["--cfgs_pkl", str(cfgs_pkl)]
    return cmd


def _copy_to(dst: Path, src: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Pre-flight + path resolution
# ---------------------------------------------------------------------------

def _preflight(args: argparse.Namespace) -> None:
    """v0 actor ckpts must exist before cycle 1 launches (pretrain mode only).

    train_mode=scratch: ckpt 인자 무시 + 강제로 빈값 처리. cycle 1 attacker는
    --defender stationary로 학습. cycle 1 defender는 cycle 1 attacker 산출
    ckpt 상대로 학습 (정상 alternating).
    """
    if args.train_mode == "scratch":
        # 사용자가 실수로 ckpt 줬어도 scratch 모드면 무시 + 알림.
        if args.attacker_ckpt or args.defender_ckpt or args.attacker_resume or args.defender_resume:
            print(f"[preflight] train_mode=scratch — ckpt 인자 모두 무시 (fresh PPO init)",
                  flush=True)
        args.attacker_ckpt = ""
        args.defender_ckpt = ""
        args.attacker_resume = ""
        args.defender_resume = ""
        return

    # pretrain mode: ckpt required
    for label, path_s in [
        ("attacker_ckpt", args.attacker_ckpt),
        ("defender_ckpt", args.defender_ckpt),
    ]:
        if not path_s:
            raise SystemExit(
                f"[preflight] train_mode=pretrain — --{label} required"
            )
        p = Path(path_s)
        if not p.exists():
            raise SystemExit(f"[preflight] {label} not found: {p}")
    # resume optional — if given, must exist.
    for label, path_s in [
        ("attacker_resume", args.attacker_resume),
        ("defender_resume", args.defender_resume),
    ]:
        if path_s:
            p = Path(path_s)
            if not p.exists():
                raise SystemExit(f"[preflight] {label} not found: {p}")


# ---------------------------------------------------------------------------
# Nash gate
# ---------------------------------------------------------------------------

def _check_convergence(history: list[dict], window: int, eps: float, draw_max: float) -> bool:
    if len(history) < window:
        return False
    recent = history[-window:]
    wr_ok = all(abs(h["attacker_winrate"] - h["defender_winrate"]) < eps for h in recent)
    draw_ok = all(h["draw_rate"] < draw_max for h in recent)
    return wr_ok and draw_ok


# ---------------------------------------------------------------------------
# Stdout markdown table
# ---------------------------------------------------------------------------

def _print_cycle_row(cycle: int, h: dict, converged: bool) -> None:
    att = h["attacker_winrate"]
    dfn = h["defender_winrate"]
    drw = h["draw_rate"]
    status = "✓ converged" if converged else ""
    print(
        f"{cycle:5d} | {att:6.3f} | {dfn:6.3f} | {drw:6.3f} | "
        f"{abs(att-dfn):6.3f} | {status}",
        flush=True,
    )


def _print_table_header() -> None:
    print(
        f"\n{'Cycle':>5} | {'Att%':>6} | {'Def%':>6} | {'Draw%':>6} | "
        f"{'|Δ|':>6} | Status",
        flush=True,
    )
    print("-" * 60, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:   # noqa: PLR0912 — orchestrator drives 3-phase per-cycle loop
    args = parse_args()
    if args.smoke:
        _apply_smoke(args)

    _preflight(args)

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    # Path absolute — subprocesses may have different cwd; never pass relative.
    logs_root = Path(args.logs_root).resolve()
    logs_root.mkdir(parents=True, exist_ok=True)
    if args.train_mode == "scratch":
        cur_a_actor: Path | None = None
        cur_d_actor: Path | None = None
        cur_a_resume: Path | None = None
        cur_d_resume: Path | None = None
    else:
        cur_a_actor = Path(args.attacker_ckpt).resolve()
        cur_d_actor = Path(args.defender_ckpt).resolve()
        cur_a_resume = Path(args.attacker_resume).resolve() if args.attacker_resume else None
        cur_d_resume = Path(args.defender_resume).resolve() if args.defender_resume else None

    py = sys.executable
    train_a = str(ROOT / "scripts" / "train_attacker.py")
    train_d = str(ROOT / "scripts" / "train_defender.py")
    eval_h2h = str(ROOT / "scripts" / "eval_head_to_head.py")

    print(f"[ams-s2] logs_root = {logs_root}", flush=True)
    print(f"[ams-s2] cycle 0 attacker = {cur_a_actor}", flush=True)
    print(f"[ams-s2] cycle 0 defender = {cur_d_actor}", flush=True)
    print(f"[ams-s2] backend={args.backend}  num_envs={args.num_envs}  "
          f"switch_every={args.switch_every}  max_cycles={args.max_cycles}  "
          f"n_h2h={args.n_h2h_episodes}  eps={args.convergence_eps}  "
          f"draw_max={args.draw_max}", flush=True)

    history: list[dict] = []
    # Phase E.1 — asymmetric critic flag 자식 cmd에 전달용.
    extra_train_args: list[str] = []
    if args.use_asymmetric_critic:
        extra_train_args.append("--use_asymmetric_critic")
        print("[ams-s2] Phase E.1 — asymmetric critic ON (자식 train_attacker/defender)",
              flush=True)
    if args.deterministic:
        extra_train_args.append("--deterministic")
        print("[ams-s2] deterministic ON — 자식 train scripts가 cudnn determinism 강제",
              flush=True)
    if args.minimal_log:
        extra_train_args.append("--minimal_log")
        print("[ams-s2] minimal_log ON — 자식 터미널 'Mean episode {*}' extras 차단",
              flush=True)

    # 2026-05-13 — run_name prefix: cli --wandb_project. 자식 cmd에 forward.
    run_prefix = str(args.wandb_project) if args.wandb_project else "exp"
    if args.wandb_project:
        extra_train_args.extend(["--wandb_project", str(args.wandb_project)])
    print(f"[ams-s2] run_name prefix = '{run_prefix}'", flush=True)

    # Phase D — FP pool 누적 (cycle별 산출 ckpt 추가). use_fp=False면 사용 안 됨.
    attacker_pool: list[Path] = []
    defender_pool: list[Path] = []
    if args.use_fp:
        if cur_a_actor is not None:
            attacker_pool.append(cur_a_actor)
        if cur_d_actor is not None:
            defender_pool.append(cur_d_actor)
        print(f"[ams-s2 FP] use_fp=True initial pool sizes: "
              f"attacker={len(attacker_pool)} defender={len(defender_pool)}", flush=True)

    _print_table_header()

    converged = False
    for cycle in range(1, args.max_cycles + 1):
        cycle_dir = logs_root / f"cycle_{cycle:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        a_train_dir = cycle_dir / "attacker_train"
        d_train_dir = cycle_dir / "defender_train"

        # ---- A) attacker trains vs frozen prev defender ----
        # scratch + cycle 1: stationary 상대로 cycle 1 attacker fresh init 학습.
        # cycle 2+ (또는 pretrain mode): pretrained defender 상대.
        # Phase D (use_fp): cycle 1 (no pool) stationary, cycle 2+ pool 사용.
        def_args = _defender_args_for_attacker_train(
            args.use_fp, defender_pool, cur_d_actor, pold=args.pold,
        )
        cmd_a = [
            py, train_a, *def_args,
            "--num_envs", str(args.num_envs),
            "--max_iterations", str(args.switch_every),
            "--seed", str(args.seed),
            "--backend", args.backend,
            "--logger", args.logger,
            "--logs_root", str(a_train_dir),
            "--env_yaml", args.env_yaml,
            "--train_yaml", args.attacker_train_yaml,
            "--run_name", f"{run_prefix}_attacker_cycle_{cycle}_round_0",
            *extra_train_args,
        ]
        if "stationary" in def_args:
            print(f"[ams-s2 cycle {cycle}] attacker scratch — vs stationary (fresh PPO init)",
                  flush=True)
        elif "pretrained_pool" in def_args:
            print(f"[ams-s2 cycle {cycle}] attacker — vs FP defender pool "
                  f"(K={len(defender_pool)})", flush=True)
        if cur_a_resume is not None:
            cmd_a += ["--resume", str(cur_a_resume)]
        rc = _run_child(cmd_a)
        if rc != 0:
            _fail(f"train_attacker rc={rc}; cmd: {' '.join(cmd_a)}", cycle=cycle)
        a_actor_src = _find_actor_pt(a_train_dir, "attacker")
        a_model_src = _find_model_pt(a_train_dir)
        a_cfgs_src = _find_cfgs_pkl(a_train_dir)
        if a_actor_src is None or a_actor_src.stat().st_size == 0:
            _fail(f"attacker actor ckpt missing/empty under {a_train_dir}", cycle=cycle)
        if a_model_src is None or a_model_src.stat().st_size == 0:
            _fail(f"attacker model_*.pt missing/empty under {a_train_dir}", cycle=cycle)
        # Stable cross-cycle copies.
        a_actor_dst = _copy_to(logs_root / f"attacker_v{cycle}.pt", a_actor_src)
        a_model_dst = _copy_to(logs_root / f"model_attacker_v{cycle}.ckpt", a_model_src)
        # 2026-05-04 — also drop cfgs.pkl at logs_root so eval.py / scripts can
        # auto-resolve from any attacker_v*.pt sibling without spelunking into
        # cycle subfolders. ams uses the attacker_train cfgs.pkl as canonical
        # (defender_train's is identical modulo ego="defender" override).
        if a_cfgs_src is not None:
            _copy_to(logs_root / "cfgs.pkl", a_cfgs_src)

        # ---- A.adaptive ----
        # 2026-05-04 (사용자 의도): attacker가 frozen def_{k-1} 상대 mission rate
        # 충분한지 검증. 부족하면 attacker만 추가 학습 (def 정책 흔들지 않음).
        # cur_d_actor가 None (scratch cycle 1)이면 skip.
        if cur_d_actor is not None and args.attacker_min_winrate > 0:
            a_round = 0
            a_max = max(0, int(args.per_stage_max_rounds))
            while True:
                eval_a_json = cycle_dir / f"h2h_a_check_r{a_round:02d}.json"
                cmd_a_eval = _h2h_cmd(
                    py, eval_h2h, a_actor_dst, cur_d_actor,
                    eval_a_json, args, a_cfgs_src,
                )
                rc = _run_child(cmd_a_eval)
                if rc != 0:
                    _fail(
                        f"A.check eval r{a_round} rc={rc}; cmd: {' '.join(cmd_a_eval)}",
                        cycle=cycle,
                    )
                if not eval_a_json.is_file() or eval_a_json.stat().st_size == 0:
                    _fail(
                        f"A.check h2h r{a_round} missing/empty: {eval_a_json}",
                        cycle=cycle,
                    )
                with open(eval_a_json) as f:
                    last_a_check = json.load(f)
                att_wr = last_a_check["attacker_winrate"]
                print(
                    f"[ams-s2 cycle {cycle}] A.check r{a_round}: "
                    f"att_wr={att_wr:.3f} "
                    f"(target ≥ {args.attacker_min_winrate})",
                    flush=True,
                )
                if att_wr >= args.attacker_min_winrate:
                    break
                if a_round >= a_max:
                    break
                a_round += 1
                extra_dir = cycle_dir / f"attacker_train_extra_r{a_round:02d}"
                # Phase D — FP pool 사용 시 동일 helper.
                def_args_extra = _defender_args_for_attacker_train(
                    args.use_fp, defender_pool, cur_d_actor, pold=args.pold,
                )
                cmd_extra = [
                    py, train_a, *def_args_extra,
                    "--num_envs", str(args.num_envs),
                    "--max_iterations", str(args.switch_every),
                    "--seed", str(args.seed),
                    "--backend", args.backend,
                    "--logger", args.logger,
                    "--logs_root", str(extra_dir),
                    "--env_yaml", args.env_yaml,
                    "--train_yaml", args.attacker_train_yaml,
                    "--resume", str(a_model_dst),
                    "--run_name", f"{run_prefix}_attacker_cycle_{cycle}_round_{a_round}",
                    *extra_train_args,
                ]
                rc = _run_child(cmd_extra)
                if rc != 0:
                    _fail(
                        f"A.adaptive train_attacker rc={rc}; cmd: {' '.join(cmd_extra)}",
                        cycle=cycle,
                    )
                new_a = _find_actor_pt(extra_dir, "attacker")
                new_a_model = _find_model_pt(extra_dir)
                if new_a is None or new_a_model is None:
                    _fail(
                        f"A.adaptive attacker missing artifacts under {extra_dir}",
                        cycle=cycle,
                    )
                a_actor_dst = _copy_to(logs_root / f"attacker_v{cycle}.pt", new_a)
                a_model_dst = _copy_to(
                    logs_root / f"model_attacker_v{cycle}.ckpt", new_a_model
                )

        # ---- B) defender trains vs frozen NEW attacker ----
        # Phase D — FP: A.adaptive 끝난 a_actor_dst를 pool에 추가 후 cmd 빌드.
        if args.use_fp:
            attacker_pool.append(a_actor_dst)
        att_args = _attacker_args_for_defender_train(
            args.use_fp, attacker_pool, a_actor_dst, pold=args.pold,
        )
        cmd_b = [
            py, train_d, *att_args,
            "--num_envs", str(args.num_envs),
            "--max_iterations", str(args.switch_every),
            "--seed", str(args.seed),
            "--backend", args.backend,
            "--logger", args.logger,
            "--logs_root", str(d_train_dir),
            "--env_yaml", args.env_yaml,
            "--train_yaml", args.defender_train_yaml,
            "--run_name", f"{run_prefix}_defender_cycle_{cycle}_round_0",
            *extra_train_args,
        ]
        if cur_d_resume is not None:
            cmd_b += ["--resume", str(cur_d_resume)]
        rc = _run_child(cmd_b)
        if rc != 0:
            _fail(f"train_defender rc={rc}; cmd: {' '.join(cmd_b)}", cycle=cycle)
        d_actor_src = _find_actor_pt(d_train_dir, "defender")
        d_model_src = _find_model_pt(d_train_dir)
        if d_actor_src is None or d_actor_src.stat().st_size == 0:
            _fail(f"defender actor ckpt missing/empty under {d_train_dir}", cycle=cycle)
        if d_model_src is None or d_model_src.stat().st_size == 0:
            _fail(f"defender model_*.pt missing/empty under {d_train_dir}", cycle=cycle)
        d_actor_dst = _copy_to(logs_root / f"defender_v{cycle}.pt", d_actor_src)
        d_model_dst = _copy_to(logs_root / f"model_defender_v{cycle}.ckpt", d_model_src)

        # ---- B.adaptive ----
        # 2026-05-04 (사용자 의도): defender가 frozen att_v_k (post-A.adaptive)
        # 상대 capture rate 충분한지 검증. 부족하면 defender만 추가 학습.
        # 마지막 eval은 cycle h2h.json으로도 사용 (att/def 모두 finalized).
        last_b_check = None
        last_b_check_path = None
        if args.defender_min_winrate > 0:
            b_round = 0
            b_max = max(0, int(args.per_stage_max_rounds))
            while True:
                eval_b_json = cycle_dir / f"h2h_b_check_r{b_round:02d}.json"
                cmd_b_eval = _h2h_cmd(
                    py, eval_h2h, a_actor_dst, d_actor_dst,
                    eval_b_json, args, a_cfgs_src,
                )
                rc = _run_child(cmd_b_eval)
                if rc != 0:
                    _fail(
                        f"B.check eval r{b_round} rc={rc}; cmd: {' '.join(cmd_b_eval)}",
                        cycle=cycle,
                    )
                if not eval_b_json.is_file() or eval_b_json.stat().st_size == 0:
                    _fail(
                        f"B.check h2h r{b_round} missing/empty: {eval_b_json}",
                        cycle=cycle,
                    )
                with open(eval_b_json) as f:
                    last_b_check = json.load(f)
                last_b_check_path = eval_b_json
                def_wr = last_b_check["defender_winrate"]
                print(
                    f"[ams-s2 cycle {cycle}] B.check r{b_round}: "
                    f"def_wr={def_wr:.3f} "
                    f"(target ≥ {args.defender_min_winrate})",
                    flush=True,
                )
                if def_wr >= args.defender_min_winrate:
                    break
                if b_round >= b_max:
                    break
                b_round += 1
                extra_dir = cycle_dir / f"defender_train_extra_r{b_round:02d}"
                # Phase D — FP pool 사용 시 동일 helper.
                att_args_extra = _attacker_args_for_defender_train(
                    args.use_fp, attacker_pool, a_actor_dst, pold=args.pold,
                )
                cmd_extra = [
                    py, train_d, *att_args_extra,
                    "--num_envs", str(args.num_envs),
                    "--max_iterations", str(args.switch_every),
                    "--seed", str(args.seed),
                    "--backend", args.backend,
                    "--logger", args.logger,
                    "--logs_root", str(extra_dir),
                    "--env_yaml", args.env_yaml,
                    "--train_yaml", args.defender_train_yaml,
                    "--resume", str(d_model_dst),
                    "--run_name", f"{run_prefix}_defender_cycle_{cycle}_round_{b_round}",
                    *extra_train_args,
                ]
                rc = _run_child(cmd_extra)
                if rc != 0:
                    _fail(
                        f"B.adaptive train_defender rc={rc}; cmd: {' '.join(cmd_extra)}",
                        cycle=cycle,
                    )
                new_d = _find_actor_pt(extra_dir, "defender")
                new_d_model = _find_model_pt(extra_dir)
                if new_d is None or new_d_model is None:
                    _fail(
                        f"B.adaptive defender missing artifacts under {extra_dir}",
                        cycle=cycle,
                    )
                d_actor_dst = _copy_to(logs_root / f"defender_v{cycle}.pt", new_d)
                d_model_dst = _copy_to(
                    logs_root / f"model_defender_v{cycle}.ckpt", new_d_model
                )

        # ---- C) Cycle h2h.json — Nash 측정 ----
        # B.adaptive가 실행됐으면 마지막 B.check eval을 canonical h2h.json으로
        # 사용 (둘 다 finalized된 후 측정값). 아니면 fresh h2h 실행.
        h2h_json = cycle_dir / "h2h.json"
        if last_b_check is not None and last_b_check_path is not None:
            shutil.copy2(last_b_check_path, h2h_json)
            h = last_b_check
        else:
            cmd_c = _h2h_cmd(
                py, eval_h2h, a_actor_dst, d_actor_dst,
                h2h_json, args, a_cfgs_src,
            )
            rc = _run_child(cmd_c)
            if rc != 0:
                _fail(f"eval_head_to_head rc={rc}; cmd: {' '.join(cmd_c)}", cycle=cycle)
            if not h2h_json.is_file() or h2h_json.stat().st_size == 0:
                _fail(f"h2h.json missing/empty: {h2h_json}", cycle=cycle)
            with open(h2h_json) as f:
                h = json.load(f)
        history.append(h)
        converged = _check_convergence(
            history, args.convergence_window, args.convergence_eps, args.draw_max,
        )
        _print_cycle_row(cycle, h, converged)

        # Phase D — FP: cycle 끝 산출 defender도 pool에 추가 (attacker_pool은 B 시작 시 추가됨).
        if args.use_fp:
            defender_pool.append(d_actor_dst)
            print(f"[ams-s2 FP cycle {cycle}] pool sizes: "
                  f"attacker={len(attacker_pool)} defender={len(defender_pool)}", flush=True)

        # Roll forward — next cycle's frozen opponents + resume bases.
        cur_a_actor = a_actor_dst
        cur_d_actor = d_actor_dst
        cur_a_resume = a_model_dst
        cur_d_resume = d_model_dst

        if converged:
            print(
                f"\n[ams-s2] ✓ converged at cycle {cycle} — "
                f"|att-def|<{args.convergence_eps} AND draw<{args.draw_max} "
                f"for last {args.convergence_window} cycles.",
                flush=True,
            )
            break

    if not converged:
        print(
            f"\n[ams-s2] ⚠ max_cycles={args.max_cycles} reached without convergence. "
            f"Last cycle: |att-def|={abs(history[-1]['attacker_winrate'] - history[-1]['defender_winrate']):.3f}, "
            f"draw={history[-1]['draw_rate']:.3f}.",
            flush=True,
        )

    # Brief final summary (full summary.json deferred to PR-N.1).
    print(
        f"\n[ams-s2] cycles_run={len(history)} converged={converged} "
        f"final_a_actor={cur_a_actor} final_d_actor={cur_d_actor}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
