# Phase A — R_opp_crash 제거 + Nash gate redesign

> **상태**: 사용자 승인 대기. 학습 실행은 §11 (max_iterations ≥ 100) 별도 승인 후.
> **선행**: AMS-S2 cycle 1-20 (logs/pe1v1_v0/ams_s2/) 결과는 attacker 0.007 / defender 0.84 / draw 0.15 — Nash 비수렴.

---

## 1. 동기 (Why)

S1 진단 (`logs/pe1v1_v0/ams_s2/diagnostics_s1_termination_trajectory.png`) 결과 **시나리오 A (defender harassment) 확정**:

| metric (defender_train phase 끝) | c12 (capture peak) | c20 | Δ |
|---|---:|---:|---:|
| captured_rate | 0.896 | 0.479 | −0.42 |
| att_crash_rate | 0.005 | 0.361 | +0.356 |
| mean_episode_length | 134.5 | 211.4 | +57% |

세 시그널 (capture 감소 + att_crash 폭증 + ep_len 증가) 모두 만족 → defender가 "잡지 않고 압박해서 추락 유도" 정책 수렴. attacker_train phase의 att_crash 증가폭(+0.06)은 defender_train(+0.36)의 1/6 — omega trap (B) 신호 미약.

**근본 원인**: `R_opp_crash=0.5`가 expected reward 관점에서 capture path와 등가가 되어 defender가 더 쉬운 harassment path 학습. minimax 부등식 (`R_opp_crash ≤ R_capture_win`)만 충족했지 학습 dynamics 미고려.

## 2. 변경 사항 (What)

| 항목 | 현재 | Phase A |
|---|---|---|
| R_opp_crash (attacker side, defender side 동일) | 0.5 | **0.0** |
| Nash gate epsilon | 0.05 | **0.10** |
| Nash gate window | 5 | **3** |
| Nash gate max_cycles | 20 | **60** |
| logs_root | `logs/pe1v1_v0/ams_s2/` | `logs/pe1v1_v0/ams_s2_v2/` |
| 초기 ckpt | phase1.5 attacker_v0 + phase1.0 defender_v0 (재학습 X) | 동일 (fresh restart) |

### 신규 파일
- `configs/envs/pe_1v1_phase_a.yaml` — `pe_1v1_default.yaml` 복제 + `R_opp_crash: 0.0`
- `scripts/test/dump_trajectories_baseline.py` — c20 baseline 50-episode trajectory dump (검증용)

### 기존 파일 수정 (additive only)
- `scripts/train_ams_drl.py` — `--env_yaml` CLI 인자 추가 + 자식(cmd_a, cmd_b)에 forward.
  - 기본값은 `pe_1v1_default.yaml` 유지 → 기존 invocation 무영향.
  - eval (cmd_c)는 cfgs.pkl 자동 직렬화로 자식과 일관성 유지 — 변경 불필요.

## 3. 영향 범위 (Scope)

| 영역 | 영향 |
|---|---|
| obs / action shape | 무변경 |
| 체크포인트 호환성 | 무변경 (actor weight만 사용; reward는 학습 시점 결정) |
| minimax 부등식 (env.py:155-161) | `R_opp_crash(0) ≤ R_capture_win(1)` 약화된 등호 만족 ✓ |
| Phase 1.0 / 1.5 ckpt | actor weight 그대로 사용. 학습 reward 변경은 transfer에만 영향 (의도) |
| 기존 logs/ams_s2/ | 무손상 보존 |

## 4. 물리/이론 sanity check

**(a) defender "wait & let crash" 학습 위험**: chase reward가 `d=1m`에서 −3.7e-4/step, capture 시 R_capture_win=+1.0. attacker 자살 baseline ≈ 5% (phase1.5 학습 측정). expected reward:
- wait policy: `0.05 × 0 + 0.95 × (−ε)` ≈ −small (chase + existential 누적 손해)
- chase policy: `0.6 × 1.0 + 0.4 × (−ε)` ≈ +0.6
→ chase 우세 보장.

**(b) defender "active recovery"**: R_opp_crash가 음수 안 됨 (0). attacker 추락 회피 인센티브 없음. 추격 행동 변화 없음 예상.

**(c) attacker 자살 trap**: R_self_crash=−3.0, R_captured=−1.0 무변경. attacker 행동 incentive에 직접 변경 없음.

**(d) Nash gate 통과 가능성**: epsilon=0.10, window=3에서 cycle별 |Δwr| swing 0.1 amplitude도 통과 후보. 측정 SE ≈ 0.025 (n=400) → epsilon=0.10은 4σ 폭. 통계적으로 의미 있는 안정 영역.

## 5. Rollback plan

- `logs_root` 분리로 ams_s2/ 무손상.
- 새 yaml + AMS CLI 인자 모두 부가적 (default 무변경) → 실패 시 기존 invocation으로 즉시 복귀.
- 만약 P0 단독으로 회복 안 되면 Phase B (P1 omega clamp 또는 Phase C defender 자세 안정성).

## 6. 통과 기준 (Gate for Phase A 종료)

5-10 cycle 후 defender_train phase 마지막 5-점 평균 동시 만족:
- captured_rate **> 0.80**
- att_crash_rate **< 0.10**
- mean_episode_length **< 200**

추가 Nash 시그널: 3-cycle rolling |att_wr − def_wr| < 0.10 AND draw < 0.20.

| 결과 | 다음 step |
|---|---|
| 모두 만족 | Phase A 종료 → S2 trajectory dump (Phase A retrain 후) vs c20 baseline 비교 |
| 부분 만족 (예: att_crash 0.10–0.20) | Phase B (P1 omega_coef + clamp) |
| 실패 (att_crash > 0.20 또는 capture < 0.6) | 재진단 (S2 dump 즉시 + reward decomp wrapper) |

## 7. 실행 명령 (사용자 승인 후)

```bash
# Phase 1.0 / 1.5 산출물 경로는 기존 ckpt 디렉토리에서 확인.
# 예: logs/pe1v1_v0/attacker_v0/phase1.5_attacker_stationary_seed0_<ts>_<sha>/
python scripts/train_ams_drl.py \
    --attacker_ckpt logs/pe1v1_v0/attacker_v0/<run>/attacker_v0.pt \
    --defender_ckpt logs/pe1v1_v0/defender_v0/<run>/defender_v0.pt \
    --attacker_resume logs/pe1v1_v0/attacker_v0/<run>/model_<n>.pt \
    --defender_resume logs/pe1v1_v0/defender_v0/<run>/model_<n>.pt \
    --env_yaml configs/envs/pe_1v1_phase_a.yaml \
    --num_envs 1024 --switch_every 50 --max_cycles 60 \
    --n_h2h_episodes 400 --convergence_window 3 --convergence_eps 0.10 \
    --draw_max 0.2 --seed 0 --logger wandb \
    --logs_root logs/pe1v1_v0/ams_s2_v2
```

## 8. S2 — Trajectory baseline dump (선택, ~30분)

Phase A 적용 *전* c20 ckpt에 대해 trajectory 50 episode 기록 → att_crash sub-cause 분포 (z_floor / pitch_max / roll_max / v_max). Phase A retrain 후 같은 dump 비교용.

```bash
python scripts/test/dump_trajectories_baseline.py \
    --attacker_ckpt logs/pe1v1_v0/ams_s2/attacker_v20.pt \
    --defender_ckpt logs/pe1v1_v0/ams_s2/defender_v20.pt \
    --cfgs_pkl logs/pe1v1_v0/ams_s2/cycle_020/attacker_train/<run>/cfgs.pkl \
    --n_episodes 50 --num_envs 32 --seed 0 \
    --output_dir logs/pe1v1_v0/ams_s2/diag_baseline_c20/
```
