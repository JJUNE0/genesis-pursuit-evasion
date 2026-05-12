# Phase B — defender fragility 처방 (action_smooth per-side + omega clamp)

> **상태**: 사용자 승인 후 학습 실행 (§11).
> **선행**: Phase A (P0 R_opp_crash=0) 학습 결과 (`logs/.../ams_s2_v2/`) — c7 |Δ|=0.003 균형 통과 후 c8+ collapse. fragility 진단(`scripts/test/diag_defender_fragility.py`)으로 가설 A+C 동시 STRONG.

---

## 1. 동기 (Why)

`logs/pe1v1_v0/ams_s2_v2/diag_def_v10/summary.json` 진단 결과 (defender_v10 vs StationaryAttacker, 100 ep):

| metric | 값 | 가설 |
|---|---:|---|
| stationary 상대 def_crash | **0.82** | C (fragile) STRONG |
| action sat_frac (4축 평균) | **0.99** | A (bang-bang) STRONG |
| \|ω\| mean / p99 / max | 5.18 / 5.99 / 7.48 rad/s | C 보강 (max_body_rate=π=3.14를 1.6× 초과) |
| att_crash | 0.00 | B (OOD) REJECTED |

**근본 원인**:
- `omega_coef_defender = 0.0` (PR-F 결정 — chase 자연 자세 차단 방지)이 자세 안정성 학습 신호를 차단.
- defender PPO가 "공격적 추격 = saturated rate command가 capture에 좋다" 학습 → 4축 모두 ±1 saturate.
- max_body_rate=π에 PD inner loop 추종 시도 → inertia로 overshoot → 실제 \|ω\| 5.18 mean.
- stationary 표적 상대로도 자기 자세 발산이 capture 시도보다 빠름 → def_crash 82%.

PR-K (rate controller PID 튜닝) 노트의 "bang-bang on rate command 미검증" 시나리오 직접 입증.

## 2. 변경 사항 (What)

| ID | 항목 | 현재 (Phase A) | Phase B |
|---|---|---|---|
| **B1** | action_smooth (per-side 분리) | `action_smooth: 1.0e-4` (양측 공유) | `action_smooth_attacker: 1.0e-4` (불변) + `action_smooth_defender: 5.0e-4` (5×) |
| **B2** | omega_coef_defender | 0.0 | 2.0e-5 |
| **B2c** | omega_clamp_max_sq | (없음, 무제한) | **25.0** (clamp \|ω\|²) — \|ω\|>5 영역 페널티 saturate |
| **B3** | max_body_rate | π | π (보류 — B1+B2 결과 보고 ±π/2 검토) |

### 신규 / 수정 파일

- 신규: `configs/envs/pe_1v1_phase_b.yaml`
- 신규: `scripts/test/diag_defender_fragility.py` (이미 작성)
- 수정: `envs/pe_1v1/env.py` — additive only:
  - `action_smooth_attacker` / `action_smooth_defender` 키 분리, legacy `action_smooth`로 fallback (Phase 1.0/1.5/A yaml 무영향)
  - `omega_clamp_max_sq` 키 (default `inf` = clamp 없음, backward compat)
  - 두 reward 함수에서 변수 사용 + `omega_sq.clamp(max=...)` 적용

## 3. 영향 범위 (Scope)

| 영역 | 영향 |
|---|---|
| obs / action shape | 무변경 |
| 체크포인트 호환성 | actor weight 그대로 사용. reward는 학습 시점 결정 → eval/transfer 무관 |
| Phase 1.0/1.5/A yaml 학습 재현성 | 무영향 (action_smooth fallback + omega_clamp default `inf`로 동일 동작) |
| minimax 부등식 | 동일하게 유지 (R_self_crash=-3, R_opp_crash=0, R_captured=-1, R_capture_win=+1) |

## 4. 물리/이론 sanity check

**(a) action_smooth_defender=5e-4 누적**:
- bang-bang 한계: \|action_diff\|² ≈ 4 (각 축 ±1 swing 시 차이 2² + ...) → step 페널티 ≈ -2e-3.
- 1000 step ≈ -2.0 (R_capture_win=+1.0의 2배).
- bang-bang 차단 incentive 강함.
- 정상 chase (|action_diff|≈0.5): step ≈ -1.25e-4, 1000 step ≈ -0.125 (작음).

**(b) omega_coef_defender=2e-5 + clamp 25**:
- 정상 chase \|ω\|≈3 → omega_sq=9 → step -1.8e-4, 1000 step -0.18 (작음).
- 자세 발산 \|ω\|=5 → omega_sq=25 (clamp) → step -5.0e-4, 1000 step -0.5.
- \|ω\|=10 (catastrophic) → omega_sq=25 (clamp 동일) → 같은 -0.5. **즉 self-crash incentive 차단** (R_self_crash=-3 유지가 더 큼).
- chase 약화 최소화 사용자 의도 충족.

**(c) attacker 영향**:
- B1: attacker action_smooth=1e-4 그대로 (Phase A 동등).
- B2: omega_coef_attacker=5e-5 그대로. Phase A까진 clamp 없음, B에선 clamp 25 적용. attacker도 \|ω\|=10 trap이 신호로 -1.25e-3 (1e-2 → 1.25e-3) 약화. 다만 R_self_crash=-3가 종료 시 강한 신호로 남음. attacker fragility 진단 결과 미약(c12 a_cr 0.06)이라 부수적 영향 OK.

**(d) clamp가 chase 학습 차단?**:
- \|ω\|<5 영역에서 quadratic 신호 그대로 → chase 학습 자유도 보존.
- \|ω\|>5 영역에서만 saturate → "더 흔들리면 더 페널티" 신호 차단되지만 그 영역은 이미 자세 발산 → 학습 회피 신호로 충분.

## 5. Rollback plan

- `logs_root` 분리 (`ams_s2_v3`)로 ams_s2/ ams_s2_v2/ 무손상 보존.
- env code 변경은 모두 additive (legacy fallback). 기존 yaml 학습 무영향.
- 실패 시 Phase A yaml로 invocation 복귀.

## 6. 통과 기준 (5-10 cycle 후 측정)

defender_train phase 마지막 5-점 평균:
- `def_crash_rate` **< 5%** (현재 ams_s2_v2 c11 a_train: 63%)
- `sat_frac (mean of 4 dims)` **< 70%** (현재 99%)
- `|ω| mean` **< 4 rad/s** (현재 5.18)
- `captured_rate` **> 60%**

추가 Nash 시그널: 3-cycle rolling \|att_wr − def_wr\| < 0.10.

| 결과 | 다음 step |
|---|---|
| 모두 만족 | Phase B 종료 → ams_s2_v3 vs ams_s2_v2 비교 |
| 부분 만족 (def_crash 0.05–0.20) | B3 (max_body_rate ±π/2) 추가 |
| 실패 (def_crash > 0.20) | 재진단 (defender_v_new vs StationaryAttacker) |

## 7. 실행 명령 (사용자 승인 후)

```bash
.venv/bin/python scripts/train_ams_drl.py \
    --attacker_ckpt logs/pe1v1_v0/attacker_v0/phase1.5_attacker_stationary_seed0_1777450261_2e97bec-dirty/attacker_v0.pt \
    --defender_ckpt logs/pe1v1_v0/defender_v0/phase1.0_defender_seed0_1777450253_2e97bec-dirty/defender_v0.pt \
    --attacker_resume logs/pe1v1_v0/attacker_v0/phase1.5_attacker_stationary_seed0_1777450261_2e97bec-dirty/model_1999.pt \
    --defender_resume logs/pe1v1_v0/defender_v0/phase1.0_defender_seed0_1777450253_2e97bec-dirty/model_1999.pt \
    --env_yaml configs/envs/pe_1v1_phase_b.yaml \
    --num_envs 1024 --switch_every 50 --max_cycles 30 \
    --n_h2h_episodes 400 --convergence_window 3 --convergence_eps 0.10 \
    --draw_max 0.2 --seed 0 --logger wandb \
    --logs_root logs/pe1v1_v0/ams_s2_v3
```

(max_cycles=30으로 줄임 — 5-10 cycle 후 통과 기준 점검 + 회복 추세 보고 결정. ams_s2_v2 11 cycle에서 collapse 전조 보였으므로 30으로 충분 + 조기 종료 가능)

## 8. 검증 후 추가 진단 (선택)

5 cycle 시점에서 defender_v5 (ams_s2_v3) 같은 fragility diag 재실행:
```bash
.venv/bin/python scripts/test/diag_defender_fragility.py \
    --defender_ckpt logs/pe1v1_v0/ams_s2_v3/defender_v5.pt \
    --cfgs_pkl logs/pe1v1_v0/ams_s2_v3/cycle_005/defender_train/<run>/cfgs.pkl \
    --n_episodes 100 --num_envs 32 --seed 0 --backend gpu \
    --output_dir logs/pe1v1_v0/ams_s2_v3/diag_def_v5
```
ams_s2_v2 v10 vs ams_s2_v3 v5 직접 비교. Phase B 처방 효과 양적 검증.
