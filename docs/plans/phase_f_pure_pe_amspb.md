# Phase F — AMSPB-style Pure PE Baseline (2026-05-10)

> **상태**: 사용자 승인 (2026-05-10). 즉시 구현.
> **선행**: v15~v18 결과 — vanilla AMS-DRL alternating PPO + reward tuning + use_fp(pold=1.0) 모두 Nash 도달 못함. 사용자 critique 3건으로 근본 issue 진단.
> **Goal**: AMSPB paper와 direct apple-to-apple comparison + 변수 분리 원칙.

---

## 1. 동기 — 사용자 critique (검증된 3대 issue)

| # | Issue | v18 증거 | Phase F 처치 |
|---|---|---|---|
| 1 | Population pold=1.0 (AMSPB 0.5보다 강함) | a_check 0.142 (pool dilution) | **pold=0.5 mixing** |
| 2 | Reward 15+ 항 (AMSPB 5항) | mode 진동 (mission/도주/자살) | **reward 5~6항으로 단순화** |
| 3 | 게임 구조 (mission+evasion vs pure PE) | g_mission이 attacker mode 진동 야기 | **pure PE 모드 도입 (mission 비활성)** |

**변수 분리 원칙**: 3가지 동시 fix 대신 **pure PE + 단순 reward + pold=0.5**로 깨끗한 baseline → Nash 검증 → mission 추가.

---

## 2. Reward Spec — AMSPB Table II 기반

### Pure PE 모드 (Phase F.1)

**Attacker (evader)**:
```
r_a = +κa·(d_t - d_{t-1})           # attacker-defender 거리 증가 (도망)
    + κs                              # survival (per step) — attacker만
    - κbr·‖ω‖²                        # body rate penalty (L2)
    + terminal:
        +κt              (timeout = escape)
        -κc              (captured)
        +R_self_crash    (자기 hard crash/oob — 자세 90°/v_max/z_floor/bound)
        # opponent crash/oob → 보상 X (R_opp_crash 제거)
```

**Defender (pursuer)**:
```
r_d = +κa·(d_{t-1} - d_t)           # 거리 감소 (chase) — dense shaping
                                       # ★ κs 없음 — defender 생존 보상 X (chase 약화 방지)
    - κbr·‖ω‖²
    + terminal:
        +κc              (capture)
        -κt              (timeout = miss)
        +R_self_crash    (자기 hard crash/oob)
        # opponent crash/oob → 보상 X
```

### 계수 (사용자 확정)

| key | 값 | 의도 |
|---|---:|---|
| κa (approach/escape shaping) | **0.1** | AMSPB 0.5 × (62.5Hz / 12.5Hz) ≈ 0.1 (env step 5× 큼) |
| κs (survival, attacker only) | 0.005 | existential 동등 |
| κc (terminal capture/escape) | 10 | AMSPB κc 동일 |
| κbr (body rate, L2) | 0.005 | chase banking 규제 |
| κt (timeout) | ±5 | symmetric |
| **R_self_crash** | **-10** | hard crash/oob → capture 동등 페널티 (사용자 spec 2026-05-10) |
| R_opp_crash | **0 (제거)** | opponent crash → 별도 보상 X |

### Mission 모드 (Phase F.2 — F.1c 통과 시)

Phase F.1 reward에 **mission progress 추가**:
```
r_a += κa·(d_g_{t-1} - d_g_t)   # mission goal 향한 progress
    + +κc (R_mission_win, terminal)
r_d += -κc (R_mission_failed, terminal)
```

→ mission objective re-introduce. F.1 ckpt warm-start.

---

## 3. Termination (Hard Crash 복원)

```yaml
termination:
  crash_on_attitude: true   # ← 복원 (AMS yaml false였음)
  pitch_max_deg: 90.0       # ← 사용자 결정 (60°→90° 넉넉, defender chase 살림)
  roll_max_deg:  90.0
  crash_on_speed:    true   # ← 복원
  v_max:             15.0
  z_floor:           0.05
  bound_x: 12, bound_y: 6, bound_z: 6
```

soft penalty 모두 제거 (`attitude_pad/coef`, `velocity_pad/coef`, `z_safety_*`, `bound_safety_*`).

---

## 4. Population — pold=0.5 Mixing

`PretrainedDefenderPool` / `PretrainedAttackerPool` 수정:
- `current_best_ckpt` 추가 (직전 cycle 학습된 best-response)
- 매 reset 시: `pold` 확률로 pool에서 random sample, `(1-pold)` 확률로 current_best
- AMSPB와 동일 메커니즘

```python
class PretrainedDefenderPool:
    def __init__(self, ..., current_best_ckpt=None, pold=0.5):
        self.pool_actors = [load(p) for p in ckpt_paths]
        self.current_best = load(current_best_ckpt) if current_best_ckpt else None
        self.pold = pold
    
    def reset(self, envs_idx):
        # pold 확률로 pool index, else "current_best" sentinel index
        for i in envs_idx:
            if random.random() < self.pold and self.pool_actors:
                self.env_to_ckpt[i] = random.randint(0, len(self.pool_actors))
            else:
                self.env_to_ckpt[i] = -1  # current_best sentinel
```

---

## 5. 환경 / Schedule (No Curriculum)

**Stage 4 arena만 사용** (사용자 결정 — curriculum 제거):
- spawn att x [0.5, 4.0], def x [-4.0, -0.5]
- bound 12/6/6
- 단 reward + termination + game_mode는 phase_f yaml 따름

**Alternating**:
- `--num_envs 1024 --switch_every 200 --max_cycles 10`
- `--use_fp --pold 0.5`
- `--attacker_min_winrate 0.4 --defender_min_winrate 0.4 --per_stage_max_rounds 5`
- Nash gate: `--convergence_eps 0.05 --draw_max 0.2 --convergence_window 3`

---

## 6. 신규 / 수정 파일

### 신규
- `configs/envs/pe_1v1_phase_f_pure_pe.yaml` — Phase F 전용 yaml
- `tests/test_pure_pe_reward.py` — reward 함수 단위 테스트
- `tests/test_pold_mixing.py` — pold=0.5 mixing 검증

### 수정 (additive flag)
- `envs/pe_1v1/cfgs.py` — `game_mode` cfg 키 ("mission" | "pure_pe")
- `envs/pe_1v1/env.py` — game_mode 분기 + pure_pe reward 함수
- `envs/pe_1v1/scripted_defender.py` — PretrainedDefenderPool에 `current_best_ckpt` + `pold`
- `envs/pe_1v1/scripted_attacker.py` — 동일
- `scripts/train_ams_drl.py` — `--pold` flag, cmd build 시 전달
- `scripts/train_attacker.py` / `train_defender.py` — `--current_best_ckpt`, `--pold` flag

---

## 7. 학습 단계 (3단계)

### F.1a — Defender Sanity (alternating 전)

```bash
.venv/bin/python scripts/train_defender.py \
  --attacker_mode random_walk \
  --env_yaml configs/envs/pe_1v1_phase_f_pure_pe.yaml \
  --train_yaml configs/train/ppo_defender_tanhm.yaml \
  --num_envs 1024 --max_iterations 1000 --seed 0 \
  --backend gpu --logger tensorboard \
  --logs_root logs/pe1v1_v0/phase_f/sanity_def
```

**성공 기준**: capture_rate > 80%, def_crash_rate < 5% (random_walk attacker 잡기)

### F.1b — Attacker Sanity (도망 학습)

```bash
# 일단 stationary defender로 도망 확인
.venv/bin/python scripts/train_attacker.py \
  --defender stationary \
  --env_yaml configs/envs/pe_1v1_phase_f_pure_pe.yaml \
  --train_yaml configs/train/ppo_attacker_tanhm.yaml \
  --num_envs 1024 --max_iterations 1000 --seed 0 \
  --backend gpu --logger tensorboard \
  --logs_root logs/pe1v1_v0/phase_f/sanity_att
```

**성공 기준**: timeout_rate > 50% (attacker 5초 끝까지 살아남음). pure PE라 mission 없으니 timeout = escape success.

### F.1c — Pure PE Alternating Co-training

```bash
.venv/bin/python scripts/train_ams_drl.py \
  --train_mode pretrain \
  --attacker_ckpt logs/.../sanity_att/attacker_v0.pt \
  --attacker_resume logs/.../sanity_att/model_999.pt \
  --defender_ckpt logs/.../sanity_def/defender_v0.pt \
  --defender_resume logs/.../sanity_def/model_999.pt \
  --env_yaml configs/envs/pe_1v1_phase_f_pure_pe.yaml \
  --num_envs 1024 --switch_every 200 --max_cycles 10 \
  --use_fp --pold 0.5 \
  --attacker_min_winrate 0.4 --defender_min_winrate 0.4 \
  --per_stage_max_rounds 5 \
  --convergence_window 3 --convergence_eps 0.05 --draw_max 0.2 \
  --logs_root logs/pe1v1_v0/phase_f/alternating_v1 \
  --seed 0 --logger tensorboard
```

**성공 기준** (Nash 도달):
- |att_wr - def_wr| < 0.1 (3 cycle 연속)
- draw < 0.3 (degenerate equilibria 방지)

### F.2 — Mission Objective 추가 (F.1c 통과 시)

- env_cfg `game_mode = "mission"`로 전환
- F.1c attacker_v_K, defender_v_K ckpt warm-start
- mission progress + R_mission_win 추가
- alternating 진행

---

## 8. Sanity Check (학습 시작 전)

### 단위 테스트
- `test_pure_pe_reward.py`:
  - pure_pe 모드 attacker reward = κa(d_t - d_{t-1}) + κs - κbr|ω|² + terminal
  - mission_reached가 reward에 영향 X (pure_pe)
  - defender reward symmetric 검증
- `test_pold_mixing.py`:
  - pold=0.5 → 절반은 current_best, 절반은 pool
  - pold=1.0 → 모두 pool (기존 동작)
  - pold=0.0 → 모두 current_best (single ckpt 동등)

### Smoke
- cpu, num_envs=16, switch_every=2: F.1c 1 cycle 정상 실행 확인

---

## 9. Rollback Plan

Phase F.1c가 Nash 도달 못 하는 경우:

| 시도 | 변경 | 의도 |
|---|---|---|
| 1차 | κa: 0.1 → 0.05 | dense shaping 더 약화 (terminal 비중 ↑) |
| 2차 | κbr: 0.005 → 0.001 | omega regulation 약화 (chase 자유도 ↑) |
| 3차 | crash 90° → 100° | def_crash 더 줄임 |
| 4차 | pold: 0.5 → 0.7 (pool 비중 ↑) or 0.3 (current 비중 ↑) | mixing ratio 탐색 |
| 5차 | NFSP 도입 (best response + average policy 분리) | vanilla pool 한계 시 |

---

## 10. 시간 예산

| 작업 | 시간 |
|---|---:|
| yaml + env reward 수정 | 4h |
| pold=0.5 mixing 수정 + 테스트 | 3h |
| 단위 테스트 (reward + pold) | 2h |
| Smoke (cpu) | 30분 |
| **F.1a/b sanity (gpu)** | 3h |
| **F.1c alternating (gpu)** | 6h |
| 결과 분석 + F.2 결정 | 1h |
| **F.2 mission 추가** | 3h + 6h |
| **합계** | **~28h** |

---

## 11. TODO 작업 순서

1. **(승인)** 사용자 plan 승인 (2026-05-10 OK 받음)
2. `configs/envs/pe_1v1_phase_f_pure_pe.yaml` 작성
3. `envs/pe_1v1/cfgs.py` `game_mode` 키 추가
4. `envs/pe_1v1/env.py` reward 함수 분기 + pure_pe reward
5. `envs/pe_1v1/scripted_*.py` Pool에 current_best + pold mixing
6. `scripts/train_*.py` flag 추가
7. `tests/test_pure_pe_reward.py` + `tests/test_pold_mixing.py` 작성
8. Smoke (cpu)
9. **F.1a / F.1b** sanity (gpu) — defender / attacker 단독 학습 검증
10. **F.1c** alternating (Pure PE Nash 검증)
11. 결과 분석
12. (F.1c 통과 시) **F.2** mission objective 추가

---

## 12. 학술적 의미

이 phase가 통과되면:
- **AMSPB와 direct comparison 가능** (pure PE + 5항 reward + pold=0.5)
- **변수 분리** — game/reward/population 각각의 효과 측정
- **Curriculum 필요성 reframe** — pure PE에서 curriculum 없이 Nash 잡히면 "curriculum은 mission objective 위한 것" 명확
- **Paper main result**: F.1c (pure PE Nash) → F.2 (mission Nash) ablation 표

---

*승인 후 즉시 §11 TODO 2번 (yaml 작성)부터 진행.*
