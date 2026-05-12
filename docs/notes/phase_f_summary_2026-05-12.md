# Phase F 작업 보고서 (2026-04-30 ~ 2026-05-12)

> Phase 2 (= Phase 5 in old naming, AMS-DRL co-training) 본격 진척 요약.
> 8 reward shaping iteration + alternating order swap + cross-eval metric (NPS)
> + forecaster 인프라 + BPQL τ-delay infra 부분 구현.

---

## 1. 작업 기간 / 흐름

| 기간 | Phase | 핵심 작업 |
|---|---|---|
| 2026-04-30 ~ 2026-05-04 | Phase A-D | r_opp_crash, defender fragility, curriculum scratch, Fictitious Play |
| 2026-05-05 ~ 2026-05-07 | curriculum v1~v12 | per_stage yaml + AMS s2 v23~v30 series |
| 2026-05-08 ~ 2026-05-10 | Phase E.1, E.2, F | asymmetric critic infra + forecaster buffer/model/trainer + pure_pe AMSPB style |
| **2026-05-11 ~ 2026-05-12** | **swap + ams_v2~v6** | **alternating order swap (cycle 2+ B→A) + z_safety iteration → best baseline** |

---

## 2. 주요 코드 변경 (모두 untracked, commit 대기)

### 새 파일
| 파일 | 목적 |
|---|---|
| `agents/ppo_runner.py` | rsl-rl 5.2 wrapper |
| `agents/squashed_gaussian.py`, `tanh_mean_gaussian.py` | PR-O actor 변형 |
| `agents/forecaster/{model,replay_buffer,trainer}.py` | Phase E.2 forecaster 인프라 (Transformer 2-layer + FIFO 1M buffer + supervised trainer; 27/27 tests pass) |
| **`scripts/train_ams_drl_swap.py`** | **🆕 2026-05-11. cycle 1 attacker-only, cycle 2+ defender→attacker swap. 마지막 학습=attacker → h2h(a_v_K, d_v_K) attacker 우세 측정.** |
| `scripts/compute_nps.py` | 🆕 Nash Proximity Score (`1 − 2·std/√(μ(1−μ))`) + bootstrap 95% CI + noise-corrected NPS |
| `tests/test_{critic_priv_obs,forecaster_*,policy_pool,curriculum_stages}.py` | 47/47 tests pass |
| `configs/envs/pe_1v1_ams_v{2..6}.yaml` | iterative reward tuning |

### 변경된 파일
| 파일 | 변경 |
|---|---|
| `envs/pe_1v1/env.py` | `game_mode={mission, pure_pe}` 분기, `defender_chase_attenuation`, `_critic_priv_obs()` (10D) |
| `envs/pe_1v1/scripted_attacker.py` | `PretrainedAttackerPool` (FP + pold mixing) |
| `envs/pe_1v1/scripted_defender.py` | `PretrainedDefenderPool` + `ChaseDefender` (Phase F.1.5, deprecated — attitude 부재로 self-crash) |
| `envs/base_pe_env.py` | `obs_groups({actor, critic})` Phase E.1, **🆕 `defender_state_history` 6D ring buffer (BPQL τ-delay infra 시작 — 단 `torch.roll` 인자 버그)** |
| `envs/pe_1v1/cfgs.py` | `CRITIC_PRIV_DIM = 10`, `use_asymmetric_critic` |
| `scripts/train_ams_drl.py` | `--use_fp`, `--pold`, `--use_asymmetric_critic` flags |
| `scripts/train_attacker.py` | `--defender chase` choice + `PretrainedDefenderPool` + `--use_asymmetric_critic` forward |
| `scripts/train_defender.py` | 유사 — `PretrainedAttackerPool` + `--use_asymmetric_critic` forward |

---

## 3. Phase 2 (AMS) 학습 결과

### 공통 setup
- `train_ams_drl_swap.py --train_mode scratch --switch_every 50 --max_cycles 20`
- `--attacker_min_winrate 0.7 --defender_min_winrate 0.5 --per_stage_max_rounds 5`
- backend=gpu, num_envs=1024, n_h2h=200

### 결과 표 (cycle별 best peak)
| Run | yaml 핵심 | a_win peak | mission % | a_cr_z (후반) | balanced |
|---|---|---|---|---|---|
| ams_v2_swap | stage_4 (z_floor=0.05 hard) | 0.731 (cyc 4) | 73% | 100–180 ❌ dive | cyc 6 (37/31/32) |
| ams_v3 (coef=1.0) | z_safety 너무 강 | 0.075 (cyc 10) | 7% | 100–159 잔존 | cyc 19–20 |
| ams_v4 (coef=0.5) | z_floor=0 + z_safety 균형 | 0.816 (cyc 5) | 82% | 4–204 (cyc 9 폭발) | cyc 8 (27/47/27), 14–15 |
| ams_v5 (coef=0.1) | z_safety 약함 | 0.035 (cyc 12) | 3.5% | 19–148 | cyc 19 |
| **ams_v6** | **z_floor=−1000, spawn/goal z=[2.5, 4.5], bound_z=6** | **0.836 (cyc 16)** ⭐ | **84%** | **0 전체** ✅ | cyc 15 (56/44/0) |

### ams_v6 전체 cycle (1–17 정상 종료, cycle 18에서 base_pe_env.py 버그로 fail)
| cyc | a_win | d_win | draw | mission | cap | a_cr_z |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.027 | 0.163 | 0.809 | 7 | 0 | 0 |
| 3 | 0.057 | 0.138 | 0.805 | 16 | 28 | 0 |
| 4 | 0.035 | 0.038 | 0.927 | 9 | 8 | 0 |
| 5 | 0.283 | 0.072 | 0.645 | 78 | 15 | 0 |
| 6 | 0.600 | 0.250 | 0.150 | 120 | 1 | 0 |
| 7 | 0.680 | 0.205 | 0.115 | 136 | 6 | 0 |
| 8 | 0.721 | 0.174 | 0.104 | 145 | 4 | 0 |
| 9 | 0.785 | 0.190 | 0.025 | 157 | 1 | 0 |
| 10 | 0.740 | 0.020 | 0.240 | 148 | 3 | 0 |
| 11 | 0.750 | 0.095 | 0.155 | 150 | 4 | 0 |
| 12 | 0.770 | 0.200 | 0.030 | 154 | 1 | 0 |
| 13 | 0.165 | **0.825** | 0.010 | 33 | 0 | 0 |
| 14 | 0.267 | **0.733** | 0.000 | 54 | 0 | 0 |
| 15 | 0.560 | 0.440 | 0.000 | 112 | 0 | 0 |
| **16** | **0.836** | 0.164 | 0.000 | **168 (84%)** | 0 | **0** | ⭐ BEST
| 17 | 0.815 | 0.185 | 0.000 | 163 | 0 | 0 |

### Cross-eval NPS (ams_v2_swap cycle 5–9, 25 cells × 1000 episode)
- μ(M) = 0.661, std(M) = 0.261, **NPS = 0.000** [95% CI 0, 0.127]
- 결론: 단일 cycle Nash 주장 불가. **range claim**만 가능 ("cycle K–K' 구간에서 balanced policy distribution").

---

## 4. Best baseline 확정 — **ams_v6 cycle 16**

| 지표 | 값 |
|---|---|
| `attacker_winrate` | **0.836** |
| `mission_reach` | 168 / 200 (84%) |
| `defender_winrate` | 0.164 |
| `att_crash_z` | **0** (clean) |
| `def_crash_z` | **0** |

**Why best**:
1. **Clean** — dive crash 0 전체 cycle (vs v4 cycle 9의 a_cr_z=204)
2. **Sustain** — cycle 6–12 모두 a_win 60–79% (7 cycles)
3. **True alternating dynamic** — cycle 13–14에서 defender 회복 (a_win 0.16–0.27) → cycle 15 reset → cycle 16 attacker 다시 peak. Nash 진동이 *작동함* (v4/v2는 cycle 후반 collapse)
4. **Peak는 v4 cycle 5 (82%)보다 강함** — clean + higher peak

**Ckpt 경로**:
```
logs/phasef/ams_v6_highspawn/attacker_v16.pt
logs/phasef/ams_v6_highspawn/defender_v16.pt
logs/phasef/ams_v6_highspawn/cfgs.pkl
```

### 대안 baseline 후보
- cycle 9 — a_win 0.785, mission 157 (79%) — clean peak (cycle 16과 비슷, 약간 약함)
- cycle 15 — a_win 0.560, d_win 0.440 — Nash 가장 가까움 (단 mission 56%만)

---

## 5. 알려진 Issue

### 🔴 base_pe_env.py `torch.roll` 인자 버그 (cycle 18 fail 원인)
```python
# Line 684-685 (현재):
self.defender_state_history = torch.roll(
    self.defender_state_history, shifts=-1, dim=0,   # ← dim → dims 수정 필요
)
```
**Fix**: `dim=0` → `dims=0` (PyTorch `torch.roll` 의 키워드는 plural `dims`).
v6 cycle 18 A.check r4에서 child process 가 새 base_pe_env.py 로드 시 TypeError. cycle 1–17 결과는 코드 변경 전 child 라 valid.

### 🟡 swap 모드 cycle 2 warmup 표시 (분석 시 제외)
- defender fresh PPO init이라 noise 큼
- **best ckpt 선정은 cycle 3 이후만** (코드 comment 명시됨, `train_ams_drl_swap.py` lines 488–502)

### 🟡 forecaster — standalone 검증만 완료
- Phase E.2 buffer/model/trainer 구현 + tests pass
- **PPO simultaneous 학습 미통합** (사용자 결정: rollout 후 separate supervised)

### 🟡 AsymmetricCritic — 인프라 구현됐으나 학습 결과 0
- `--use_asymmetric_critic` flag 동작
- ams_v2~v6 학습 모두 **flag 없이** 진행 (symmetric critic baseline)
- ⚠️ **현 구현 PR-E.1 critic_priv = 10D (vel + ang_vel + quat) 는 BPQL spec과 어긋남**. BPQL spec은 current ground truth pos + vel = 6D. 재구현 필요 (다음 step).

### 🟡 Action delay vs Observation delay 혼동
- yaml에 `simulate_action_latency`, `action_latency_steps` 있지만 **코드에서 사용 0** (inert)
- Observation delay (BPQL τ-delay)는 base_pe_env.py 일부 구현됨 (defender_state_history) 단 버그

### 🟢 FP pool dilution
- cycle K 진행할수록 pool에 약한 옛 ckpt 누적. AMSPB은 FIFO. 현 구현 unbounded.
- NPS = 0.0의 한 원인 가능. 필요시 FIFO trim.

---

## 6. 다음 step (Phase 3 → 4 → 5)

### Phase 3 — τ-delay buffer 완전 구현 (env 책임, CLAUDE.md §6 정합)
- [ ] `torch.roll` 인자 fix (`dim` → `dims`)
- [ ] `tau_delay` env_cfg 키 추가, `_attacker_obs()` 에서 `defender_state_history[0]` 사용
- [ ] Action history `a^a_{t−τ:t−1}` 도 actor obs에 (현 last_action 1개만 → τ 또는 K개)
- [ ] τ=0 backwards compat 검증 (smoke)
- [ ] **τ-sweep eval** — ams_v6 cycle 16 ckpt를 τ ∈ {0, 5, 10, 25, 50, 100}에서 1000 episode 평가 (no retraining). attacker_winrate degradation 곡선.

### Phase 5 — BPQL Asymmetric Critic 재구현
- [ ] `_critic_priv_obs()` 재정의: opponent **current** pos(3) + vel(3) = **6D** (no delay)
- [ ] `CRITIC_PRIV_DIM`: 10 → 6
- [ ] `tests/test_critic_priv_obs.py` 갱신
- [ ] τ-delay와 짝 — τ=0이면 actor obs = critic obs, asymmetric 효과 0. paper에 명시.

### Phase 4 — Forecaster wiring
- [ ] Phase E.2 인프라 + baseline ckpt rollout → supervised train
- [ ] Actor obs에 forecaster output detached append (CLAUDE.md §8: gradient 차단)
- [ ] τ-delay + forecaster 결합 학습

### Paper §5 Ablation (4 rows, BPQL spec 정합)
| 변종 | actor obs | critic obs | 측정 |
|---|---|---|---|
| (1) baseline | full | full (sym) | reference — **ams_v6 cycle 16** |
| (3) τ-delay only | delayed | delayed (sym) | delay degradation |
| (4) τ-delay + forecaster | delayed + forecast | delayed + forecast (sym) | forecaster effect |
| (5) τ-delay + forecaster + BPQL critic | delayed + forecast | full (priv 6D) | BPQL critic effect |

Main claim = (5) full system.

---

## 7. 결정 사항 / 미해결 질문

- [x] **swap mode 채택** — 마지막 학습 = attacker, h2h에서 attacker 우세 측정
- [x] **z_safety_coef = 0.5** (v4/v6 공통) — 적정 sweet spot
- [x] **High z spawn/goal [2.5, 4.5]** — dive crash 0 효과 결정적 (v6)
- [x] **cycle 2 warmup 마킹** — best ckpt 선정 cycle 3 이후만
- [ ] **τ-delay 구현 마무리** (base_pe_env.py 버그 fix + actor obs wiring)
- [ ] **action history 길이 K** — paper τ개 vs 단축 K개 (K=5~10) — 사용자 결정 대기
- [ ] **forecaster 학습 path** — separate supervised vs joint with PPO — 사용자 spec: separate
- [ ] **BPQL critic_priv 6D 재구현** + τ-delay 후 학습

---

*작성: 2026-05-12. 다음 작업: τ-delay buffer 버그 fix → ams_v6 cycle 16 baseline τ-sweep evaluation 시작.*
