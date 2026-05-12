# TODO.md — Genesis Pursuit-Evasion (CORL 2026 forecaster paper)

> **이 문서는 진척 관리만.** 코딩 규칙은 [CLAUDE.md](./CLAUDE.md).
> 완료 항목은 `[x]` 체크. 우선순위: 🔴 높음 / 🟡 중간 / 🟢 낮음.
> Phase 간 dependency 위반 PR은 reject.

---

## 📛 학습 단계 명명 (2026-05-12 정정 — CORL forecaster paper 방향)

CORL 2026 submission 기준 재정리. Phase 3 (paper main system)을
sub-phase로 쪼갬 — τ-delay + Forecaster + BPQL critic이 통합 실험.

| 새 명명 | 작업 | 상태 |
|---|---|---|
| **Phase 1** | Baseline (stationary defender/attacker pretraining) | ✅ 완료 |
| **Phase 2** | AMS-DRL co-training (mission+evasion baseline) | 🟡 cycle 20까지 + cross-eval |
| **Phase 3** | τ-delay + Forecaster + BPQL critic (paper main system) | 🔴 |
| Phase 3a    | τ-delay buffer 구현 (env-side, ~30 LOC) | 🔴 |
| Phase 3b    | 데이터 수집 (~50K episode, zarr) | 🔴 |
| Phase 3c    | Forecaster 학습 (Flow Matching, 20-step Euler solver) | 🔴 |
| Phase 3d    | BPQL critic 코드 수정 (10D extra → 6D ground truth) | 🔴 |
| Phase 3e    | 5 variant 학습 + τ sweep eval (paper main result) | 🔴 |
| Phase 4     | Sim-to-real | 🟢 future |
| Phase 5     | 문서/공개 | 🟢 future |

기존 §번호 alias (paper/section3 본문 참조):
- old §1.0/1.5 → Phase 1
- old §2 (τ-delay) → Phase 3a
- old §3 (Forecaster) → Phase 3b + 3c
- old §4 (Asymmetric Critic) → Phase 3d
- old §5 (AMS-DRL) → Phase 2
- old §6 (평가) → Phase 3e
- old §7 (Sim-to-real) → Phase 4

---

## 📌 논문과의 매핑 (CORL 2026)

### 학습 PoV
* Attacker (mission-driven evader-pursuer hybrid) 학습
* Defender: Phase 2 AMS-DRL alternating로 baseline 정책 학습 후 frozen

### Main contribution (3 paired components)
1. **Asymmetric Goal-Delay POMDP framework**:
   - Augmented state (ideal POMDP 정의): $\bar{s}^a_t = (s^a_t, s^d_{t-\tau}, a^a_{t-\tau:t-1}, g_{mission})$
   - **Practical implementation (옵션 C, 2026-05-12 결정)**:
     - Actor obs = $(s^a_t, s^d_{t-\tau}, \hat{s}^d_t, g, \text{last\_action})$ (fixed dim)
     - Forecaster가 action history $(a^a_{t-\tau:t-1})$ 압축 → $\hat{s}^d_t$로 변환
     - Actor는 $\hat{s}^d_t$ 통해 augmented state 정보 간접 활용 (DFBT 스타일)
   - **Frame convention (PR-M, 2026-04-29)**: Actor obs / critic_priv / forecaster output
     모두 **attacker body frame relative**. sim-to-real 친화 + 일관성.
   - Mission goal 있는 setting에서 communication delay 모델링
   - **Paper §3.4 revision 필요**: augmented state는 이론적 정의, forecaster가 압축 변환

2. **Forecaster $f_\psi$ (Conditional Flow Matching)**:
   - Generative model: $\hat{s}^d_t \sim p_\psi(\cdot | \text{history H=50 of } s^a, s^d, a^a, \tau)$
   - Trajectory output (τ steps), last step만 actor에 전달
   - Output: attacker body frame relative (rel_pos_d, rel_vel_d 6D)
   - **Action history 처리는 forecaster 내부** (transformer + attention mask)
   - **τ-conditional 단일 모델**: 학습 시 τ ~ U(0, 50) random

3. **BPQL-style asymmetric critic** [Kim et al. 2023 NeurIPS]:
   - Actor: delayed augmented state 입력
   - Critic: ground truth current state $s^d_t$ (no delay) 직접 사용 — **6D body frame relative** (rel_pos_d_GT, rel_vel_d_GT)
   - Beta Q-function: augmented state 폭증 회피 + 안정 학습
   - Deployment 시 critic 안 씀 (actor만)

### 평가 — 5 variant ablation (τ sweep)

**Actor obs dim (옵션 C 기준)**:
- Variant 1, 2, 4: 31 dim (현재 ATTACKER_OBS_DIM 그대로)
- Variant 3, 5: 31 + 6 = 37 dim (forecaster output 6D 추가)

| Variant | Actor obs | Critic obs | 측정 |
|---|---|---|---|
| (1) No-delay baseline | $s^a, s^d_t, g, a^a_{t-1}$ | (same as actor) | reference |
| (2) Delay + naive | $s^a, s^d_{t-\tau}, g, a^a_{t-1}$ | (same as actor) | degradation, info 부족 baseline |
| (3) Delay + Forecaster only | $s^a, s^d_{t-\tau}, \hat{s}^d_t, g, a^a_{t-1}$ | (same as actor) | forecaster alone |
| (4) Delay + BPQL critic only | $s^a, s^d_{t-\tau}, g, a^a_{t-1}$ | $s^a, s^d_t, g, a^a_{t-1}$ | critic alone (actor obs는 variant 2와 동일) |
| (5) **Full: Forecaster + BPQL critic** | $s^a, s^d_{t-\tau}, \hat{s}^d_t, g, a^a_{t-1}$ | $s^a, s^d_t, g, a^a_{t-1}$ | **main claim** |

**Note**: Variant 2의 "정보 부족" 상태는 의도된 baseline. Paper에서 "naive
delay handling 한계" 보여주는 비교 포인트.

### Paper claim 정직성
"AMS-DRL alternating로 mission+evasion 환경에서 reasonable한 양쪽 정책 확보.
완벽한 Nash 아닌 asymmetric strength baseline (e.g., att 35-79% / def 17-47%)
사용. 이 baseline 위에서 forecaster + BPQL critic의 effect 측정."

### 주요 참고
- **BPQL** [Kim et al. 2023, NeurIPS]: critic 디자인 직접 기반
- **AMS-DRL** [Xiao & Feroskhan 2024, IEEE TAI]: alternating co-training
- **AMSPB** [Sánchez Roncero et al. 2025]: pursuit-evasion baseline, 가장 가까운 경쟁작
- **DFBT** [Wu et al. 2025, ICML]: forecaster baseline 비교 (deferred)
- **Flow Matching** [Lipman et al. 2023]: forecaster generative model
- **SimpleFlight** [Chen et al. 2025, RA-L]: sim-to-real (Phase 4, deferred)

자세한 인용 → `paper/section3_4_v9.md` (revised, BPQL 정정 반영 필요)

---

## 🔧 PR 히스토리 (요약)

> 디테일은 git log + docs/plans/, docs/notes/ 참조. CORL 진척 추적용 요약.

### Phase 1 baseline 관련
- **PR-A~C** (2026-04-26): 1v1 env 골격, defender/attacker pretraining 스크립트
- **PR-D** (2026-04-26): Defender 학습 안정화 (chase reward, spawn 분리)
- **PR-E** (2026-04-28): Terminal reward minimax (per-side crash/oob split)
- **PR-F** (2026-04-28): Attacker omega_coef (자세 발산 차단)
- **PR-G** (2026-04-28): Attacker dense incentive (goal_dense scale 매치)
- **PR-H** (2026-04-28): Body-rate PD controller (action space 변경)
- **PR-I~K** (2026-04-28~29): PD 게인 튜닝 + 검증 + trajectory tracking
- **PR-L** (2026-04-29): Hover RL validation env
- **PR-M** (2026-04-29): Obs schema body-frame 변환

### Phase 2 (AMS-DRL) 관련
- **PR-N** (2026-04-29): Subprocess orchestrator MVP (train_ams_drl.py)
- **PR-O** (~2026-05-01): SquashedGaussian distribution (PPO 안정성, 후속 폐기)
- **PR-Q** (~2026-05-10): **TanhMean Gaussian distribution** (현 사용 중, PR-O 대체)
  - `mean = tanh(mlp_output)` 구조적 [-1, 1] 강제
  - Standard Gaussian log_prob (atanh 없음, PPO ratio 폭발 차단)
  - ams_s2_v[10-13] NaN crash 해결
- **PR-E.1** (~2026-05-08): Asymmetric critic 인프라 (`obs_groups`, `--use_asymmetric_critic`)
  - **⚠️ 현 구현이 BPQL 의도와 어긋남** — Phase 3d에서 정정

### Phase F (pure PE, **폐기 결정 2026-05-11**)
- 다양한 reward/yaml 조합 시도 (z_safety_coef, soft boundary, 10s episode 등)
- 모든 시도 alternating 발산 또는 attacker dive crash
- **결정**: Paper §5 ablation으로 "g_mission 없으면 alternating 발산" 보고

---

## ⚠️ 검증 의무 — **모두 통과 (2026-04-26)**

* [x] 🔴 **V1**: Genesis 한 Scene에서 batched dual-drone control
* [x] 🔴 **V1.5**: differential closed-loop dispatch
* [x] 🔴 **V2**: 두 OnPolicyRunner alternating 시 GPU/optimizer 격리
* [x] 🔴 **V3**: obs_groups로 actor/critic obs 분리

---

## ✅ Phase 0 — 새 레포 부트스트랩 (완료 2026-04-25)
## ✅ Phase 1 — 1v1 환경 골격 (완료 2026-04-29)

> 디테일은 기존 TODO 히스토리 + docs/plans/ 참조. Phase 0/1은 안정화.

---

## Phase 2 — AMS-DRL Co-training (Mission+Evasion Baseline) 🟡 진행 중

### 목표
Forecaster 실험의 baseline 정책 (attacker + defender 페어) 확보.
완벽한 Nash 균형은 paper claim 아님 — "reasonable한 양쪽 정책" + 다양한 강도
스펙트럼 확보가 목표.

### 진행 상황 (2026-05-12)

**완료**:
- [x] Subprocess orchestrator (PR-N)
- [x] Swap 모드 도입 (cycle K attacker vs cycle K-1 defender, 2026-05-11)
- [x] **Phase F (pure PE) 실험 → 폐기 결정.** paper §5에 ablation으로 보고
- [x] ams_v6 cycle 15 완료 시점 관찰: cycle 9 peak (a_win 79%, mission 79%, dive 0)
  - cycle 6~12 sustain (a_win 60~79%) ⭐
  - cycle 13-14 defender 회복 → cycle 15 attacker 회복 (진동 정상)
  - dive crash 0 전체 ✅

**진행 중**:
- [ ] 🔴 ams_v6 cycle 20까지 학습 (현재 cycle 15 진행, ~30분 남음)

**남은 작업**:
- [ ] 🔴 **Cross-eval matrix** (cycle 20 후)
  - 후보 cycle: **TBD** (cycle 20 학습 결과 보고 결정)
  - Each cell: 1000 episode eval
  - NPS metric (noise-corrected)
  - Output: best balanced pair (forecaster experiment baseline) 1~2개 확정
- [ ] 🟡 결과 정리 → `docs/notes/phase2_ams_v6_results.md`

### Baseline 정책 활용 계획 (Phase 3 입력)

| 용도 | 사용 ckpt |
|---|---|
| Forecaster 학습 데이터 수집 | cycle {TBD} 페어 (다양성, ~4 pairs) |
| τ sweep eval 시 frozen opponent | cycle {TBD} defender (cross-eval로 확정) |
| Variant 1~5 attacker 학습 시작점 | **fresh init** (over-fit 회피) |

### 진입 조건 (Phase 3로)
- ams_v6 cycle 20 완료
- Cross-eval 통해 NPS > 0.5 + μ ∈ [0.3, 0.7] cycle 1개 이상 발견
- 못 찾으면 v7 (spawn 조정) launch 검토

---

## Phase 3 — τ-delay + Forecaster + BPQL critic 🔴 (paper main system)

> 통합 실험. 3a-3e sub-phase로 쪼개 진행. 5 variant ablation이 paper main result.

### Phase 3a — τ-delay Buffer 구현 🔴

**목표**: Defender state observation을 τ-step 지연시킴. Actor만 영향.

- [ ] 🔴 `envs/base_pe_env.py` 수정 (~30 LOC)
  ```python
  # __init__
  self.tau_delay = int(env_cfg.get("tau_delay", 0))
  # Option a (fixed τ per variant): scalar int, not per-env tensor
  buf_size = max(1, self.tau_delay + 1)
  self.defender_state_history = torch.zeros(
      (buf_size, B, 6), device=device  # 6D kinematics (pos+vel), body frame
  )

  # step end: ring buffer
  self.defender_state_history = torch.roll(
      self.defender_state_history, shifts=-1, dim=0
  )
  self.defender_state_history[-1] = current_def_kinematics

  # reset_idx: 전체 채움 (warmup garbage 방지)
  self.defender_state_history[:, envs_idx] = (
      current_def_state[envs_idx].unsqueeze(0)
  )
  ```
- [ ] 🔴 `envs/pe_1v1/env.py::_attacker_obs()` 수정 (~10 LOC)
  - `delayed_def_state = self.defender_state_history[0]`
  - `rel_pos_d`, `rel_vel_d` 계산 시 delayed state 사용
- [ ] 🔴 yaml schema
  - `pe_1v1_ams_v2.yaml`: `tau_delay: 0` (default, backward compat)
- [ ] 🔴 Test: `tests/test_tau_delay.py`
  - τ=0: 기존 동작과 동일 (backward compat)
  - τ=5: actor obs가 5 step 전 defender 위치 받음
  - Reset 후 history 정확 초기화
  - Multi-env (1024) shape consistency

**변경하지 않는 것**: defender obs, critic obs (3d에서 별도), reward 계산,
termination

**예상 시간**: 2~3시간
**진입 조건**: Phase 2 cycle 20 완료

### Phase 3b — 데이터 수집 🔴

**목표**: Forecaster 학습용 trajectory 데이터셋 (~50K episode, ~6GB)

**Frame 결정 (2026-05-12)** — **world frame raw 저장**:
- 환경 내부 mirror buffer (`self.defender_pos/vel`, `self.attacker_pos/vel/quat`)
  와 동일 frame. body-frame rotation은 zarr 저장 단계가 아니라
  forecaster→actor wiring 단계에서 (env가 이미 `_attacker_obs`에서 함).
- 이유 (i): Phase 3a `defender_state_history` 가 world frame raw로 운영 →
  inference 시 forecaster output을 그대로 `defender_state_history[0]`
  슬롯에 끼워 넣을 수 있어 wiring 0 LOC.
- 이유 (ii): generative target stability — body frame은 attacker pose에
  의존하는 time-varying frame이라 history 50 step 학습이 어려움.
  world frame은 frame-invariant trajectory라 flow matching 수렴 친화.
- 이유 (iii): sim-to-real 친화 — mocap/IMU/EKF가 모두 world frame.
- pos/vel scaling은 저장 시 적용 (`obs_scales` 곱한 dimensionless ~[-1, 1]).

- [ ] 🔴 `scripts/collect_forecaster_data.py` 신규
  - Input: AMS-DRL cycle ckpt pairs (Nash 후보 ~4 cycles)
  - Output: Zarr chunked storage
  - Per transition: (s^a, s^d, a^a, done, episode_id, timestep)
  - **World frame raw + obs_scales 적용** (위 결정)
- [ ] 🔴 Stochastic rollout (TanhMean σ 그대로) — 행동 다양성
- [ ] 🔴 Validation split: held-out 1 pair (~25%)
- [ ] 🔴 Storage spec:
  ```
  forecaster_data.zarr/
  ├── episodes/
  │   ├── pair_XX/  # cycle 별
  │   │   ├── s_a    (n_ep, 1000, 18+6+3+4=31)  world frame, scaled
  │   │   │                                      (pos+vel+rot+ang+rel+g+last_a)
  │   │   ├── s_d    (n_ep, 1000, 6)             world frame, scaled (pos+vel)
  │   │   ├── a_a    (n_ep, 1000, 4)             raw action [-1, 1]^4
  │   │   ├── attacker_quat (n_ep, 1000, 4)      raw quat (forecaster
  │   │   │                                      context 시 frame embedding용)
  │   │   └── episode_length (n_ep,)
  ├── metadata.json
  └── frame.json     # {"defender_state": "world", "scale": "obs_scales"}
  ```

**예상 시간**: 1~2시간
**진입 조건**: Phase 3a 완료 + Phase 2 cross-eval로 cycle 선정

### Phase 3c — Forecaster 학습 (Flow Matching) 🔴

**목표**: τ-conditional flow matching forecaster 학습

**Frame contract** (Phase 3b 결정 연동):
- Input/output 모두 **world frame raw + obs_scales 적용**.
- Forecaster output ŝ^d_t (6D, world frame) → `defender_state_history[0]`
  슬롯에 직접 wiring → `_attacker_obs`가 기존 body-frame rotation 그대로 사용.
- 즉 forecaster는 frame transform을 학습하지 않음 — 학습 부담 ↓.

- [ ] 🔴 `agents/forecaster/flow_matching.py` (사용자 기존 코드 base 활용)
  - History encoder (transformer): H=50 step → context embedding
  - Velocity field $v_\theta(s, t, \text{context}, \tau)$
  - τ embedding (sinusoidal 또는 learned)
  - Output: trajectory (τ, 6) **world frame, scaled**
  - Attention mask (variable episode length)
- [ ] 🔴 `agents/forecaster/trainer.py`
  - Loss: $\|v_\theta(s_t, t, c) - (s_1 - s_0)\|^2$
  - Random τ ∈ U(0, 50) per batch
  - Optimizer: Adam, lr=1e-4
  - Batch size: 256
  - Training: ~100K steps
- [ ] 🔴 `agents/forecaster/inference.py`
  - **Euler solver, 20 steps**
  - Batch inference (1024 envs)
  - Latency target: < 20ms per call (benchmark 필요)
- [ ] 🔴 Test: `tests/test_forecaster.py`
  - Shape consistency
  - τ-conditional behavior (다른 τ → 다른 output)
  - Gradient isolation (forecaster grad가 PPO loss로 안 흐름)
  - Inference latency benchmark

**Hyperparameters**:
- d_model: 128
- n_layers: 4 (encoder) + 2 (velocity field)
- n_heads: 4

**예상 시간**: 1일 (코드 + 학습 + validation)
**진입 조건**: Phase 3b 데이터 수집 완료

### Phase 3c.1 — Forecaster Validation (다양하게)

- [ ] 🔴 **Held-out pair validation** (필수)
  - 학습: cycle {TBD-train} pairs
  - Validation: cycle {TBD-val} pair
  - Metric: MSE per τ ∈ {0, 10, 25, 50}
- [ ] 🔴 **τ generalization** (필수)
  - 학습 범위 외 포함 τ ∈ {0, 5, 10, 25, 50, 75, 100}
  - Graceful degradation 확인
- [ ] 🟡 **Multi-step prediction error**
  - τ=50일 때 각 step (t-49, ..., t-1, t) 별 MSE
  - Plot: step vs MSE
- [ ] 🟡 **Trajectory visualization**
  - 1 episode rollout: actual vs predicted defender trajectory
- [ ] 🔴 **Downstream task validation** (Phase 3e와 통합)
  - Forecaster 사용 → mission rate 측정
- [ ] 🟢 (선택) Stochastic prediction variance
- [ ] 🟢 (선택) Calibration

### Phase 3d — BPQL-style Asymmetric Critic 🔴

**목표**: Critic이 학습 시 ground truth current state $s^d_t$ 직접 사용
(BPQL Kim et al. 2023). 기존 PR-E.1 구현이 BPQL 의도와 어긋남 → 정정.

**현 상태** (PR-E.1 폐기 부분):
- 현재: `_critic_priv_obs() → opp_vel(3) + opp_ang_vel(3) + opp_quat(4) = 10D`
  (추가 dynamics info, BPQL 의도 아님)
- 목표: `_critic_priv_obs() → defender_pos(3) + defender_vel(3) = 6D`
  (ground truth current state, no delay)

- [ ] 🔴 `envs/pe_1v1/env.py::_critic_priv_obs()` 수정
  ```python
  # BEFORE (PR-E.1, 폐기)
  # return [opp_vel, opp_ang_vel, opp_quat]  # 10D extra dynamics

  # AFTER (BPQL 의도, body frame relative — actor obs와 일관)
  rel_pos_d_GT = rotate_to_body_frame(
      self.defender_pos - self.attacker_pos, self.attacker_quat
  )  # 3D body frame
  rel_vel_d_GT = rotate_to_body_frame(
      self.defender_vel - self.attacker_vel, self.attacker_quat
  )  # 3D body frame
  return torch.cat([rel_pos_d_GT, rel_vel_d_GT], dim=1)  # 6D, ground truth, body frame
  ```
- [ ] 🔴 `envs/pe_1v1/cfgs.py::CRITIC_PRIV_DIM` 10 → 6
- [ ] 🔴 `tests/test_critic_priv_obs.py` 업데이트
  - τ=0 case: actor obs와 critic_priv 의미 동일성
  - τ>0 case: critic_priv는 current GT, actor는 delayed (다름)
- [ ] 🔴 Deployment 안전성: critic_priv 키 deploy 시 build 안 됨 검증
- [ ] 🟡 Paper §3.5 BPQL [Kim et al. 2023] 인용 추가 (별도 task)

**BPQL theoretical 정당화 (paper §3.5)**:
- Belief projection: $\bar{Q}^{\bar\pi}(\bar{s}, a) \approx \mathbb{E}_{P(s_t|\bar{s}_t)}[Q^{\bar\pi}_\beta(s_t, a)]$
- Beta Q-function: original state space (augmented 폭증 회피)
- Contraction property (BPQL Proposition 3.2) → 안정 학습
- 우리 setting: $s_t$ = current defender state, $\bar{s}_t$ = augmented attacker obs

**변경 위험**: 기존 use_asymmetric_critic=True ckpt invalidated. 다행: ams_v6은
False로 학습 중 → 영향 없음. Variant 4, 5 모두 fresh init이라 무관.

**예상 시간**: 2~3시간 (코드 + 테스트)
**진입 조건**: Phase 3a 완료 (τ-delay 인프라 필요)

### Phase 3e — 5 Variant 학습 + τ Sweep Eval 🔴 (paper main result)

**목표**: Paper main result 생성. 5 variants × multi-seed × τ sweep.

#### 학습 spec

- **Defender**: Phase 2 cross-eval로 확정된 best balanced cycle (frozen)
- **Attacker**: 5 variant 각각 **fresh init** 학습
- **Episode 길이**: 5s (ams_v6 호환)
- **환경**: ams_v2 yaml + τ-delay (variant별 τ_train 다름)
- **τ_train (Option a, Fixed per variant, 2026-05-12 결정)**:
  - Variant 1: τ_train = 0 (no-delay baseline)
  - Variant 2, 3, 4, 5: τ_train = **25** (sweep 중간값)
  - Eval: τ ∈ {0, 10, 20, 30, 40, 50} sweep (모든 variant)
  - Paper claim: "Actor trained at τ=25, evaluated across τ — OOD generalization 측정"
- **Frame convention**: 모두 attacker body frame relative (Actor / Critic / Forecaster output)
- **Actor obs dim (옵션 C)**:
  - Variant 1, 2, 4: 31 dim (현재 그대로)
  - Variant 3, 5: 37 dim (forecaster output 6D 추가, body frame)
  - Action history는 forecaster 내부 처리 (actor obs에 직접 미포함)
- **PR-Q TanhMean distribution 그대로 사용**

#### Multi-seed (Option C, 2026-05-12 합의)

| Variant | Seeds | Total runs |
|---|---|---|
| (1) No-delay baseline | 3 | 3 |
| (5) Full system | 3 | 3 |
| (2), (3), (4) | 1 each | 3 |
| **Total** | | **9 학습** |

#### 학습 병렬 (GPU 1대 + 컴퓨터 2개)

```
Wave 1 (병렬): Variant 1 seed 0  |  Variant 5 seed 0
Wave 2 (병렬): Variant 1 seed 1  |  Variant 5 seed 1
Wave 3 (병렬): Variant 1 seed 2  |  Variant 5 seed 2
Wave 4 (병렬): Variant 2          |  Variant 3
Wave 5 (단독): Variant 4
```

학습 1 run ~4시간 → 5 waves × ~4시간 = **20시간**

#### τ Sweep (각 variant마다)
- τ ∈ {0, 10, 20, 30, 40, 50} (6 conditions)
- 1000 episode eval per condition
- 30분/variant × 9 = ~5시간

#### 작업
- [ ] 🔴 학습 스크립트 prep (variant별 yaml + launch script)
- [ ] 🔴 Variant 1 (no-delay baseline) × 3 seeds
- [ ] 🔴 Variant 5 (full) × 3 seeds
- [ ] 🟡 Variant 2, 3, 4 (action history 결정 후) × 1 seed each
- [ ] 🔴 τ sweep eval all variants
- [ ] 🔴 Main result table (τ × variant matrix)
- [ ] 🔴 Main figure (τ sweep plot, 5 curves, shading for multi-seed)
- [ ] 🟡 Statistical tests (variant 5 vs others)
- [ ] 🟡 Ablation analysis (forecaster vs critic vs both)

#### Action history 디자인 — 옵션 C 확정 (2026-05-12)
- Actor obs에 action history 직접 미포함 (last_action 1개만)
- Forecaster 내부에서 history 압축 처리 (transformer + attention mask)
- Paper §3.4 augmented state 정의 revision 필요 (별도 task)

#### 예상 시간 (CORL 데드라인 핵심)
- 학습: 20시간
- Eval: 5시간
- Plot + analysis: 1일
- **Total: ~3일**

**진입 조건**: Phase 3a, 3c, 3d 모두 완료

---

## Phase 4 — Sim-to-Real (deferred) 🟢

> 너무 멀어서 CORL 범위 외. SimpleFlight 5 factors, ROS bridge, mocap.
> 디테일은 기존 TODO 보존.

---

## Phase 5 — 문서/공개 🟢

> CORL submission 후. README full, reproduction guide, model weights release.

---

## ✅ 완료 (Done)

* [x] 비판적 검토 + 새 레포 결정 (2026-04-25)
* [x] Phase 0 부트스트랩 완료 (2026-04-25)
* [x] Phase 1 PR-A~M (2026-04-26~29)
* [x] Phase 2 PR-N orchestrator (2026-04-29)
* [x] PR-Q TanhMean distribution (~2026-05-10) — 현 사용 중
* [x] Phase F (pure PE) 폐기 결정 (2026-05-11)
* [x] CORL 방향 paper claim 재정리 (2026-05-12)
* [x] BPQL 정합 확인 + critic_priv 수정 spec 확정 (2026-05-12)
* [x] Forecaster spec 확정 (Flow Matching, 20-step solver, zarr) (2026-05-12)
* [x] 5 variant ablation table 확정 (2026-05-12)
* [x] Action history 처리 옵션 C 확정 — forecaster 내부 처리, actor obs dim 고정 (2026-05-12)
* [x] **Phase 3a 코드 완료** — τ-delay ring buffer in base_pe_env (world frame raw), env.py _attacker_obs uses history[0], yaml/cfg schema, 12 unit tests passing. tau_delay=0 default = backward compat (2026-05-12)
* [x] **Phase 3d 코드 완료** — BPQL critic_priv 10D→6D body-frame relative (rel_pos_GT+rel_vel_GT), test_critic_priv_obs.py 재작성 (RandomWalkDefender로 GT vs delayed unconditional contrast), 11 tests passing. use_asymmetric_critic=False default = backward compat (2026-05-12)
* [x] Phase 3b/3c frame 결정 — **world frame raw 저장**, forecaster output world frame, body-frame rotation은 env _attacker_obs에서 (2026-05-12)
* [x] τ_train Option a 확정 — variant별 fixed τ, eval만 sweep (2026-05-12)
* [x] Frame convention 통일 — 모두 attacker body frame relative (2026-05-12)

---

## 📎 외부 참고

* **BPQL** [Kim et al. 2023, NeurIPS]: critic 디자인 직접 기반
* **AMS-DRL** [Xiao & Feroskhan 2024, IEEE TAI]: alternating co-training
* **AMSPB** [Sánchez Roncero et al. 2025]: 가장 가까운 경쟁작
* **DFBT** [Wu et al. 2025, ICML]: forecaster baseline (deferred)
* **Flow Matching** [Lipman et al. 2023]: generative model 기반
* **SimpleFlight** [Chen et al. 2025, RA-L]: sim-to-real (deferred)
* 전체 28편 → `paper/section3_4_v8.md` (v9 revision 필요)

---

## 🎯 CORL 2026 진행 순서 (Critical Path)

1. **오늘**: ams_v6 cycle 20 완료 대기 (~30분)
2. **D+1**: Cross-eval matrix + Phase 3a (τ-delay 구현)
3. **D+2**: Phase 3b 데이터 수집 + Phase 3d BPQL critic 수정
4. **D+3~4**: Phase 3c Forecaster 학습 + validation
5. **D+5~7**: Phase 3e 5 variant 학습 (병렬 wave)
6. **D+7~8**: τ sweep eval + plot + paper draft

CORL 마감 임박. 일정 빡빡하지만 가능 범위.

---

*작업 시작 전 Phase의 V (Validation) 항목 먼저 ✅. Phase 간 dependency 위반 PR은 reject.*