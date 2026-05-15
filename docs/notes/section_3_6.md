# §3.6 AMS-DRL Co-Training of the Defender — Implementation

> 이 문서는 paper §3.6 의 이론 framework 를 본 repository (`genesis-pursuit-evasion`)
> 에서 **실제 구현된 형태** 로 다시 정리한다. 이론 원안과 다른 implementation
> 선택은 명시적으로 표시한다.
>
> 코드 출처: `scripts/train_ams_drl_swap.py`, `envs/pe_1v1/env.py`,
> `envs/pe_1v1/scripted_{attacker,defender}.py`, `configs/envs/pe_1v1_ams_v9.yaml`.

---

## 3.6.1 Motivation — Adopted from Paper

Paper §3.6.1 의 동기 그대로 유지. AMS-DRL framework [Xiao & Feroskhan 2024]
을 채택한 이유:

1. **Overfitting 방지**: warm-start (또는 stationary) defender 만으로 학습된
   attacker 는 단순 회피 정책으로 overfit. 실험상 확인됨 — Phase 1.5 attacker
   가 stationary defender 상대로만 학습되면 reactive defender 등장 시 win rate
   급락 (Phase F 초반 진단, `docs/notes/phase_f_summary_2026-05-12.md`).
2. **Adversarial robustness**: 본 repository `ams_v9` 학습 결과, cycle 별
   alternating dynamic 이 실제로 작동 — attacker 가 강해질 때 (cycle 7 a_win
   0.905) 와 defender 가 강해질 때 (cycle 17 cap rate 100%) 가 교차.
3. **Phase-wise stationarity**: 각 phase 에서 한 쪽 policy 를 freeze 함으로써
   §3.1~§3.5 의 가정 (transition 환경의 stationarity) 을 만족.

---

## 3.6.2 Bipartite Training Schedule — Implementation Specifics

### Defender Warm-up — **Skipped (Implementation 차이)**

⚠️ Paper §3.6.2 는 RL pretrained hovering/tracking 정책 $\pi^d_{\text{warm}}$
으로 defender 를 initialize 하지만, **본 repository 는 이를 적용하지 않는다**.
대신:

- **Cycle 1 (Attacker only)**: defender = `StationaryDefender` (hover 자세
  유지, action ≡ 0). Attacker 가 무대응 target 상대로 scratch PPO 학습.
- **Cycle 2 이후**: cycle 1 attacker 가 산출되면, defender 가 그 attacker
  상대로 scratch PPO 학습 시작 (warm-up 없음).

이 단순화는 다음 트레이드오프를 동반한다:

- (+) 구현 단순. 별도 hovering pretrain pipeline 불필요.
- (+) Defender 의 학습 신호가 처음부터 attacker-aware — 무의미한 hover 정책
  으로의 sunk cost 없음.
- (−) Cycle 2 의 defender 학습이 cycle 1 의 *미숙한* attacker (50–100 iter
  scratch) 에 over-fit 될 위험. Cycle 2 결과는 **warm-up cycle 로 간주하고
  baseline 후보에서 제외**한다 (코드 comment: `train_ams_drl_swap.py:488-502`).

### Cycle Ordering — Defender-First (Swap Mode)

Paper §3.6.2 의 pseudocode 는 **Phase A (Attacker turn) 먼저 → Phase B
(Defender turn) 뒤** 순서다. 본 repository 는 **반대로** 구현했다:

```
# 본 repository (scripts/train_ams_drl_swap.py)
for cycle in 1..N:
    if cycle == 1:
        # Cycle 1 only: Attacker scratch vs StationaryDefender
        train_attacker(vs=stationary, init=scratch)
        # NO h2h, NO defender update
        continue
    # Cycle 2+ : B → A swap
    train_defender(vs=cur_attacker)   # B block: defender turn first
    train_attacker(vs=new_defender)   # A block: attacker turn second
    h2h(a_v_K, d_v_K)                 # cycle h2h eval
```

**Swap 의도**: h2h(a_v_K, d_v_K) 에서 attacker 가 *직전 학습된* defender_v_K
에 over-fit 한 상태로 측정 → attacker 측 우세 측정 가능. Paper §3.6.2 의 default
순서 (A then B) 에서는 h2h 시점에 defender 가 가장 최근 학습 결과 (attacker
상대 over-fit) 라 *defender 측 우세* 가 측정된다. 본 repository 의 swap 모드는
attacker-centric paper framing 과 일관성을 확보하기 위함.

**한계**: Empirical Nash gap 측정 시, swap-mode h2h 는 over-fit 방향 편향이
있다. Cross-eval matrix (다른 cycle 간 attacker vs defender pairing) 가
unbiased estimator 에 더 가깝다. §3.6.5 에서 다룬다.

### Forecaster Co-Training — **Not Yet Integrated**

⚠️ Paper §3.6.2 Phase B 의 "Forecaster $f_\psi$ update via stage-weighted
replay" 부분은 본 repository 에서 **아직 통합되지 않았다**. 인프라는 구현 완료
(`agents/forecaster/{model,replay_buffer,trainer}.py`, 27/27 unit tests pass)
이나, alternating loop 안에서 PPO 와 동시에 update 되는 wiring 은 미완성.

본 repository 의 현재 phase F 학습 (`ams_v[2-9]`) 은 모두 **forecaster 없는
baseline** 이며, forecaster 학습은 별도 supervised pipeline (rollout 후
post-hoc training) 으로 진행할 예정. Paper §3.6.2 의 Phase B inner 단계는
다음과 같이 단순화됨:

```
# Phase B (현재 구현, forecaster 미통합)
freeze attacker (cycle K-1)
update defender via standard PPO (+ B.adaptive)
```

### Adaptive Inner Loop (Paper 원안에 없음)

본 repository 는 paper §3.6.2 의 fixed $K_a, K_d$ inner iteration 대신
**adaptive trigger** 를 사용한다 (`scripts/train_ams_drl_swap.py`,
`--attacker_min_winrate`, `--defender_min_winrate`, `--per_stage_max_rounds`).

각 phase 의 inner loop:

```
attacker_train (switch_every iter) → attacker_v_K
A.check: eval(attacker_v_K vs frozen defender_v_{K-1}, 200 episodes)
while att_winrate < attacker_min_winrate AND round < per_stage_max_rounds:
    attacker_train extra round (switch_every iter, resume)
    A.check
```

Defender 측에도 동일한 B.adaptive trigger 가 작동 (`defender_min_winrate`).

**구체 값** (ams_v9 학습):
- `switch_every = 50` PPO iterations (base round)
- `attacker_min_winrate = 0.7` (attacker 가 frozen def 상대 70% mission 도달)
- `defender_min_winrate = 0.5` (defender 가 frozen att 상대 50% capture)
- `per_stage_max_rounds = 5` (최대 5 round 추가 학습)

**Implication**: Threshold 비대칭 (0.7 vs 0.5) 은 학습 *round 수* 에는 큰
차이를 만들지 않는다 — 실제 ams_v9 학습에서 양쪽 모두 cycle 대부분 5 round
풀로 사용. Threshold 가 작동하는 시점은 attacker 가 70% 를 *일찍 통과* 하는
cycle 4–5 정도뿐. 따라서 본 비대칭은 attacker 가 강할 때만 학습 절약을 가져
오는 부수적 효과로 작동.

---

## 3.6.3 Defender Side — Standard Symmetric AC (Paper §3.6.3 일치)

Paper §3.6.3 의 결정 그대로: defender 는 **standard symmetric PPO** 로 학습.
Asymmetric AC 의 motivation 인 *privilege gap* 이 defender 측에서는 부재
(§3.3.1 의 ground radar 가정에 의해 $\tau^d \approx 0$).

본 repository 의 `configs/train/ppo_defender.yaml` 은 rsl-rl 5.x 의 default
PPO + symmetric MLP critic. `obs_groups = {actor: ["policy"], critic:
["policy"]}` (`envs/base_pe_env.py:182-189`).

⚠️ Phase E.1 (`--use_asymmetric_critic` flag) 의 asymmetric critic 인프라는
구현되어 있으나 **본 phase F 학습에서는 사용되지 않음** (default OFF). Critic
side privileged obs 의 의미와 dimension 정의는 paper §3.5 의 attacker-side
critic 에 한정.

### Defender Observation Schema

```
defender_obs (28-dim) = [
    s^d (18-dim): pos_d(world, scaled), vel_d(body), rot_d(quat→9D), ang_vel_d(body),
    s^a_kin (6-dim): rel_pos_a(body), rel_vel_a(body),   # NO mission goal
    last_action (4-dim)
]
```

`envs/pe_1v1/scripted_defender.py::_build_defender_policy_obs()` 와
`envs/pe_1v1/env.py::_defender_obs()` 가 mirror schema. Mission goal
$g_{\text{mission}}$ 은 **defender 측에 비노출** (paper §3.6.3 그대로).

---

## 3.6.4 Domain Randomization — **Minimal Scope**

⚠️ Paper §3.6.4 의 broad DR 카테고리는 본 repository 에서 **최소 범위로만
적용**. 현재 학습 (`pe_1v1_ams_v9.yaml`) 에서 randomize 되는 것:

| 카테고리 | Randomization | Range |
|---|---|---|
| Attacker spawn | $x, y, z$ uniform box | `attacker_init_pos: x ∈ [0.5, 4.0], y ∈ [-1, 1], z ∈ [2.5, 4.5]` |
| Defender spawn | 동일 | `defender_init_pos: x ∈ [-4.0, -0.5], y ∈ [-1, 1], z ∈ [2.5, 4.5]` |
| Mission goal | uniform box | `g_mission.box: x ∈ [-10, -4], y ∈ [-1.5, 1.5], z ∈ [2.5, 4.5]` |
| Initial velocity | fixed zero | `init_lin_vel: [0, 0, 0]` |

미적용 카테고리:
- ❌ Mass / inertia randomization (SysID-based DR)
- ❌ Motor response 시정수
- ❌ Sensor noise
- ❌ Communication delay $\tau$ randomization (Phase 3a infra 만 존재,
   학습 시 미사용)

이 단순화는 Phase F 의 학습 목표가 *환경 안에서 의미 있는 baseline 정책 확보*
이기 때문. 실제 DR 은 paper main contribution 인 forecaster + asymmetric
critic 실험 (Phase 3–5) 진입 후 적용 예정.

---

## 3.6.5 Empirical Validation — `ams_v9` Results

### 학습 setup

```yaml
env_yaml: configs/envs/pe_1v1_ams_v9.yaml
attacker_train_yaml: configs/train/ppo_attacker.yaml
defender_train_yaml: configs/train/ppo_defender.yaml
num_envs: 1024
switch_every: 50            # train_ams_drl_swap.py default
max_cycles: 20              # default
n_h2h_episodes: 200         # cli override (script default 400)
attacker_min_winrate: 0.7   # cli (default 0.0 = adaptive OFF)
defender_min_winrate: 0.5   # cli (default 0.0)
per_stage_max_rounds: 5     # cli override (default 1)
backend: gpu (RTX 3090)
seed: 0
```

> ⚠️ Script default 와 다른 값은 `train_ams_drl_swap.py` 호출 시 cli 로 명시
> 전달되어야 한다. Phase F 학습의 실제 cli 는 `logs/phasef/ams_v9/cycle_1/`
> 의 cfgs.pkl 또는 wandb run config 에서 재구성 가능.

Reward (`pe_1v1_ams_v9.yaml`) — relaxed-crash variant of stage_4 base:
- `progress_max: 0.01`, `action_smooth: 1e-2` symmetric, `omega_coef: 5e-4`
  symmetric, `attitude_coef: 2e-3`, `velocity_coef: 0.01`, `z_safety_coef:
  0.05`, `bound_safety_coef: 0.01`
- `R_mission_win: 20`, `R_captured: -7`, `R_self_crash: -10`,
  `R_timeout: -1`, `R_capture_win: 20`
- `action_mag_coef: 5e-4` per side (Phase F 추가, saturate 차단 의도 — 효과는
  미미했음, 아래 분석)

### Cycle-wise alternating dynamic 결과

| cycle | attacker_winrate | defender_winrate | mission | captured | comment |
|---:|---:|---:|---:|---:|---|
| 2 (warmup) | 0.000 | 0.050 | 0 | 0 | def stationary 잔재, baseline 제외 |
| 5 | 0.860 | 0.140 | 172 | 27 | attacker peak |
| **6** | **0.630** | **0.275** | 126 | **49 (24.5%)** | balanced |
| 7 | 0.905 | 0.080 | 181 | 16 | attacker peak (2nd) |
| **8** | **0.460** | **0.520** | 92 | **104 (52%)** | ⭐ **best balanced** |
| 13 | 0.675 | 0.010 | 135 | 2 | attacker dom |
| 14 | 0.693 | 0.228 | 140 | 46 | balanced |
| **17** | 0.000 | **1.000** | 0 | **200 (100%)** | ⭐ defender peak |
| 20 | 0.328 | 0.363 | 66 | 57 | balanced (draw 31%) |

핵심:

1. **Cycle 8 — 양쪽 정책 모두 의미 있게 학습된 best balanced point**.
   Capture rate 52%, mission rate 46%, draw 2%. Paper §5 의 forecaster
   실험 baseline 으로 적합 (양쪽 모두 능력 보유 → τ-delay 효과 측정에 의미).
2. **Cycle 17 — defender 완전 dominant (cap 100%)**. AMS-DRL 의
   alternating 이 실제로 두 쪽 모두 강해지는 phase 를 만들어냄을 보여줌.
3. **Cycle 4–7 → 8 → 17 — 진동 패턴**. Paper §3.6.5 의 "convergence" 는
   본 setting 에서 *정적 fixed point* 가 아니라 **alternating Nash 진동** 으로
   관찰됨. Cross-eval matrix 가 진정한 baseline 평가에 필요.

### Action Distribution — Saturation 분석

`scripts/eval_head_to_head.py` 가 cycle h2h 마다 attacker/defender action
distribution (per-dim mean/std/sat>.95) 을 기록. Ams_v9 의 saturation:

| | T_norm sat>.95 | wx_ref sat>.95 |
|---|---|---|
| attacker (cycle 4–20 mean) | 0.95–0.99 | 0.94–0.97 |
| defender (cycle 4–20 mean) | 0.87–0.99 | 0.84–0.97 |

→ **양쪽 모두 action saturation 90%+ 지속**. 이는 다음 요인의 결합:

1. **Task-structural**: mission goal 까지의 평균 거리 ~9 m, episode 5 s,
   $v_{\max} = 15$ m/s. Attacker 가 mission 빨리 도달하려면 max thrust + max
   body rate 사용이 *physically optimal*. Defender chase 도 동일.
2. **PPO Gaussian mean unconstrained**: rsl-rl 의 `GaussianDistribution` (
   tanh squashing 미사용) 에서 actor mean 은 ±∞ 학습 가능. Env 의 action
   clip `[-1, 1]` 이 mean = ±5+ 정도까지 흡수 → sampled action 거의 saturate.
3. **`std_range: [1e-6, 1.0]` cap 작동**: log_std 폭주는 차단 (`std_range`
   인자, paper §3.5 의 *training-time numerical guard*). 단 saturate 자체는
   못 막음 — mean 발산이 원인.
4. **`action_mag_coef` 효과 미미**: ams_v8 (coef=5e-3) 은 defender chase 학습
   을 무너뜨려 cycle 8+ 에서 defender 자살률 90% 폭증. Ams_v9 (5e-4) 는
   학습 dynamic 보존하나 saturation 95% → 96% 정도로만 변동.
5. **Tanh-mean variant**: `TanhMeanGaussianDistribution` (`ppo_attacker_tanhm.
   yaml`) 시도 (cycle 2–4 검증). Cycle 2 에서 sat 0%, cycle 3 36%, cycle 4
   96% — 결국 same saturated regime 으로 수렴. 학습 *속도* 만 느려짐.

**결론** (paper §5 reporting plan): Saturation 은 **task structural property**
로서 보고. Action distribution table 을 paper appendix 에 포함. Sim-to-real
(§7) 단계에서 thrust scale 또는 quartic penalty 로 별도 처치.

---

## 3.6.6 Consistency with §3.1–§3.5 — Verified Items / Open Items

Paper §3.6.5 의 4 항목 중 본 implementation 에서 검증된 것과 open:

### ✅ Verified

1. **Phase-wise stationarity**: Cycle 안에서 한 쪽 policy freeze.
   `_attacker_args_for_defender_train()` 와 `_defender_args_for_attacker_train()`
   helper 가 frozen pretrained ckpt 를 자식 train script 에 전달
   (`scripts/train_ams_drl_swap.py:251-300`). Within-cycle MDP 는 stationary.

2. **Critic privilege 무결성 (vacuous in current learning)**: Phase F 학습에
   서 asymmetric critic flag 가 OFF 이므로 paper §3.5 Prop 3.17 의 unbiasedness
   는 *trivially* 성립 (actor = critic obs).

### ⚠️ Open / Future

3. **Phase-boundary non-stationarity**: Cycle K → K+1 전환 시 $\mathcal{P}^d_
   {\pi^d}$ 분포 shift. Paper §3.6.5 는 AMS-DRL framework 채택으로 수렴
   보장, 본 setting 의 empirical 수렴은 §5 검증 예정. 현재 ams_v9 의 cycle
   16–20 진동 (cap 100% → cap 14 → balanced → cap 57) 은 **fixed-point
   수렴이 아닌 alternating Nash 진동** 으로 해석. Cross-eval matrix 평가가
   진정한 수렴 판정에 필요 (`scripts/compute_nps.py` — NPS metric 정의).

4. **Forecaster co-adaptation**: 미통합. Phase 4 (Phase E.2 인프라 활용,
   별도 supervised pipeline) 에서 다룬다.

---

## 3.6.7 Implementation File Map

| 역할 | 파일 |
|---|---|
| Alternating orchestrator (swap mode) | `scripts/train_ams_drl_swap.py` |
| Alternating orchestrator (canonical A→B) | `scripts/train_ams_drl.py` |
| Attacker training (PPO) | `scripts/train_attacker.py` |
| Defender training (PPO) | `scripts/train_defender.py` |
| Cycle h2h evaluation | `scripts/eval_head_to_head.py` |
| Cross-eval Nash proximity score | `scripts/compute_nps.py` |
| Env (base + 1v1) | `envs/base_pe_env.py`, `envs/pe_1v1/env.py` |
| Scripted opponents (warm-start helpers) | `envs/pe_1v1/scripted_attacker.py`, `scripted_defender.py` |
| Reward / spawn yaml | `configs/envs/pe_1v1_ams_v9.yaml` |
| PPO yaml (attacker/defender) | `configs/train/ppo_{attacker,defender}.yaml` |
| Forecaster infra (미통합) | `agents/forecaster/{model,replay_buffer,trainer}.py` |
| Asymmetric critic infra (미사용) | `envs/base_pe_env.py:182-189`, `envs/pe_1v1/env.py::_critic_priv_obs()` |

---

### 📎 §3.6 implementation 의 핵심 결정 (요약)

- **Defender warm-up X** — stationary defender 로 cycle 1 시작 (paper 와 차이)
- **Cycle ordering — Defender-first (swap mode)** — paper 의 A→B 와 반대
- **Forecaster co-training X** — phase F 학습은 forecaster 없는 baseline
- **Asymmetric critic X (Phase F)** — `--use_asymmetric_critic` 인프라만, 학습에선 OFF
- **Adaptive inner loop** — A.adaptive / B.adaptive trigger (paper 원안에 없음)
- **Best baseline candidate**: `ams_v9 cycle 8` (a 46% / d 52% / cap 52%)
- **Saturation** — task structural property, 본 phase 에서 보고만, sim-to-real 단계로 미룸
