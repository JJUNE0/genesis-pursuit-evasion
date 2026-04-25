# Section 4 — Method (v3)

> **목적**: §3 의 formulation 을 작동하는 학습 시스템으로 instantiate 한다. 새 이론 0건, 모든 설계 결정에 §3 reference 표기.
>
> **§3 와의 관계**: §3 의 모든 결정사항 — sufficient statistic $x_t^{\text{goal}}$ (Prop 3.4), forecaster $f_\psi$ (Def 3.10), asymmetric AC (Def 3.15), AMS-DRL bipartite alternating (§3.6) — 그리고 v8 의 K1–K12 정정사항이 §4 에서 그대로 구현된다.
>
> **v2 → v2-final 변경**: trajectory sampling mix (§4.4.1), DR bounds (§4.5.1), stage-weighted replay $\alpha$ default 의 결정 근거 (§4.4.4), simulator final choice (§4.7.1) 를 lock. TBD 4개 → 1개로 축소 (§4.8).
>
> **v2-final → v3 변경 (코드 정합성 점검)**: 사용자가 이미 구축·검증한 `genesis_quadrotor_simulator` (Nova drone hovering 검증, SimpleFlight 4 factor 구현 완료) 를 base 로 활용함을 §4.7 에 명시. 자가 점검 결과 4건의 정정 사항을 반영: (i) rotation parameterization 정합성 — §3.1.2 rotation matrix 결정과 SimpleFlight Factor 1 (9D rot replaces 4D quat) 가 정합 → 18-dim 구성 명시 (§4.2.2/§4.2.3), (ii) Base simulator actor `[128, 128]` 2-layer vs 본 논문 3-layer/256 의 차이 명시 (§4.2.2), (iii) Action history default $k = 0$ 은 SimpleFlight 의 $k = 1$ 표준과의 의도적 deviation 으로 framing 정정 (§4.2.2), (iv) 사용자 base simulator 의 reward / DR / SimpleFlight 구현이 본 논문 base 와 직접 통합 (§4.5, §4.7).

---

## 4.1 System Overview

### 4.1.1 Three-Component Architecture (Attacker)

§3.4–§3.5 의 결정사항을 그대로 instantiate:

1. **Forecaster $f_\psi$** (§3.4 Def 3.10): 지연된 defender 관측 + 즉시 관측되는 attacker 궤적/action 이력 → 현재 시점 defender state 추정 $\hat{s}^d_t$.
2. **Actor $\pi_\phi$** (§3.5 Def 3.15, K1): forecaster 의 terminal output + attacker 자기 상태 + mission goal → action distribution.
3. **Critic $Q_\theta$** (§3.5 Def 3.15): training-time 의 simulator oracle 로부터 ground-truth defender full state $\tilde{s}^d_t$ 를 직접 입력받음 (richer-than-observation privilege).

세 component 는 end-to-end 가 아닌 **staged training** 으로 학습된다 (§4.4 의 bipartite schedule). Forecaster 는 phase 별 freeze/update 가 명확히 구분된다 (§3.6 K4).

### 4.1.2 Inference-Time Data Flow (Deployment)

배포 시 simulator oracle 미사용. Attacker drone 의 onboard pipeline:

```
[step t 시작]
  1. Sensor 갱신:           s^a_t  ← onboard IMU + state estimator
  2. Comm relay 갱신:        s^d_{t-τ} ← delayed defender kinematic projection
  3. Buffer 조회:            s^a_{t-τ:t-1}, a^a_{t-τ:t-1}, g_mission
  4. Forecaster forward:    ŝ^d_t = f_ψ(s^a_t, s^a_{t-τ:t-1}, s^d_{t-τ}, a^a_{t-τ:t-1})
  5. Actor forward:         a^a_t ~ π_φ(· | ŝ^d_t, s^a_t, g_mission)   [k=0 default; §4.2.2]
  6. Action 송출:           low-level FCU 에 RPM command
  7. Buffer 업데이트
[step t+1]
```

**핵심**: Critic $Q_\theta$ 는 inference 에 등장하지 않는다 — 학습 종료 후 폐기 (§3.5 Remark 3.20 항목 1).

### 4.1.3 Training-Time Data Flow

학습 중 simulator 가 동시 제공:

- **Actor / Forecaster 입력**: §4.1.2 와 동일 (inference 분포와 일치)
- **Critic 입력**: 위 + simulator oracle 의 ground-truth $\tilde{s}^d_t$ (full attitude 포함, 18-dim)

**중요**: Forecaster output $\hat{s}^d_t$ 는 critic 에 들어가지 않는다 — actor 만 forecaster 를 거친다. 이것이 §3.5 의 핵심 비대칭.

### 4.1.4 Defender Side Concurrent Loop

§3.6.3 에 따라 별도 PPO loop:

- **Defender Actor** $\pi^d(a^d \mid \tilde{s}^d_t, s^a_t)$ + **Defender V-critic** $V^d_\theta(\tilde{s}^d_t, s^a_t)$ (K6)
- 입력: $(\tilde{s}^d_t, s^a_t)$ — actor·critic 동일 (privilege gap 부재)
- $g_{\text{mission}}$ **비노출** (§3.6.3)

Bipartite alternating (§4.4) 에 의해 한 번에 attacker-side 또는 defender-side 만 update; 각 phase 내부에서 한 쪽 policy 는 freeze 되어 §3.1–§3.5 의 stationarity 가정 유지.

### 4.1.5 Block Diagram (Figure 1)

```
[Simulator: Genesis]
  │
  ├─ Attacker rollout buffer ─→ Forecaster f_ψ ─→ ŝ^d_t ──┐
  │   (s^a_{t-τ:t}, a^a_{t-τ:t-1},                          │
  │    s^d_{t-τ}, g_mission)                                ↓
  │                                                  Actor π_φ ─→ a^a_t
  │                                                  (deployable)│
  │                                                              │
  ├─ Oracle ground-truth ─────→ Critic Q_θ ←────────────────────┘
  │   (s^a_t, ~s^d_t, g_mission)   (training only, discarded after)
  │
  └─ Defender side ──→ π^d ──→ V^d_θ
      (s^a_t, ~s^d_t)        (PPO, no g_mission)
```

---

## 4.2 Network Architectures

### 4.2.0 Hyperparameter Anchoring on Prior Work

본 절의 default hyperparameter 들은 다음 reference 들의 published 값을 anchor 로 채택, §5 sensitivity sweep 의 search space 를 좁게 유지한다.

| Component | Reference | 본 논문 채택 |
|---|---|---|
| **Base simulator** | 사용자 기존 `genesis_quadrotor_simulator` (Nova hovering 검증됨) | scene/buffer/step·reset/reward/DR layer 그대로 활용 (§4.7.1) |
| Forecaster transformer | **DFBT** [Wu et al. 2025], MuJoCo $\Delta = 128$ 까지 검증 | Architecture, optimizer, dropout 그대로 차용 |
| Actor / Critic MLP scale | **SimpleFlight** [Chen et al. 2025], Crazyflie sim-to-real | hidden dim, layer 수 (단 base simulator default 와의 차이 §4.2.2 에 명시) |
| **State representation (rotation)** | **SimpleFlight Factor 1**: 9D rotation matrix replaces 4D quaternion (이미 base simulator `_quat_to_rotmat_flat` 으로 구현) | $\tilde{s}^d, s^a$ 18-dim = pos(3)+vel(3)+rot_mat(9)+ang_vel(3) |
| Action history 사용 여부 | SimpleFlight 의 obs 에 `last_actions` 포함 ($k = 1$ 표준) | 본 논문 default $k = 0$ — forecaster redundancy 논거에 따른 의도적 deviation (§4.2.2) |
| Trajectory sampling class | SimpleFlight (smooth polynomial + zigzag) | trajectory mix base (§4.4.1) |
| Drone DR bounds | SimpleFlight Crazyflie 2.1 setting (base simulator `_apply_domain_rand` 에 일부 구현) | DR table base (§4.5.1) |
| Simulator | Genesis [Genesis-Embodied-AI 2024], drone hovering RL tutorial 이 Kaufmann 2023 align | base simulator (§4.7.1) — 사용자 기존 구현 활용 |
| Decision Transformer scale [Chen et al. 2021] | MuJoCo $L=3, H=1, d=128$ | 작은 task 용 — DFBT 가 long-delay 에 더 적합 |

**원칙**: Architecture-level + sim-to-real hyperparameter 는 reference 와 동일하게 lock; **본 논문 contribution 영역의 hyperparameter** ($\lambda_{\text{aux}}$, action history $k$, replay weighting $\alpha$, communication delay $\tau$) 만 §5 에서 sweep. 이는 (i) compute 예산 통제, (ii) reproducibility, (iii) 본 논문의 contribution 이 architecture/DR tuning 이 아니라 problem formulation (goal-delay POMDP) 과 forecasting framework 임을 명확히 하기 위함.

### 4.2.1 Forecaster $f_\psi$ — Transformer Encoder-Decoder

§3.4 Def 3.10 의 direct trajectory forecaster 를 sequence-to-sequence transformer 로 구현. Wu et al. (2025) DFBT framework base, §3.4 Remark 3.16 의 4가지 specialization 반영 — 특히 **attacker 궤적 conditioning** 이 token sequence 로 자연스럽게 들어감.

#### Input tokenization

각 time step 별 attacker 측 정보 (자기 상태 + action) 를 token 으로:

$$
e_{t-\tau+i} = \mathrm{Embed}(s^a_{t-\tau+i},\ a^a_{t-\tau+i}), \quad i = 0, 1, \ldots, \tau-1
$$

마지막 step (action 미결정) → placeholder $\emptyset$:

$$
e_t = \mathrm{Embed}(s^a_t,\ \emptyset)
$$

Defender delayed observation 을 별도 token: $e_{\text{def}} = \mathrm{Embed}(s^d_{t-\tau})$.

전체 input sequence = $(e_{\text{def}}, e_{t-\tau}, \ldots, e_t)$, 길이 $\tau + 2$. Sinusoidal positional encoding (Vaswani 2017).

#### Encoder / Decoder

- **Encoder**: $L_{\text{enc}}$-layer self-attention with causal mask, pre-norm + GELU FFN
- **Decoder**: $L_{\text{dec}}$-layer cross-attention 으로 encoder output 을 query → defender state sequence
- **Output sequence**: $(\hat{s}^d_{t-\tau+1}, \ldots, \hat{s}^d_t)$, 길이 $\tau$
- **Parallel decoding** (DFBT 와 동일) — recursive 회피의 architectural enforcement

#### Output head

각 timestep 별 6-dim regression: $(\hat{p}^d, \hat{v}^d) \in \mathbb{R}^6$ (kinematic projection $\phi$ 일치).

#### Hyperparameters — Locked Defaults

| Hyperparameter | Symbol | **Default (lock)** | §5 Sensitivity Range | 근거 |
|---|---|---|---|---|
| Encoder layers | $L_{\text{enc}}$ | **4** | – | DFBT 의 10 total → 4+4=8 |
| Decoder layers | $L_{\text{dec}}$ | **4** | – | 동일 |
| Total layers | $L$ | **8** | $\{4, 8, 12\}$ | DFBT (10) 보다 약간 작게 — 본 setting 의 reactive defender single closed-loop kernel + attacker traj explicit conditioning 으로 task complexity 가 single-agent MuJoCo 보다 낮음 |
| Attention heads | $H$ | **4** | – | DFBT 와 동일. per-head $d/H = 64$ standard |
| Hidden dim | $d$ | **256** | – | DFBT 와 동일 |
| FFN dim | $d_{\text{ff}}$ | **$4d = 1024$** | – | standard ratio |
| Dropout (attn/res/hidden) | – | **0.1** all | – | DFBT 와 동일 |
| Max sequence length | – | $\tau_{\max} + 2$ = 202 | – | $\tau_{\max}=200$ |
| Aux loss weight | $\lambda_{\text{aux}}$ | **0.5** | $\{0.1, 0.5, 1.0\}$ | 본 논문은 terminal 이 actor 입력으로 사용 — DFBT 의 uniform 보다 낮춤 |

#### Optimizer (DFBT 차용 lock)

| | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-4}$ (warmup 5% + cosine decay) |
| Weight decay | $1 \times 10^{-4}$ |
| Betas | $(0.9, 0.999)$ |
| Gradient clipping | $\|g\|_2 \leq 1.0$ |
| Batch size | 256 |

#### $g_{\text{mission}}$ 미포함의 architectural enforcement

Forecaster 는 $g_{\text{mission}}$ 없이 학습 (token embedding 단계 차단). Defender dynamics $\tilde{\mathcal{P}}^d$ 가 $g_{\text{mission}}$ 비의존 (§3.4 Prop 3.7) — separation of concerns.

---

### 4.2.2 Attacker Actor $\pi_\phi$ — MLP

§3.5 Def 3.15 의 actor signature (K1) 를 그대로 구현.

#### Input — Action History 결정

| Component | **Default ($k=0$)** | Ablation |
|---|---|---|
| $\hat{s}^d_t$ | 6-dim, forecaster terminal output (kinematic projection $\phi$: $(p^d, v^d)$) | – |
| $s^a_t$ | **18-dim** = pos(3) + vel(3) + **rot_mat(9)** + ang_vel(3) — SimpleFlight Factor 1 (9D rot replaces 4D quat) | – |
| $a^a_{t-k:t-1}$ | **omitted ($k = 0$)** | $k \in \{1, 4, 10\}$ |
| $g_{\text{mission}}$ | 3-dim | – |

**Default actor input**: $(\hat{s}^d_t, s^a_t, g_{\text{mission}}) \in \mathbb{R}^{27}$ (no action history).

**Note (코드 정합성)**: 사용자 base simulator (`envs/base_drone_env.py::_update_observation`) 는 `last_actions` 1-step ($k = 1$) 을 obs 에 포함하는 SimpleFlight 표준 layout 을 사용 중. 본 논문은 $k = 0$ 으로 의도적 deviation — forecaster $f_\psi$ 가 attacker 궤적 + action 이력을 internal token sequence 로 처리하므로 actor input 에서 redundant 라는 논거 (아래).

#### $k = 0$ Default 의 근거

1. **Forecaster 가 이미 attacker 궤적 internal 처리**: §3.4 Def 3.10 의 forecaster 입력에 $a^a_{t-\tau:t-1}$ 가 token 으로 들어가 self-attention 으로 처리됨. Actor 가 또 받는 것은 redundancy — forecaster output $\hat{s}^d_t$ 가 이미 "attacker 궤적 conditioned defender estimate" 다.
2. **SimpleFlight 표준 ($k = 1$) 과의 의도적 deviation**: SimpleFlight (Chen 2025), Kaufmann 2023 모두 actor 의 obs 에 last action 1-step 포함 ($k = 1$). 본 논문은 forecaster 가 forecaster 입력단에서 action sequence 를 처리한다는 점에서 actor 의 last action 도 redundant 로 보고 $k = 0$ default 채택. Action smoothness 는 reward 의 action difference regularization (§4.3.2 의 $r_{\text{smooth}}$ 항) 으로 처리 — 이는 SimpleFlight Factor 3 와 동일.
3. **Sufficient statistic 측면**: §3.5 Def 3.15 의 $x_t^{\text{goal}}$ 는 sufficient — forecaster 통과 후 $\hat{s}^d_t$ 와 $s^a_t, g_{\text{mission}}$ 만으로 충분 조건 만족.

#### Ablation 의 의의 (§5)

$k \in \{0, 1, 4, 10\}$ 비교는 다음 경험적 question 들에 답한다:

- **$k = 0$ vs $k = 1$**: SimpleFlight 표준과의 직접 비교 — 본 논문의 forecaster 가 last action 정보를 actor 단에서 redundant 하게 만드는지 검증 (가장 critical ablation)
- **$k = 0$ vs $k = 4$**: actor 측 action smoothness regularization 효과 — reward 단의 $r_{\text{smooth}}$ 만으로 충분한지, actor input 단에서도 추가 신호가 필요한지
- **$k = 0$ vs $k = 10$**: long-history input 이 PPO advantage estimation 안정화에 도움 되는지 — long-delay regime 에서 noisy advantage 를 actor 가 history 로 직접 보완 가능한지
- **특히 long-delay regime ($\tau \geq 100$) 에서**: $k > 0$ 이 forecaster error $\epsilon^{\text{dir}}(\tau)$ 를 actor 단에서 보완하는 역할을 가지는지 — Cor 3.14 의 belief error 가 actor 학습에 미치는 영향을 actor input redundancy 로 완화 가능한지에 대한 empirical question

→ Default $k = 0$ 이 robust 하다는 가설 검증; 만약 $k > 0$ 가 의미 있는 개선을 보이면 §4.2.2 의 default 를 update.

#### Architecture (SimpleFlight 표준)

| | Value |
|---|---|
| Layers | **3** |
| Hidden dim | **256** |
| Activation | tanh |
| Normalization | LayerNorm |
| Output head | $\mu \in \mathbb{R}^4$ + state-independent learnable log-std |

**Note (코드 정합성)**: 사용자 base simulator (`scripts/train.py::get_train_cfg`) 는 default actor 가 `[128, 128]` 2-layer / hidden 128. 본 논문은 단일 hovering 보다 PE setting 이 복잡 (attacker traj + defender forecast + adversarial dynamics) 하므로 capacity 확장. §5 ablation 에서 `[128, 128]` vs `[256, 256, 256]` 비교 가능.

#### Output

- Tanh-Gaussian: $a^a_t = \tanh(\mu + \sigma \cdot \epsilon)$, $\epsilon \sim \mathcal{N}(0, I)$
- Action space: $a^a = (T, \omega_x, \omega_y, \omega_z) \in \mathbb{R}^4$ (CTBR — SimpleFlight, Kaufmann 2023 표준)

---

### 4.2.3 Attacker Critic $Q_\theta$ — MLP with Privileged Input

§3.5 Def 3.15 의 richer-than-observation privilege.

#### Input

$$
\text{critic input} = (s^a_t,\ \tilde{s}^d_t,\ a^a_t,\ g_{\text{mission}}) \in \mathbb{R}^{43}
$$

| 항목 | Dim | 구성 (§3.1.2 + SimpleFlight Factor 1) |
|---|---|---|
| $s^a_t$ | 18 | pos(3) + vel(3) + **rot_mat(9)** + ang_vel(3) — attacker full state |
| $\tilde{s}^d_t$ | 18 | pos(3) + vel(3) + **rot_mat(9)** + ang_vel(3) — **defender full** state (attitude 포함, simulator oracle) |
| $a^a_t$ | 4 | attacker action (CTBR) |
| $g_{\text{mission}}$ | 3 | mission goal |

**Rotation parameterization**: §3.1.2 가 rotation matrix 를 명시했고, SimpleFlight Factor 1 (9D rotation matrix replaces 4D quaternion) 가 quadrotor sim-to-real 의 검증된 표준 — 사용자 base simulator (`_quat_to_rotmat_flat`) 도 이를 그대로 채택. 본 논문 18-dim 결정은 두 reference 와 정합.

#### Architecture (locked)

| | Value |
|---|---|
| Layers | 3 |
| Hidden dim | **512** (actor 보다 크게) |
| Activation | tanh + LayerNorm |
| Output | scalar Q-value |

#### 선택 근거

§3.5 Prop 3.16 (Lambrechts 2025: aliasing 제거) + Prop 3.17 (Ebi 2025: unbiased) + Prop 3.18 (Kim 2023 BPQL specialization).

#### Defender policy parameter 비주입

§3.5 Remark 3.20 항목 3: critic 은 $\pi^d$ neural network parameter 미접근 — phase 마다 $\pi^d$ update 되어도 critic input signature 고정 → §3.6.5 Item 4 unbiasedness 보존.

---

### 4.2.4 Defender Actor / Critic — Standard PPO

§3.6.3 의 privilege gap 부재 → standard symmetric AC.

**Defender Actor** $\pi^d(a^d \mid \tilde{s}^d_t, s^a_t)$: 3-layer MLP, hidden 256, tanh-Gaussian output.

**Defender V-critic** $V^d_\theta(\tilde{s}^d_t, s^a_t)$ (K6): 3-layer MLP, hidden 512, scalar output.

**$g_{\text{mission}}$ 비노출**: Implementation level 차단.

---

## 4.3 Training Objectives

### 4.3.1 Forecaster Loss

#### Terminal regression loss

$$
\mathcal{L}_{\text{terminal}} = \mathbb{E}\bigl[\|\hat{s}^d_t - s^d_t\|_2^2\bigr]
$$

#### Auxiliary trajectory loss

$$
\mathcal{L}_{\text{trajectory}} = \mathbb{E}\!\left[\frac{1}{\tau-1}\sum_{i=1}^{\tau-1}\|\hat{s}^d_{t-\tau+i} - s^d_{t-\tau+i}\|_2^2\right]
$$

#### Total

$$
\mathcal{L}_{\text{fore}} = \mathcal{L}_{\text{terminal}} + \lambda_{\text{aux}} \cdot \mathcal{L}_{\text{trajectory}}, \qquad \lambda_{\text{aux}} = 0.5\ \text{(default)}
$$

#### Gradient flow boundary (Phase-aware)

- **Phase A** (K4): forecaster freeze. `f_ψ.eval()` + `torch.no_grad()`.
- **Phase B** (K5): attacker $\pi_\phi, Q_\theta$ freeze. Forecaster 만 stage-weighted replay (§4.4.4) 로 학습.

---

### 4.3.2 Attacker Actor-Critic Loss (PPO + Asymmetric)

#### Critic loss

$$
\mathcal{L}_{\text{critic}} = \mathbb{E}\!\left[\bigl(Q_\theta(s^a_t, \tilde{s}^d_t, a^a_t; g_{\text{mission}}) - Q^{\text{target}}_t\bigr)^2\right]
$$

Target: GAE-$\lambda$. Bootstrapping next-state critic 도 ground-truth $\tilde{s}^d_{t+1}$ 사용.

#### Advantage estimation

$$
\hat{A}_t = Q_\theta(s^a_t, \tilde{s}^d_t, a^a_t; g_{\text{mission}}) - V^a_t
$$

Baseline $V^a_t$: 별도 V-critic (SimpleFlight 권장).

#### Actor loss (PPO clipped)

$$
\mathcal{L}_{\text{actor}} = -\mathbb{E}\!\left[\min\bigl(r_t(\phi)\hat{A}_t,\ \mathrm{clip}(r_t(\phi), 1-\epsilon, 1+\epsilon)\hat{A}_t\bigr)\right]
$$

$\epsilon = 0.2$ (PPO standard).

#### Action smoothness regularization (SimpleFlight 차용)

$$
r_{\text{smooth}}(t) = -c_{\text{smooth}} \cdot \|a^a_t - a^a_{t-1}\|_2^2
$$

$c_{\text{smooth}}$ 값: SimpleFlight 의 reward design 차용 (구체값 §4.5 reward design 부록).

#### 핵심 비대칭 enforcement

| 항목 | Actor pipeline | Critic pipeline |
|---|---|---|
| Defender state | $\hat{s}^d_t$ via $f_\psi$ (6-dim, projected) | $\tilde{s}^d_t$ via simulator oracle (18-dim, full) |
| Latency | forecaster 추정 (deployment 와 동일) | 0 (현재 시점 ground-truth) |

#### Total attacker loss

$$
\mathcal{L}_{\text{att}} = \mathcal{L}_{\text{actor}} + c_v \cdot \mathcal{L}_{\text{critic}} - c_e \cdot \mathcal{H}[\pi_\phi]
$$

$c_v = 0.5$, $c_e = 0.01$ (PPO standard).

---

### 4.3.3 Defender PPO Loss

Standard symmetric PPO:

$$
\mathcal{L}_{\text{def}} = \mathcal{L}_{\text{actor}}^{\pi^d} + c_v \cdot \mathcal{L}_{V^d} - c_e \cdot \mathcal{H}[\pi^d]
$$

---

## 4.4 AMS-DRL Bipartite Schedule

### 4.4.1 Defender Warm-up Pretraining

§3.6.2 의 사전 단계 — bipartite alternating 시작 전에 reactive tracking 능력 확보.

#### Task

임의 3D 좌표 sequence 를 reactive 추적:
- **Input**: $(\tilde{s}^d_t, p^a_{\text{ref}, t}, v^a_{\text{ref}, t})$
- **Action**: 4-dim CTBR
- **Reward**: $-\|p^d_t - p^a_{\text{ref}, t}\|_2^2 - \alpha_v \|v^d_t - v^a_{\text{ref}, t}\|_2^2$ + control penalty

#### Trajectory sampling mix (LOCKED)

SimpleFlight (Chen 2025) 의 trajectory class 를 base 로 하되, 본 setting 의 reactive defender 가 추적해야 할 motion 을 추가:

| Pattern | Source | **Weight (lock)** |
|---|---|---|
| Smooth polynomial 5th-order spline | SimpleFlight 차용 | **0.4** |
| Zigzag / infeasible (sharp turns) | SimpleFlight 차용 | **0.3** |
| Sinusoidal ($A \sim \mathcal{U}(1,5)$ m, $\omega \sim \mathcal{U}(0.5, 2)$ Hz) | classical control benchmark | **0.2** |
| Reactive-inspired (random walk + occasional sharp reversal) | 본 setting 특수 (attacker proxy) | **0.1** |

**근거**: SimpleFlight 의 두 class (smooth poly + zigzag) 가 quadrotor reactive tracking 의 minimum 을 cover (Crazyflie zero-shot sim-to-real 검증). 본 setting 에서는 attacker 가 evasive maneuver 를 보일 수 있으므로 reactive-inspired class 10% 추가. Sinusoidal 은 frequency-domain 다양성 확보용 classical baseline.

§5 ablation: 3-way mix 비교 — SimpleFlight-only (0.5/0.5) vs 본 default vs reactive-heavy (0.2/0.2/0.2/0.4).

#### Episode 구성

- 길이: 500–1000 steps at 50 Hz (10–20 초)
- Initial state: arena 내 random spawn

#### Optimizer & budget

- PPO with hyperparameters from SimpleFlight
- $\sim 5 \times 10^6$ environment steps
- Convergence criterion: tracking error < 0.5 m (avg over last 100 episodes)

#### Output

$\pi^d_{\text{warm}}$ — checkpoint 저장 후 bipartite alternating initial defender.

---

### 4.4.2 Bipartite Alternating Hyperparameters

#### Outer loop

- $N_{\text{outer}} = 15$ (TBD via §5 monitoring)
- 매 iteration: Phase A → Phase B

#### Phase A (Attacker Turn)

| Setting | Value |
|---|---|
| Environment steps $K_a$ | $1 \times 10^6$ |
| Frozen | $\pi^d, f_\psi$ (K4) |
| Updated | $\pi_\phi, Q_\theta$ via §4.3.2 |
| Buffer | phase-내 rollout (on-policy PPO) |

#### Phase B (Defender Turn + Forecaster Co-Training)

| Setting | Value |
|---|---|
| Environment steps $K_d$ | $1 \times 10^6$ |
| Frozen | $\pi_\phi, Q_\theta$ (K5) |
| Updated (defender) | $\pi^d, V^d_\theta$ via §4.3.3 |
| Updated (forecaster) | $f_\psi$ via stage-weighted replay (§4.4.4) |
| Forecaster fine-tune freq | 매 $10^5$ env steps 마다 1 epoch (Phase B 당 10 회) |

#### Pseudocode

```python
# Pretraining
π^d_warm ← train_hovering_tracker(steps=5e6, dist=traj_mix_v2_final)

# Initialize
π^d ← π^d_warm
π_φ, f_ψ, Q_θ, V^d_θ ← random_init()
buffer ← StageWeightedReplayBuffer(α=0.7, max_size=5e6)

# Bipartite alternating
for outer_iter in range(N_outer):  # = 15

    # ─── Phase A: Attacker Turn ────────────────
    freeze(π^d), freeze(f_ψ)
    for step in range(K_a):
        rollout(π_φ vs π^d, DR sampled)
        update(π_φ, Q_θ) via PPO+AsymCritic

    # ─── Phase B: Defender Turn + Forecaster ───
    freeze(π_φ), freeze(Q_θ)
    for step in range(K_d):
        rollout(π_φ vs π^d, DR sampled)
        update(π^d, V^d_θ) via PPO
        buffer.append(transitions, phase_idx=outer_iter)
        if step % 1e5 == 0:
            update(f_ψ, buffer.sample(weighted=α))

    log_reward_plateau()
    log_empirical_nash_gap()
```

---

### 4.4.3 Convergence Criterion

#### Main: Fixed iteration budget

$N_{\text{outer}} = 15$.

**근거**: 단순/재현, AMS-DRL 의 asymmetric goal-delay 세팅 closed-form Nash gap 부재 (§3.6.5 Item 2), 학습 비용 통제.

#### Monitoring (보고용 only)

- **Reward plateau**: attacker reward EMA last 3 outer iter $|\Delta|/|\bar R| < 0.05$
- **Empirical Nash gap**: best-response evaluation
  - Phase A 종료 후 frozen $\pi^d$ 에 대한 fresh $\pi^a_{\text{BR}}$ → $R^a_{\text{BR}}$
  - Phase B 종료 후 frozen $\pi^a$ 에 대한 fresh $\pi^d_{\text{BR}}$ → $R^d_{\text{BR}}$
  - Gap = $R^a_{\text{BR}} - R^a_{\text{current}}$ (mirror 양쪽)

→ §5 reward + Nash gap curve plot — Xiao 2024 symmetric 결과의 본 setting transfer empirical 검증.

---

### 4.4.4 Stage-Weighted Replay (Forecaster)

§3.4 Remark 3.15 메커니즘 구체화.

#### Buffer 구조

```python
class StageWeightedReplayBuffer:
    def __init__(self, α=0.7, max_size=5e6):
        self.transitions = []
        self.α = α
        self.max_size = max_size

    def append(self, transition, phase_idx):
        transition['phase_idx'] = phase_idx
        self.transitions.append(transition)
        if len(self.transitions) > self.max_size:
            self.transitions.pop(0)

    def sample(self, batch_size, current_phase):
        weights = [self.α ** (current_phase - t['phase_idx'])
                   for t in self.transitions]
        weights = normalize(weights)
        idx = np.random.choice(len(self.transitions),
                                batch_size, p=weights)
        return [self.transitions[i] for i in idx]
```

#### $\alpha$ Default 결정 — Effective Half-Life 기반

Phase $p$ sample 추출 확률: $P(p) \propto \alpha^{N_{\text{outer, current}} - p}$. Effective half-life $h$ = $\log 0.5 / \log \alpha$.

| $\alpha$ | half-life (phases) | 해석 |
|---|---|---|
| 0.5 | 1.0 | 매우 강한 recency — forgetting risk |
| **0.7** | **1.94** | **default** — 약 2 phase half-life |
| 0.8 | 3.11 | – |
| 0.9 | 6.58 | uniform 에 근접 |
| 1.0 | $\infty$ | uniform replay (no recency bias) |

**Default $\alpha = 0.7$** 결정 근거:

본 논문의 $N_{\text{outer}} = 15$ 하에서:
- Half-life $\geq N_{\text{outer}}/2 \approx 7$: uniform 에 근접 — 분포 mismatch
- Half-life $\leq 1$: 거의 latest-only — catastrophic forgetting risk
- **Half-life 2 phase ($\alpha = 0.7$)**: sweet spot — 최근 phase 우세하지만 직전 2–3 phase 정보 유의미 보존

→ §3.4 Remark 3.15 의 "최근 $\pi^d$ 분포 match (recency)" 와 "이전 phase 분포 보존 (forgetting 회피)" 의 두 목표를 balance.

#### Buffer & Sampling

- $N_{\text{buffer}} = 5 \times 10^6$ transition (memory 한계)
- **Default $\alpha = 0.7$**
- §5 ablation: $\alpha \in \{0.5, 0.7, 0.9, 1.0\}$ — 1.0 (uniform) baseline 으로 "recency weighting 효과 있는가" 동시 측정

---

## 4.5 Domain Randomization

§3.6.4 의 카테고리별 구체 분포. 모든 phase 에서 적용.

### 4.5.1 Drone Physical Parameters — SimpleFlight 차용 (LOCKED)

본 카테고리는 SimpleFlight (Chen et al. 2025) 의 Crazyflie 2.1 zero-shot sim-to-real 검증 분포를 그대로 차용.

| Parameter | Symbol | **Distribution (lock)** | Source |
|---|---|---|---|
| Mass | $m$ | $\mathcal{U}(0.8 m_{\text{nom}}, 1.2 m_{\text{nom}})$ | SimpleFlight |
| Inertia (per-axis) | $I_{xx}, I_{yy}, I_{zz}$ | $\mathcal{U}(0.8 I_{\text{nom}}, 1.2 I_{\text{nom}})$ | SimpleFlight |
| Motor time constant | $\tau_m$ | $\mathcal{U}(0.01, 0.05)$ s | SimpleFlight |
| Thrust coefficient | $k_f$ | $\pm 10\%$ around SysID nominal | SimpleFlight |
| Position noise (std) | $\sigma_p$ | $\mathcal{U}(0.005, 0.02)$ m | SimpleFlight |
| Velocity noise (std) | $\sigma_v$ | $\mathcal{U}(0.01, 0.05)$ m/s | SimpleFlight |
| Wind force | – | **omitted** | SimpleFlight 도 omit (zero-shot Crazyflie 검증 충분) |

**Nominal**: Crazyflie 2.1 — $m_{\text{nom}} = 0.027$ kg, $I_{\text{nom}}$ standard. SysID 실측값으로 nominal 만 update, range 는 SimpleFlight 비율 그대로.

**근거**: 이 영역은 본 논문 contribution 영역 외부 — Chen 2025 가 zero-shot Crazyflie sim-to-real 성공한 검증된 bounds. 본 논문에서 더 보수적/공격적으로 갈 이유 없음.

### 4.5.2 Communication Layer (본 논문 main contribution axis)

| Parameter | Symbol | Distribution |
|---|---|---|
| Communication delay | $\tau$ | **Training**: $\mathcal{U}(50, 200)$ steps at 50 Hz (1.0–4.0 s) |
| | | **Eval**: $\tau \in \{10, 50, 100, 200, 500\}$ |

- $\tau = 10$: short-delay (기존 벤치마크 영역)
- $\tau \in \{50, 100, 200\}$: training distribution 내부 — main results
- $\tau = 500$: training 외삽 — out-of-distribution test (Cor 3.14 점근적 우월성)

Episode 시작 시 $\tau$ 한 번 sampling, episode 내 고정 (§3.3.2 Remark 3.2).

### 4.5.3 Mission & Arena

| Parameter | Distribution |
|---|---|
| $g_{\text{mission}}$ | $\mathcal{U}(\Omega_{\text{goal}})$, $\Omega_{\text{goal}} \subset \Omega$ |
| Attacker spawn $p^a_0$ | $\mathcal{U}(\Omega_{\text{spawn}, a})$ |
| Defender spawn $p^d_0$ | $\mathcal{U}(\Omega_{\text{spawn}, d})$, $\|p^a_0 - p^d_0\| > d_{\min}$ |
| Arena boundary $\Omega$ | $[-L, L]^3$, $L = 30$ m |
| Initial separation $d_{\min}$ | 5 m |
| Episode time limit $T_{\max}$ | 1000 steps (20 s at 50 Hz) |
| Collision radius $r_{\text{collide}}$ | 0.5 m |

### 4.5.4 DR vs Stage-Weighted Replay 분리 + §5 Ablation

§3.6.4 의 보완 관계:
- **DR**: episode 시작 시 환경 parameter sampling — within-iteration diversity
- **Stage-weighted replay** (§4.4.4): forecaster training buffer phase 가중 — across-iteration continuity

#### §5 DR Ablation (compute 효율)

전체 6 parameter sweep 은 explosion → 두 관점으로 압축:

| Ablation | 비교 | 측정하는 것 |
|---|---|---|
| **DR-on vs DR-off** | Full DR vs No DR | sim-to-real gap test, DR end-to-end 효과 |
| **Half-range DR** | Full DR vs Half-range (각 parameter ±50%) | DR scope sensitivity — 더 좁아도 충분한가 |

→ 2 ablation point, compute budget 합리적.

---

## 4.6 Sim-to-Real Considerations

### 4.6.1 SimpleFlight-Style SysID

Chen et al. (2025) 의 SysID 절차 (5대 요소 중 (4)) 채택:

1. Mass / inertia identification: 실기체 weighing + bifilar pendulum
2. Motor response: step input test 로 $\tau_m$ 측정
3. Thrust mapping: PWM-to-thrust curve fitting

→ Nominal value 를 §4.5.1 DR 분포 center 로 사용. DR range bracket 은 SimpleFlight 비율 그대로.

### 4.6.2 Low-Level Controller Delegation

Action space = body-rate CTBR. 실기체 시:
- Onboard FCU (PX4 / Betaflight) 가 CTBR-to-PWM 변환
- High-level policy (학습된 $\pi_\phi$) 만 onboard companion computer (Raspberry Pi 4 / NVIDIA Jetson) 에서 실행
- Forecaster + actor inference latency: < 10 ms 추정 (TBD measurement)

### 4.6.3 Communication Relay Simulation

실제 LTE/SATCOM 대신 controlled latency injection:
- Simulator 내부에서 defender state 를 ring buffer 에 저장
- $\tau$ step 후 retrieve → `s^d_{t-τ}` 로 attacker 에 제공
- Episode 시작 시 $\tau$ 한 번 sampling, 내내 고정 (§3.3.2 Remark 3.2)
- Packet loss 는 main 범위 외 (§3.3.3 Remark 3.3 — future work)

### 4.6.4 Hardware Deployment Scope

- **Main 실험**: simulation only (Genesis, §4.7)
- **Demonstrative experiment** (§5): Crazyflie 2.1 single-episode demonstration 가능 — quantitative hardware study 는 future work

---

## 4.7 Implementation Stack

### 4.7.1 Simulator — Genesis + 기존 Base Simulator 활용 (LOCKED)

**Choice**: **Genesis** [Genesis-Embodied-AI 2024] 위에서 사용자가 이미 구축·검증한 `genesis_quadrotor_simulator` 코드베이스를 base 로 활용.

#### 기존 base simulator 의 활용 가능 자산

사용자 코드베이스가 본 논문에 직접 활용 가능한 부분 (재구현 불필요):

| Asset | 위치 | 본 논문 활용 |
|---|---|---|
| Vectorized Genesis scene + drone entity | `envs/base_drone_env.py::_build_scene` | 그대로 — DR-aware multi-env scene 의 base |
| 9D rotation matrix obs (SimpleFlight Factor 1) | `_quat_to_rotmat_flat` + `_update_observation` | 본 논문 $s^a, \tilde{s}^d$ 의 18-dim 구성에 직접 사용 |
| Asymmetric actor-critic interface | `obs_groups`: actor=`policy`, critic=`policy + critic_obs` | 본 논문의 §3.5 asymmetric AC structure 와 정확히 호환 — `critic_obs` 에 oracle ground-truth $\tilde{s}^d_t$ 추가 |
| Action smoothness reward (SimpleFlight Factor 3) | `BaseDroneEnv::_reward_smooth` | 본 논문 §4.3.2 의 $r_{\text{smooth}}$ 그대로 사용 |
| `kf` domain randomization (SimpleFlight Factor 4) | `BaseDroneEnv::_apply_domain_rand`, `dr_kf_scale` | 본 논문 §4.5.1 DR table 의 thrust coefficient row 와 호환 |
| Drone parameter loader | `utils/drone_params.py::DroneParams` | Nova / CF2X 양쪽 즉시 사용 가능 |
| RL training pipeline | `scripts/train.py` (rsl-rl PPO) | 본 논문 §4.3 의 PPO loss 구조와 호환 — actor/critic 만 dim 확장 |
| Verified single-drone tasks | HoverEnv, TrajectoryEnv, AttitudeRatesEnv, RacingEnv | warm-up pretraining (§4.4.1) 의 immediate base — Trajectory/Hover 가 4-class trajectory mix 의 일부 |

#### 본 논문이 추가 구현하는 부분 (Implementation Contribution)

| Component | 위치 (제안) | §4 / §3 reference |
|---|---|---|
| `PursuitEvasionEnv` (1v1 attacker + defender) | `envs/pursuit_evasion/env.py` (신규) | §3.3 Goal-Delay POMDP |
| Communication delay relay | `envs/pursuit_evasion/comm_relay.py` (신규) — defender state 의 ring buffer + $\tau$-step delayed retrieval | §3.3.2 Def 3.1, §4.6.3 |
| Asymmetric obs/action structure | `_get_task_obs` 가 attacker/defender 별로 다른 정보 반환 | §3.3 Def 3.1, §4.1.2 |
| Forecaster $f_\psi$ training pipeline | `models/trajectory_forecaster.py` (신규) + `agents/forecaster_trainer.py` (신규) | §3.4 Def 3.10, §4.2.1, §4.3.1 |
| Stage-weighted replay buffer | `agents/stage_weighted_buffer.py` (신규) | §3.4 Remark 3.15, §4.4.4 |
| AMS-DRL bipartite alternating loop | `scripts/train_ams_drl.py` (신규) — Phase A · Phase B alternating + forecaster co-training | §3.6, §4.4.2 |
| Critic ground-truth oracle | `_get_critic_obs` 에 simulator 의 $\tilde{s}^d_t$ ground-truth 추가 | §3.5 Def 3.15, §4.2.3 |

#### Genesis / base simulator 채택의 강점

| 측면 | 평가 |
|---|---|
| Drone hovering 검증 완료 | ✓ — Nova (EMAX ECO 2306, 5.5", 4S) 로 PPO hovering 학습 검증됨 (`logs/{nova,cf2x}-hover_*`) |
| SimpleFlight 4 factor 구현 완료 | ✓ — Factor 1 (rot_mat), Factor 2 (critic time vector), Factor 3 (smooth reward), Factor 4 (kf DR) 모두 base class 에 구현 |
| GPU parallel simulation | ✓ — `num_envs=8192` default, AMS-DRL alternating wall-clock 단축 |
| 다중 drone 모델 지원 | ✓ — Nova / CF2X (sim-to-real 시 drone class 선택 가능) |
| Logging + reproducibility | ✓ — WandB / TensorBoard 양쪽 지원, Hydra-style cfgs.pkl save 기존 구현 |

#### 한계 및 mitigation

| 한계 | Mitigation |
|---|---|
| Genesis alpha-stage maturity (2024 말 출시) — API 변경 가능성 | 사용자 코드는 이미 작동 중이므로 commit hash pinning 으로 안정성 확보 (§4.7.4) |
| 사용자 코드는 single-drone task 만 검증 (PE 환경 미구현) | 본 논문이 `PursuitEvasionEnv` 를 `BaseDroneEnv` 의 subclass 로 추가 구현 — 기존 step/reset/reward layer 재활용 |
| Base simulator 의 default actor 는 `[128, 128]` 2-layer | 본 논문 setting 에서는 `[256, 256, 256]` 으로 capacity 확장 (§4.2.2 Note); §5 sensitivity 에서 비교 |

### 4.7.2 RL Framework

- **Backbone**: PyTorch 2.x
- **PPO implementation**: **`rsl-rl-lib >= 5.0.0`** (사용자 base simulator 와 동일) — `OnPolicyRunner` 의 `obs_groups` 기능으로 actor (`policy`) / critic (`policy + critic_obs`) 비대칭 입력 분리
- **Forecaster training**: 별도 supervised loop (Phase B 내부, base simulator 의 step/rollout 위에서 동작)
- **Logger**: WandB (`genesis-quadrotor` project) 또는 TensorBoard — 사용자 코드의 `--logger` flag 로 선택

### 4.7.3 Compute Budget

| Item | Estimate |
|---|---|
| GPU | 8x NVIDIA A100 (40GB) 또는 동급 |
| Defender warm-up | $\sim 12$ h (Genesis 속도로 단축 가능) |
| Bipartite alternating ($N_{\text{outer}} = 15$) | $\sim 3$–5 days (Genesis 속도 활용 시) |
| Total per seed | $\sim 4$–6 days |
| Seeds per result | 5–10 |

### 4.7.4 Reproducibility

- **Code**: GitHub release (review 후)
- **Trained weights**: HuggingFace Hub
- **Configuration**: Hydra (YAML)
- **Environment lock**: Genesis commit hash + Python `uv lock` 파일
- **Random seed**: episode-level + DR-level 분리 fixed

---

## 4.8 §3 Decisions → §4 Instantiation Mapping (Verification)

§3 의 모든 결정사항이 §4 에서 그대로 instantiate 됨을 검증.

| §3 결정 | §4 위치 | Status |
|---|---|---|
| §3.1.1 Contextual MDP, $g_{\text{mission}}$ as observable context | §4.1.4, §4.2.2 | ✓ |
| §3.1.2 State decomposition $(s^a, \tilde{s}^d)$ | §4.2.1–§4.2.3 | ✓ |
| §3.1.2 Rotation parameterization (rotation matrix) | §4.2.2, §4.2.3 (18-dim 구성), §4.2.0 anchoring | ✓ (SimpleFlight Factor 1 9D rot mat 정합; base simulator `_quat_to_rotmat_flat` 직접 활용) |
| §3.1.2 Kinematic projection $\phi$ | §4.2.1, §4.2.2 | ✓ (output 6-dim) |
| §3.4 Def 3.10 direct trajectory forecaster | §4.2.1 | ✓ ($L=8, H=4, d=256, \lambda_{\text{aux}}=0.5$) |
| §3.4 Prop 3.7 attacker 궤적 conditioning | §4.2.1 | ✓ |
| §3.4 Remark 3.15 stage-weighted replay | §4.4.4 | ✓ ($\alpha=0.7$ half-life 2 phase) |
| §3.4 Remark 3.16 차별점 4가지 | §4.2.1 | ✓ |
| §3.5 Def 3.15 actor signature 두 layer (K1) | §4.2.2 | ✓ ($k=0$ default) |
| §3.5 Def 3.15 critic privilege ($\tilde{s}^d_t$ full) | §4.2.3 | ✓ |
| §3.5 Prop 3.16 agent state aliasing 제거 (K3) | §4.3.2 | ✓ |
| §3.5 Prop 3.17 Ebi unbiasedness ($\pi^d$ param 비주입) | §4.2.3 | ✓ |
| §3.5 Prop 3.18 BPQL specialization (K2) | §4.3.2 | ✓ |
| §3.6.2 Defender warm-up | §4.4.1 | ✓ (4-class trajectory mix lock) |
| §3.6.2 Bipartite alternating | §4.4.2 | ✓ (tentative $N_{\text{outer}}, K_a, K_d$) |
| §3.6.2 Phase A: forecaster freeze (K4) | §4.4.2, §4.3.1 | ✓ |
| §3.6.2 Phase B: $Q^a_\theta$ freeze (K5) | §4.4.2 | ✓ |
| §3.6.3 Defender V-critic (K6), $g_{\text{mission}}$ 비노출 | §4.2.4 | ✓ |
| §3.6.4 DR scope | §4.5 | ✓ (SimpleFlight 차용 lock) |
| §3.6.5 Convergence (보류) | §4.4.3 | ✓ Fixed iteration budget |
| §3.6.5 Item 4 critic input 명세 phase 무관 보존 | §4.2.3 | ✓ |

#### v1 → v2 → v2-final TBD 진행

| 항목 | v1 | v2 | **v2-final** |
|---|---|---|---|
| Forecaster $L, H, d, d_{\text{ff}}$ | TBD | ✓ (DFBT anchor) | ✓ |
| Forecaster optimizer / dropout | TBD | ✓ (DFBT anchor) | ✓ |
| Action history $k$ | TBD | ✓ ($k=0$ default) | ✓ |
| $\lambda_{\text{aux}}$ | TBD | ✓ (0.5 default) | ✓ |
| **Trajectory mix** | TBD | TBD | **✓ (4-class lock)** |
| **DR bounds** | TBD | partial TBD | **✓ (SimpleFlight 차용 lock)** |
| **Stage-weighted replay $\alpha$** | TBD | tentative 0.7 | **✓ (0.7 lock + half-life 근거)** |
| **Simulator** | TBD | TBD (Isaac Lab vs Aerial Gym) | **✓ (Genesis lock)** |
| AMS-DRL hyperparams ($N_{\text{outer}}, K_a, K_d$) | TBD | tentative | **TBD (§5 monitoring)** |

**v2-final 잔여 TBD 1개**: AMS-DRL hyperparams ($N_{\text{outer}}=15, K_a=K_d=10^6$ tentative) — fixed iteration budget 으로 가므로 finalize 의 의미는 "최종 보고용 정확한 값" 일 뿐 학습 로직과 무관.

→ **§4 v2-final 은 사실상 finalized state.**

---

## 4.9 Connection to §5 Experiments

§4 의 implementation 이 §5 의 모든 ablation/measurement 의 base.

| §5 Experiment | §4 의존 | 변형 axis |
|---|---|---|
| Cor 3.14 검증 (direct vs recursive) | §4.2.1 forecaster | recursive baseline 도입 (transformer step-by-step variant) |
| $L_d$ sample-based estimation | §4.5.2 ($\tau$ DR) | rollout 에서 closed-loop kernel Lipschitz empirical 측정 |
| $\epsilon^{\text{dir}}(\tau)$ scaling | §4.2.1 + §4.5.2 | $\tau \in \{10, 50, 100, 200, 500\}$ |
| AMS-DRL Nash convergence | §4.4 | $N_{\text{outer}}$ scan + reward + Nash gap |
| Stage-weighted replay ablation | §4.4.4 | $\alpha \in \{0.5, 0.7, 0.9, 1.0\}$ |
| Critic privilege ablation | §4.2.3 | full $\tilde{s}^d$ vs projection $s^d$ vs no $s^d$ |
| Goal $\rho_g$ generalization | §4.5.3 | training $\rho_g$ vs eval $\rho_g$ 불일치 |
| Defender attitude observability | §4.2.1 | actor input 에 $\hat{R}^d, \hat{\omega}^d$ 추가 variant |
| Action history sweep | §4.2.2 | $k \in \{0, 1, 4, 10\}$ — $k = 1$ 은 SimpleFlight 표준 비교 |
| Forecaster size sweep | §4.2.1 | $L \in \{4, 8, 12\}$ |
| Actor architecture sweep | §4.2.2 | base simulator `[128, 128]` vs 본 논문 `[256, 256, 256]` |
| $\lambda_{\text{aux}}$ sweep | §4.3.1 | $\lambda_{\text{aux}} \in \{0.1, 0.5, 1.0\}$ |
| Trajectory mix sweep | §4.4.1 | 3-way: SimpleFlight-only / 본 default / reactive-heavy |
| DR ablation | §4.5.4 | DR-on/off + Half-range DR |
| Simulator stability test | §4.7.1 | Genesis vs OmniDrones fallback (필요 시) |
| Packet loss extension (선택) | §4.6.3 | Gilbert-Elliott 2-state Markov |

---

### 📎 §4 의 역할

§3 가 추상 framework 라면 §4 는 그것의 1:1 instantiation. 새 이론 0건. §3 의 K1–K12 정정사항이 모두 implementation level 에서 reflect 됨을 §4.8 verification table 로 확인.

### 💡 §4 → §5 의 핵심 흐름

1. §4 v2-final 의 implementation 으로 main system 학습 (single configuration)
2. §5 의 ablation 으로 §3 의 핵심 주장 검증:
   - **Cor 3.14** (direct beats recursive in long delay) — §4.2.1 + §4.5.2
   - **Prop 3.17** (critic privilege 효과) — §4.2.3 input variation
   - **§3.6** (AMS-DRL convergence in asymmetric goal-delay) — §4.4 Nash gap monitoring
3. v2-final 신규 sweep 으로 hyperparameter 선택 정당성 검증:
   - Forecaster size ($L$), action history ($k$), $\lambda_{\text{aux}}$, $\alpha$, trajectory mix, DR scope

### 🔬 v2-final → v3 핵심 변화 (코드 정합성 점검)

| 항목 | v2-final | v3 (이번 라운드) |
|---|---|---|
| Simulator 정의 | Genesis lock + fallback plan (OmniDrones) | **사용자 기존 `genesis_quadrotor_simulator` 활용** lock — Nova hovering 검증, SimpleFlight 4 factor 구현 완료 / fallback plan 제거 (이미 작동 중) |
| §4.7.1 구조 | "채택 근거 + 한계" 두 섹션 | **"기존 자산 + 추가 구현 + 강점 + 한계"** 4-section 으로 재구성 |
| §4.2.0 anchoring | 7 row | **9 row** — base simulator + state representation (9D rot) + action history (k=1 표준) 추가 |
| §4.2.2 Actor input | 18-dim "attacker full state" 만 표기 | **18-dim = pos(3)+vel(3)+rot_mat(9)+ang_vel(3)** 명시 (SimpleFlight Factor 1) |
| §4.2.2 $k = 0$ 근거 | "SimpleFlight 가 action history 미사용" (잘못된 framing) | **"SimpleFlight 표준 $k = 1$ 과의 의도적 deviation"** (정확한 framing) — forecaster redundancy 논거 |
| §4.2.2 Ablation | $k \in \{0, 4, 10\}$ | **$k \in \{0, 1, 4, 10\}$** — $k = 1$ 추가로 SimpleFlight 표준 직접 비교 |
| §4.2.2 Architecture | "SimpleFlight 와 동일 256/3-layer" 만 표기 | + **base simulator default `[128, 128]` 와의 차이 명시** + ablation 추가 |
| §4.2.3 Critic input | 18-dim "defender full" 만 표기 | **18-dim 구성 명시 + rotation parameterization rationale** ($§$3.1.2 + SimpleFlight Factor 1 정합) |
| §4.7.2 RL Framework | "stable-baselines3 또는 custom" | **`rsl-rl-lib >= 5.0.0` lock** (사용자 코드와 동일) + WandB / TensorBoard 명시 |
| §4.8 verification | rotation parameterization 항목 부재 | **rotation row 추가** — $§$3.1.2 → $§$4.2.2/§4.2.3 18-dim 정합 검증 |
| §4.9 §5 sweep | 15 entries | **17 entries** — actor architecture sweep + $k=1$ 추가 |

### 🎯 다음 작업

§4 v3 사실상 frozen. 다음 작업 후보:

1. **§5 Experiments protocol** — 가장 critical. 특히:
   - $L_d$ sample-based estimation 의 measurement procedure
   - Recursive baseline 구체 implementation (Cor 3.14 검증의 핵심)
   - Sánchez Roncero 2025 KTH 와의 direct comparison 설계
2. **Implementation milestone planning** — `PursuitEvasionEnv`, `comm_relay`, `trajectory_forecaster`, `stage_weighted_buffer`, `train_ams_drl` 의 incremental 구현 순서 설계
3. §2 Related Work prose 변환
4. §1 Introduction (§3-§5 안정화 후)
5. Appendix A–G (theorem proofs, latency table)

---

*📝 본 문서는 Section 4 의 v3 본문. v2-final 의 모든 lock 사항 보존 + 사용자 기존 `genesis_quadrotor_simulator` 와의 코드 정합성 점검 결과 4건 정정 (rotation parameterization, base simulator activation, action history $k$ framing, actor architecture). 다음 작업: §5 Experiments protocol, 또는 implementation milestone planning.*