# 3.1 Preliminaries — Contextual MDP & Drone Pursuit-Evasion Structure

---

## 3.1.1 Contextual MDP Formulation

본 논문의 MDP는 **Contextual MDP** [Hallak et al., 2015] 구조를 따른다. 에피소드 시작 시 context variable $g \in \mathcal{G}$ 가 사전분포 $\rho_g$ 에서 샘플링되어 **에피소드 내내 고정**되며, 이는 reward 함수와 policy를 parameterize한다. Contextual MDP를 다음 튜플로 정의:

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, R, \rho_0, \gamma, \mathcal{G}, \rho_g \rangle
$$

- $\mathcal{S}$ : state space
- $\mathcal{A}$: action space
- $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$ : transition kernel
- $R: \mathcal{S} \times \mathcal{A} \times \mathcal{G} \to \mathbb{R}$ : $g$**conditioned reward**
- $\rho_0$ : initial state distribution
- $\gamma \in (0, 1)$: discount factor
- $\mathcal{G}$ : context space, $\rho_g$: context prior

고정된 $g$ 하에서 각 에피소드는 sub-MDP $\mathcal{M}_g = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, R(\cdot, \cdot; g), \rho_0, \gamma \rangle$에서 진행된다. Policy는 $g$-conditioned로 정의: $\pi(a \mid s; g)$.

$g$-conditioned  정책 목적함수, 가치 함수, Q 함수:

$$
J(\pi; g) = \mathbb{E}*{s_0 \sim \rho_0, a_t \sim \pi(\cdot \mid s_t; g), s*{t+1} \sim \mathcal{P}(\cdot \mid s_t, a_t)} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t; g) \right]
$$

$$
V^\pi(s; g) = \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t; g) \Big| s_0 = s, g \right]
$$

$$
Q^\pi(s, a; g) = \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t; g) \Big| s_0 = s, a_0 = a, g \right]
$$

모든 Bellman equation과 optimality 논의는 $g$ **조건부로 이해**한다. POMDP 확장 시 (§3.3 이후) $g$는 완전 관측되어 관측 $o_t$ 에 포함되지만 state에는 포함되지 않는다.

### 📎 왜 Contextual MDP인가

Contextual MDP는 $g$ 를 **state에 넣지 않으면서도** reward와 policy가 $g$ 에 의존하는 것을 정식으로 허용한다. 이는 세 가지 실질적 이점을 제공한다:

1. **State는 물리량만**: $s$ 는 시간에 따라 물리적으로 진화하는 양만 담고, "임무 명세"와 분리된다.
2. **Transition 표기 경량화**: $g$가 state 밖에 있으므로 $\mathcal{P}$ 가 $g$ -independent — 매 스텝 $\delta(g_{t+1} - g_t)$를 끌고 다닐 필요 없음.
    - state 에 g 포함시 표기
        
        $\mathcal P(s_{t+1}^{\text{확장}}∣s_t^{\text{확장}},a_t)=\mathcal P^a(s_{t+1}^a∣⋅)⋅\mathcal P^d(\tilde {s}*{t+1}^d∣⋅)⋅δ(g*{t+1}−g_t)$
        
3. **GCRL/Contextual MDP 문헌 호환**: Goal-Conditioned RL [Schaul et al., 2015; Park et al., 2025]과 자연스럽게 연결됨.

## 3.1.2 1v1 Drone Pursuit-Evasion — State & Context

본 논문의 drone pursuit-evasion 세팅 (§2 참조) 은 위 Contextual MDP 구조의 인스턴스다.

### State Decomposition

Attacker drone과 defender drone의 full state는 자연스럽게 두 부분으로 분해된다:

$$
s_t^{\text{full}} = (s_t^a, \tilde{s}_t^d) \in \mathcal{S}^a \times \tilde{\mathcal{S}}^d
$$

- **Attacker (ego) full state**: $s_t^a = (p_t^a, v_t^a, R_t^a, \omega_t^a)$ — position, velocity, rotation matrix, angular velocity
- **Defender full state** (simulator-held): $\tilde{s}_t^d = (p_t^d, v_t^d, R_t^d, \omega_t^d)$ — 동일 구조

### Attacker-Observable Projection

Attacker의 통신 링크를 통해 전달되는 defender 정보는 **kinematic subset**에 한정된다. 이는 외부 레이더/지상 추적 센서가 제공하는 전형적인 관측 modality에 부합한다 (자세한 아키텍처 근거는 §3.3.1 참조). 다음 **projection map** $\phi$ 를 정의:

$$
s_t^d := \phi(\tilde{s}_t^d) = (p_t^d, v_t^d) \in \mathcal{S}^d
$$

이하 attacker의 POMDP 분석에서 "defender state"는 이 projection $s_t^d$를 지칭한다. Attitude-aware defender observation으로의 확장은 §5 ablation에서 다룬다.

### Mission Goal as Context

$$
g_{\text{mission}} \in \mathcal{G} = \mathbb{R}^3 : \text{attacker가 도달하려는 targeted navigation goal}
$$

- 에피소드 초기에 사전분포 $\rho_g$ 에서 샘플링되어 **에피소드 내 고정** (Contextual MDP context)
- **Attacker는** $g_{\text{mission}}$ **을 즉시, 정확히 관측함** (자신의 임무)
- Reward 함수 $R(s, a; g_{\text{mission}})$의 terminal bonus와 shaping 항이$g_{\text{mission}}$에 의존
- Attacker policy $\pi^a(a \mid s; g_{\text{mission}})$도 $g_{\text{mission}}$-conditioned

### Transition Factorization

Transition kernel은 attacker와 defender 부분으로 **조건부 독립 분해**된다:

$$
\mathcal{P}\big( (s_{t+1}^a, \tilde{s}*{t+1}^d) \big| (s_t^a, \tilde{s}t^d), a^a_t,a_t^d \big) = \underbrace{\mathcal{P}^a(s{t+1}^a \mid s_t^a, a^a_t)}*{\text{attacker 전이}} \cdot \underbrace{\mathcal{P}^d(\tilde{s}_{t+1}^d \mid \tilde{s}*t^d, a_t^d)}*{\text{defender 전이}}
$$

여기서 attacker 행동 $a^a_t$ 와 defender 행동 $a^d_t$ 는

$$
a^a_t\sim \pi^a(\cdot |s^a_t,s_t^d; g_{\text{mission}}),~~~~~~a^d_t\sim \pi^d(\cdot|\tilde s_t^d,s_t^a)
$$

각각의 policy $\pi^a$ 와 $\pi^d$ 에 의해서 결정된다.

### Episode Termination

에피소드는 다음 조건 중 하나가 성립하면 terminal state로 종료된다:

1. **Collision**: $|p_t^a - p_t^d| < r_{\text{collide}}$ — 두 드론 간 최소 안전거리 이탈
2. **Arena boundary 이탈**: $p_t^a \notin \Omega$ 또는 $p_t^d \notin \Omega$ — 정의된 arena $\Omega$ 밖으로 나감
3. **Time limit**: $t \geq T_{\max}$ — 에피소드 최대 길이 도달

§3.1.2의 transition factorization과 이후 §3.3–§3.5의 모든 명제는 **non-terminal transition**에 대해서만 주장된다. Terminal state는 reward의 discrete 항으로 처리되며 transition 분석에서 제외된다.

### Remark 3.1 (Defender Policy Regimes)

Defender policy $\pi^d$는 학습 단계에 따라 두 체제로 구분되며, 이는 §3.6 AMS-DRL 스케줄과 대응된다:

| Regime | Defender Policy | §3.6 위치 |
| --- | --- | --- |
| **Warm-start: Pretrained tracking policy** | RL 사전학습된 hovering/tracking 정책 $\pi^d_{\text{warm}}$ — 임의 좌표를 추적할 줄 아는 RL 정책 | §3.6.2 Defender Warm-up Pretraining (사전 단계) |
| **AMS-DRL: Reactive learning** | RL로 학습된 reactive defender $\pi^d_{\text{RL}}$ | §3.6.2 Bipartite Alternating (Phase A · B) |

**Warm-start 체제의 역할**: Attacker가 완전 random일 때 bipartite alternating을 시작하면 exploration collapse, non-stationarity, degenerate equilibrium 위험이 있다. Pretrained tracking defender는 최소한의 reactive behavior (attacker 접근 시 따라감) 를 제공하여 attacker의 기본 navigation과 evasion 학습을 가능하게 한다.

**AMS-DRL 체제가 본 논문의 평가 기준**이다 — Xiao and Feroskhan (2024) 의 AMS-DRL framework 에서 bipartite Nash convergence 가 보장되는 설정이며, §5 실험이 이 체제에서 수행된다.

### 📎 $g_{\text{mission}}$ as Context

1. **Context로서의 자격**: 에피소드 내 불변, reward·policy parameterize, state와 분리
2. **즉시 관측**: Attacker는 자신의 임무 $g_{\text{mission}}$ 을 delay 없이 정확히 앎 — POMDP 관측 함수에서 identity로 들어감
3. **Marginalization 금지**: $g_{\text{mission}}$ 이 관측된 순간 모든 conditional은 $g_{\text{mission}}$ 고정 조건부로 전개. $\rho_g$에 대한 적분은 "across-episodes generalization" 분석에서만 등장하며, single-episode value·belief·policy gradient 계산에서는 등장하지 않음.

**Goal-Conditioned RL (GCRL)과의 대조**:
$g_{\text{mission}}$ 의 역할은 GCRL 표준 [Schaul et al., 2015; Park et al., 2025] 과 동일 — 에피소드별 샘플되어 policy·reward의 conditioning input. 차이점은 우리 세팅이 **adversarial pursuit와 결합**됐다는 것으로, attacker policy $\pi^a(a \mid \cdot; g_{\text{mission}})$ 이 **goal 달성과 defender 회피를 동시에 최적화**해야 한다.

**MOMDP (Ong et al., 2010)와의 대조**:
$(s^a, \tilde{s}^d)$ factorization은 MOMDP의 mixed observability (ego 완전관측 + opponent 부분관측) 와 일치한다. 단 v8에서는 $g_{\text{mission}}$ **을 MOMDP의 hidden state가 아니라 Contextual MDP의 observable context로 취급**한다.

### 💡 본 섹션이 §3.2 이후에 주는 토대

- **§3.3 Goal-Delay POMDP**: 여기서 분리한 $s^a$(즉시 관측) vs $s^d$(지연 관측) 의 비대칭이 delay 공식화의 출발점
- **§3.4 Forecaster**: $s^d = (p, v)$ projection만 예측하면 됨 — attitude 제외되므로 output 차원 감소. Reactive defender belief의 attacker 궤적 의존성도 여기서 시작
- **§3.5 Asymmetric Critic**: Critic은 **full** $\tilde{s}^d = (p, v, R, \omega)$ 를 privileged signal로 받음 (richer-than-observation privilege). 이는 temporal privilege (delay 없음) 에 더해 dimensional privilege (attitude 접근) 도 포함하는 설계 — Ebi (2025) Informed AAC 이론으로 unbiasedness 확보 (§3.5.3)
- **§3.6 AMS-DRL**: Pretrained tracking defender (사전 단계) → Phase A · B alternating 으로 attacker 와 reactive defender 가 교대 학습. 각 Phase 내부에서 한 쪽 policy 를 freeze 하여 §3.1–§3.5 결과의 stationarity 가정이 만족됨

---

# 3.2 Background — Full-Observation-Delay POMDPs

---

## 3.2.1 Constant-Delay MDP (CDMDP)

기존 delay RL 연구의 표준 formulation은 **Constant-Delay MDP (CDMDP)** [Walsh et al., 2008] 에서 출발한다. 모든 관측이 동일하게 $\tau$step 지연된다는 가정을 Markov game에 적용하면 다음을 얻는다:

$$
\mathcal{M}_\tau^{\text{full}} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, R, \rho_0, \gamma, \mathcal{G}, \rho_g, \tau \rangle
$$

여기서 $\mathcal{S} = \mathcal{S}^a \times \tilde{\mathcal{S}}^d$  는 §3.1.2의 joint state space. 시점 $t$ 에서 attacker는 현재 joint state $s_t = (s_t^a, \tilde{s}t^d)$ *대신 **$\tau$ step 지연된 $s_{t-\tau}$*** 를 관측한다. (**핵심 가정은 attacker 자신의 상태** $s_t^a$ **도 함께 지연된다는 것** — 이는 §3.3 에서 완화된다.)

### Deterministic Transition 가정

본 논문의 이론 분석은 **deterministic transition 가정** 하에 전개된다:

$$
\mathcal{P}^a(s_{t+1}^a \mid s_t^a, a_t^a) = \delta\big(s_{t+1}^a - f^a(s_t^a, a_t^a)\big),  \\mathcal{P}^d(\tilde{s}_{t+1}^d \mid \tilde{s}*t^d, a_t^d) = \delta\big(\tilde{s}*{t+1}^d - f^d(\tilde{s}_t^d, a_t^d)\big)
$$

즉 attacker와 defender dynamics가 각각 결정론적 함수 $f^a, f^d$ 로 기술된다. 이 가정은 Katsikopoulos & Engelbrecht (2003) 이래 대부분의 delay RL 이론 연구 [Derman 2021; Wu et al. 2025; Chen 2021] 가 채택한 표준 방식이다. 센서 노이즈, motor 응답 변동, 외력 등 현실의 stochastic 교란은 이론 분석 범위 밖으로 두고, **§3.6 Domain Randomization** 에서 training 시 covariance shift로 분리 처리한다.

### Defender Closed-Loop Kernel

Attacker 관점에서 defender action $a^d_t$ 는 관측되지 않는 hidden variable이므로, 이후 belief 전개에서 defender action을 $\pi^d$ 로 marginalize한 **closed-loop kernel** 을 반복해 사용하게 된다. 표기 간결화를 위해 다음을 정의:

$$
\tilde{\mathcal{P}}^d(\tilde{s}^d_{t+1} \mid \tilde{s}^d_t, s^a_t;, \pi^d) := \int \pi^d(a^d_t \mid \tilde{s}^d_t, s^a_t) \cdot \delta\big(\tilde{s}^d_{t+1} - f^d(\tilde{s}^d_t, a^d_t)\big) , da^d_t
$$

$\tilde{\mathcal{P}}^d$ 는 "defender policy $\pi^d$ 를 흡수한 defender dynamics" — attacker 관점에서 defender 전체 시스템이 어떻게 움직이는가를 기술한다. $\tilde{\mathcal{P}}^d$ 는 defender 자기 상태와 attacker 상태 둘 다에 의존 — reactive defender 에 대응. §3.4 Assumption 3.11 이하의 Lipschitz 가정은 이 closed-loop kernel 에 대해 정의된다.

### Augmented State Construction

Markov property 복원을 위해 Katsikopoulos & Engelbrecht (2003) 는 다음 **augmented state** 를 도입했다:

$$
x_t^{\text{full}} := \big( s_{t-\tau}, a_{t-\tau}^a, a_{t-\tau+1}^a, \dots, a_{t-1}^a \big) \in \mathcal{S} \times (\mathcal{A}^a)^\tau
$$

$\tau$ step 전 관측과 그 이후 attacker 가 취한 $\tau$ 개 action을 묶은 것. Defender action $a^d_{t-\tau:t-1}$ 은 attacker가 관측하지 않으므로 augmented state에 포함되지 않는다.

Deterministic 가정 하에서, attacker의 현재 상태는 $x_t^{\text{full}}$ 로부터 **정확히 복원**된다:

$$
s_t^a = f^a\big( f^a\big( \cdots f^a(s_{t-\tau}^a, a^a_{t-\tau}) \cdots, a^a_{t-2} \big), a^a_{t-1} \big) =: \bar{s}^a_t
$$

즉 attacker 궤적 전체 $\bar{s}^a_{t-\tau:t}$ 는 $x_t^{\text{full}}$ 의 결정론적 함수이다. 반면 defender 상태 $\tilde{s}^d_t$ 는 관측되지 않은 defender action 이력 $a^d_{t-\tau:t-1}$ 에 의존하므로, $\pi^d$ **에 대한 marginalization이 여전히 필요**하다:

$$
b_\tau\big( \tilde{s}^d_t \mid x_t^{\text{full}}; \pi^d \big) = \int \prod_{i=0}^{\tau-1} \tilde{\mathcal{P}}^d\big( \tilde{s}^d_{t-\tau+i+1} ,\big|, \tilde{s}^d_{t-\tau+i},, \bar{s}^a_{t-\tau+i}; \pi^d \big) , d\tilde{s}^d_{t-\tau+1:t-1}
$$

여기서 $\bar{s}^a$ 는 $x_t^{\text{full}}$ 에서 복원된 deterministic attacker 궤적. 결국 belief는 **defender 상태에 대한 분포** 로 축소되며, 그 원인은 단 하나 — attacker가 defender action $a^d$를 관측하지 않는다는 **정보 비대칭**.

이 관찰은 CDMDP 가 단순히 "지연된 관측" 문제가 아니라 **Markov game 구조에서 defender action이 자동으로 hidden variable이 되는 문제** 임을 드러낸다. 기존 delay RL 문헌 중 이 이중 구조 — 지연 + hidden opponent action — 를 명시적으로 다룬 연구는 드물다.

## 3.2.2 Taxonomy of Existing Approaches

CDMDP를 해결하려는 기존 접근법은 belief를 어떻게 다루는지에 따라 네 계열로 분류된다.

### (A) Memoryless Augmented-State Methods

대표: Katsikopoulos & Engelbrecht (2003), Derman et al. (2021)

Augmented state $x_t^{\text{full}}$  를 policy network 입력에 그대로 넣고 표준 MDP RL (DQN, PPO 등) 을 적용. Belief를 **명시적으로 표현하지 않고** neural network가 implicit하게 처리하도록 위임.

장점: 구현 간단, 기존 RL 알고리즘 그대로 적용 가능.
단점: 차원 증가로 인한 효율성 저하 (§3.2.3).

### (B) Explicit Belief Approximation Methods

**대표**: D-SAC [Bouteiller et al., 2020], BPQL [Kim et al., 2023]

학습된 dynamics model $\hat{\mathcal{P}}$ 을 이용해 belief를 **recursive하게 근사**:

$$
\hat{b}*i = \hat{\mathcal{P}}(\cdot \mid \hat{b}*{i-1}, a^a_{t-\tau+i-1}), \quad i = 1, \dots, \tau
$$

Particle filter 또는 Gaussian parameterization으로 구현.  **BPQL은 augmented state Q-value를 원래 state Q-value 로 projecting하는 변형으로, belief 근사가 critic update 안에 통합된 형태** (상세는 §3.5 Prop 3.18).

장점: Belief가 명시적 → value estimation과 forecasting에서 활용 가능.
단점: Recursive 적용으로 근사 모델 오차가 누적 (§3.4 Theorem 3.12 에서 정량화).

### (C) Model-Based Delay-Aware Methods

**대표**: Delay-Aware MBRL [Chen et al., 2021]

학습된 transition model로 **rollout을 직접 시뮬레이션**하여 $\tau$ step 후 state를 예측, 이를 policy 입력으로 사용. (B) 의 변형이지만 planning loop을 명시적으로 사용.

장점: Planning과 학습 통합.
단점: (B) 와 동일한 recursive error 문제 + model rollout 비용.

### (D) Sequence-Model Based Methods

**대표**: Delay Transformer [Karamzade et al., 2024], DFBT [Wu et al. 2025], RNN-based delay handling

Augmented state 또는 delayed observation history 를 sequence로 보고 Transformer 또는 RNN 으로 encoding. Belief 를 latent representation 또는 forecasted state sequence 로 처리. **DFBT (Wu et al. 2025) 는 step-by-step recursive forecasting 대신 transformer 의 attention 으로 한 번에 belief 를 forecast 하여 compounding error 를 회피한다는 framing 을 명시적으로 제시한 대표 사례** — 본 논문 §3.4 의 direct forecaster 가 채택하는 framework 이다.

장점: Long-range dependency 처리에 강함.
단점: Sample efficiency 낮음, 학습 불안정.

### Fair-Comparison Disclaimer

위 네 계열의 방법은 모두 **symmetric delay (모든 관측이 동일하게 지연)** 를 가정한 세팅에서 제안되고 평가되었다. 본 논문의 goal-delay 세팅 (attacker 자신의 상태는 즉시 관측) 은 이들 방법의 설계 전제와 다르므로, 이들 방법의 "실패" 를 주장하는 대신 **symmetric-delay 전제 하의 이 방법들이 asymmetric-delay 세팅에서 어떻게 동작하는가** 를 §5 실험에서 경험적으로 관찰한다.

## 3.2.3 Fundamental Limitations

앞 절 네 계열은 belief 처리 방식이 다르지만, CDMDP 가정 자체에서 유도되는 **두 가지 근본 한계** 를 공유한다.

### Limitation 1: Effective Horizon Compression & Credit Assignment

Augmented state의 차원은 $\tau$ 에 대해 **선형적으로** 증가:

$$
\dim(x_t^{\text{full}}) = \dim(\mathcal{S}) + \tau \cdot \dim(\mathcal{A}^a)
$$

Continuous deep RL 세팅에서 이는 **exponential 상태공간 팽창** 이라기보다 다음 두 가지 실질적 문제를 낳는다:

1. **Effective horizon compression**: Policy가 $\tau$ step 이전의 $s_{t-\tau}$ 로부터 현재 decision을 유도해야 함. Discount factor $\gamma$ 하에서 $\tau$ 가 클수록 "현재 reward 에 기여하는 과거 정보"의 effective weight가 $\gamma^\tau$ 로 감소 — Policy가 참조해야 할 정보가 의사결정 관점에서 "흐릿해짐".
2. **Credit assignment 악화**: Action $a^a_t$ 의 효과가 최소 $\tau$ step 후에야 관측 가능 → gradient 신호가 희석. $\tau = 100$ 규모의 지연에서는 PPO/SAC의 advantage estimation이 매우 noisy 해짐.

이 두 효과가 결합되어, 기존 방법들이 $\tau \lesssim 10$  **규모의 단기 지연**에서는 잘 동작하지만 장기 지연으로 스케일업이 어려운 근본 이유가 된다. 본 논문의 대상인 통신 지연 규모 ($\tau \geq 100$) 는 기존 벤치마크의 주요 평가 영역 밖이다.

### Limitation 2: Recursive Model-Error Compounding

Deterministic dynamics 가정 하에서도 **learned dynamics model** $\hat{\mathcal{P}}^d$ 의 근사 오차는 불가피하다. 접근법 (B), (C) 계열은 이 모델을 recursive하게 적용:

$$
\hat{b}_\tau = \hat{\mathcal{P}}^d \circ \hat{\mathcal{P}}^d \circ \cdots \circ \hat{\mathcal{P}}^d;(b_0) \quad (\tau \text{ compositions})
$$

각 composition에서 발생하는 single-step model error $\epsilon^d := \sup_{s^d, a^d} \mathcal{W}(\hat{\mathcal{P}}^d, \mathcal{P}^d)$  가 누적된다. Dynamics가 non-contractive (Lipschitz 상수 $L_d \geq 1$ ) 인 경우 — drone과 같이 open-loop 불안정한 시스템이 대표적 — error는 step 수에 따라 **기하급수적으로 증가** 한다.

이 compounding의 **정량적 bound는 §3.4 Theorem 3.12 에서 Markov game 세팅으로 일반화된 형태로 제시** 한다. 본 절에서는 정성적 관찰에 그친다.

### 종합 — §3.3 으로 넘어가는 동기

Limitation 1은 attacker 자신의 상태까지 지연 관측한다는 **가정 자체의 비효율성** 을 드러내고, Limitation 2는 belief를 recursive하게 처리하는 **구현 전략의 비효율성** 을 드러낸다. §3.3은 전자를 **문제 정의 수준에서** 완화 (goal-delay POMDP 도입) 하고, §3.4는 후자를 **구현 전략 수준에서** 완화 — Wu et al. (2025) DFBT 의 direct forecasting framework 를 base 로 채택하되, 본 논문 setting 에 특수화한다 (§3.4 Remark 3.16 차별점 참조).

### 📎 §3.2 의 역할

"기존 방법의 한계가 왜 본질적인가" 를 설치하는 섹션. 다만 본 논문이 주장하는 것은 "기존 방법이 틀렸다" 가 아니라 **"기존 방법의 설계 전제 (symmetric delay) 가 우리 문제 (asymmetric delay) 에 과도한 정보 손실을 가정한다"** 이다. §5 실험에서도 이 관점을 유지한다.

---

# 3.3 Goal-Delay POMDP — Our Formulation

---

## 3.3.1 Motivating Observation — Asymmetric Communication Architecture

실제 1v1 드론 pursuit-evasion 세팅에서 **attacker와 defender의 통신·센싱 아키텍처는 근본적으로 비대칭적**이다. 이 비대칭이 우리 POMDP 공식화의 핵심 근거가 된다.

### 핵심 비대칭

| 측면 | Defender (방어 드론) | Attacker (공격 드론) |
| --- | --- | --- |
| 센싱 구조 | 지상 레이더 + edge compute + LOS RF cue 링크 | Onboard 센서 + 장거리 relay datalink |
| 관측 대상 | Attacker 상태 (실시간) | Defender 상태 (지연) |
| 일반적 latency | < 120 ms end-to-end | 수백 ms ~ 수 초 |
| 결과 | 사실상 latency-free ($\tau^d \approx 0$) | 유의미한 지연 ($\tau^a \geq 100$ steps at 50 Hz) |

**Defender 는 C-UAS "system of systems"의 기동 actuator** 로 동작한다 [Fortem SkyDome, Anduril Lattice, DroneShield DroneSentry 등 상용 시스템 참조]. 지상 기반 3D 레이더 (예: Robin IRIS) 가 360° 추적 돔을 유지하고, edge compute 지상국이 센서 융합을 수행한 뒤 LOS RF 로 defender drone 에 결정적 state vector를 전달. 반면 **attacker 는 적대 공역 깊숙이 고립된 SWaP-제약 노드** 로, onboard 센싱만으로는 defender 를 지속 추적 불가능하며 MAVLink 위의 원격 relay datalink 에 의존한다.

### 통신 지연 규모 (Latency Budget)

| Relay 방식 | 일반 latency |
| --- | --- |
| 4G/LTE cellular relay (RTT) | 97–150 ms |
| DroneKit MAVLink parsing buffer | +50 ms |
| 암호화 오버헤드 (AES) | +110 ms average |
| LEO SATCOM relay | 270–400 ms |
| GEO SATCOM relay | > 500–600 ms |
| SATCOM voice/data setup 베이스라인 | +700 ms |
| MAVLink HIGH_LATENCY2 (jammed/BVLOS) | 5000 ms (주기 = 5 s) |
| Link-16 TDMA (군용 메쉬) frame | 200 ms + multi-hop 누적 |

아키텍처의 구체적 공학적 세부 사항은 **Appendix A** 에서 다룬다. 본문의 핵심 메시지는: **attacker 의 defender 관측 지연 $\tau$ 가 수백 ms ~ 수 초 규모** — 50 Hz 제어 cycle 기준 $\tau \geq 100$ step 이 현실적으로 가능하며, 이는 기존 delay RL 벤치마크 (§3.2.2) 의 주요 평가 영역 ($\tau \lesssim 10$) 밖이다.

### Information-Asymmetric Framing

위 아키텍처적 비대칭은 게임 이론적으로 **information-asymmetric Markov game** 으로 자연스럽게 번역된다. 우리 POMDP 는 이를 다음과 같이 반영:

- **Attacker (ego agent)**: 자신의 상태 $s_t^a$ 와 mission goal $g_{\text{mission}}$ 은 즉시 관측하지만, defender 상태는 $\tau$-step 지연된 projection $s^d_{t-\tau} = \phi(\tilde{s}^d_{t-\tau})$ 로만 관측
- **Defender**: attacker 상태를 지상 레이더 + LOS RF 로 실시간 관측 — §3.6 AMS-DRL 에서 이 사실을 바탕으로 $\pi^d$가 학습됨

기존 pursuit-evasion 이론 [Isaacs 1965 등] 은 완전정보 게임을 가정하며, 우리 세팅은 이를 비대칭 POMDP 로 확장한 것이다.  본 세팅은 formally **2-player partially observable stochastic game**

[Shapley 1953; Littman 1994; Hansen et al. 2004] 의 인스턴스로 볼 수 있으며, 본 논문의 §3.3 이후 분석은 이 게임을 **고정된 defender policy** $\pi^d$ **하에서 attacker 관점의 single-agent POMDP 로 환원**

한 것에 대해 수행된다.

---

## 3.3.2 Definition 3.1 (Goal-Delay POMDP)

**Goal-Delay POMDP** 는 다음 튜플로 정의된다:

$$
\mathcal{M}_\tau^{\text{goal}} = \langle \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{P}, \mathcal{O}^{\text{obs}}, R, \rho_0, \gamma, \mathcal{G}, \rho_g, \tau \rangle
$$

**(고정된 defender policy** $\pi^d$ **하에서 정의;** $\pi^d$ **는 §3.6 AMS-DRL 로 학습되며 본 섹션에서는 phase-wise fixed 로 간주)**

구성 요소:

- $\mathcal{S} = \mathcal{S}^a \times \tilde{\mathcal{S}}^d$ : §3.1.2의 joint state space
- $\mathcal{A}$ : attacker action space (collective thrust + body rates)
- $\mathcal{O}$ : attacker 관측 공간 (아래 $\mathcal{O}^{\text{obs}}$ 참조)
- $\mathcal{P}$: §3.1.2의 factorized transition $\mathcal{P}^a \cdot \mathcal{P}^d$ — defender 부분은 §3.2의 closed-loop kernel $\tilde{\mathcal{P}}^d$ 로 attacker 관점 축약 가능
- $\mathcal{O}^{\text{obs}}: \mathcal{S} \times \mathcal{G} \to \mathcal{O}$ : 관측 함수 (아래 정의)
- $R: \mathcal{S} \times \mathcal{A} \times \mathcal{G} \to \mathbb{R}$ : $g$-conditioned reward (§3.1.1 Contextual MDP)
- $\rho_0$: initial state distribution
- $\gamma \in (0,1)$: discount factor
- $\mathcal{G} = \mathbb{R}^3$, $\rho_g$: mission goal context space 및 prior (§3.1 Contextual MDP)
- $\tau \in \mathbb{N}$: **비대칭 통신 지연 step 수** (attacker 의 defender 관측에만 적용)

**관측 함수**:

$$
o_t = \mathcal{O}^{\text{obs}}(s_t, g_{\text{mission}}) = \begin{pmatrix} s_t^a ,\ s^a_{t-\tau:t-1}, \ \phi(\tilde{s}^d_{t-\tau}), \ a^a_{t-\tau:t-1}, \ g_{\text{mission}} \end{pmatrix}
$$

즉:

- $s_t^a$: attacker 자신의 현재 full state (즉시 관측)
- $s^a_{t-\tau:t-1}$ : attacker 자신의 최근 $\tau$ step 상태 이력 — **모든 step 에서 자기 상태를 관측·저장하므로 buffer 에서 그대로 사용**(별도 추정 불필요)
- $\phi(\tilde{s}^d_{t-\tau}) = s^d_{t-\tau}$: defender state 의 **kinematic projection** (§3.1.2) 을 $\tau$-step 지연 관측 — 통신 링크가 전달하는 실제 정보량
- $a^a_{t-\tau:t-1}$ : attacker 자신의 최근 $\tau$ step action 이력 (자기 결정이므로 즉시 앎)
- $g_{\text{mission}}$ : mission goal (attacker 의 임무이므로 delay 없이 정확 관측 — Contextual MDP context)

**핵심 비대칭 (요약)**: Attacker 관측 대상 중 오직 **defender 의 kinematic state 만** 지연된다. Attacker 자신의 상태, action 이력, mission goal 은 모두 즉시 알려진다.

### Remark 3.2 (Stochastic Delay — Future Work)

Constant $\tau$ 를 per-step 샘플링 $\tau_t \sim \mathcal{D}\tau$ *로 확장 가능. 기본 후보:* $\tau_t \sim U(0, \tau{\max})$. **본 논문의 main 실험은 deterministic** $\tau$ **를 사용하며**, stochastic $\tau$ 확장은 future work 로 보류한다. 정당화: (i) 결정적 지연 하에서도 본 formulation 의 이론적 이점이 이미 충분히 복잡하다, (ii) stochastic $\tau$ 를 추가하면 sufficient statistic 이 $\tau_t$ 이력에도 의존하게 되어 asymptotic analysis 가 희석된다.

### Remark 3.3 (Packet Loss — Out of Scope)

Gilbert-Elliott 2-state Markov 모델이나 hold-last-sample policy 기반 packet loss 확장은 실제 무선 통신 환경에서 중요한 요인이다. 그러나 **본 논문은 시간 지연만을 main contribution 으로 다루며, packet loss 확장은 §5 ablation study 로 보류한다.** Packet loss 를 main thread 에 포함하면 (i) 이론 분석이 2-state Markov 에 종속되어 generality 가 축소되고, (ii) 실험 설계 복잡도가 증가하여 main contribution 의 명료성이 희석된다.

---

## 3.3.3 Core Theoretical Results

### Proposition 3.4 (Sufficient Statistic for Goal-Delay POMDP)

$\mathcal{M}_\tau^{\text{goal}}$ 에서, **고정된 defender policy** $\pi^d$ 하에 다음 **goal-delay information state** 가 optimal attacker policy 의 sufficient statistic 이다:

$$
\boxed{x_t^{\text{goal}} := \big(s_t^a, s^a_{{t-\tau:t-1}},s^d_{t-\tau}, a^a_{t-\tau:t-1}, g_{\text{mission}} ,\big)}
$$

여기서 $s^d_{t-\tau} = \phi(\tilde{s}^d_{t-\tau})$ 는 defender state 의 kinematic projection. 임의의 history-dependent optimal policy $\pi(a^a_t \mid h_t)$ *에 대해 (*$h_t$ *는* $g_{\text{mission}}$ *포함 전체 history),* $x_t^{\text{goal}}$ *에만 의존하는 정책* $\tilde{\pi}(a^a_t \mid x_t^{\text{goal}})$ 이 동일한 $g$-conditioned value 를 달성한다:

$$
V^{\pi}(h_t; g_{\text{mission}}) = V^{\tilde{\pi}}(x_t^{\text{goal}}; g_{\text{mission}})
$$

**v8 핵심 (재확인)**: $g_{\text{mission}}$ 은 Contextual MDP 의 observable context 이므로 sufficient statistic 에 **그대로 포함**되며, $\rho_g$ 로 marginalize 하지 않는다.

### Proof Sketch

Bellman equation 이 $x_t^{\text{goal}}$ 만으로 closed form 이 됨을 보인다. Deterministic transition 가정 (§3.2.1) 과 closed-loop kernel $\tilde{\mathcal{P}}^d$  (§3.2.1) 를 활용한다.

**Attacker 전이**: $s_{t+1}^a = f^a(s_t^a, a^a_t)$ — defender policy 나 $g_{\text{mission}}$ 에 비의존. $x_t^{\text{goal}}$ 에 $s_t^a, a^a_t$ 모두 포함됨. ✓

**Defender belief**: Attacker 는 defender 상태를 즉시 관측하지 못하므로 현재 defender 상태 $\tilde{s}^d_t$
에 대한 belief 가필요. Attacker 궤적 $s^a_{t-\tau:t-1}$ 은 **attacker 가 매 step 자기 상태를 관측·저장하므로 buffer 에서 직접 사용** 가능 — 별도 추정·복원 불필요. 따라서 belief 는 closed-loop kernel $\tilde{\mathcal{P}}^d$ 를

$\tau$ 번 composition 하는 형태로 $x_t^{\text{goal}}$  에 대해 닫힌 형태로 표현 가능 (구체 분해는 §3.4 Prop 3.7 참조). Closed-loop kernel 을 $τ$ 번 composition:

$$
b_\tau\big(\tilde{s}^d_t \mid x_t^{\text{goal}}; \pi^d\big) = \int \prod_{i=0}^{\tau-1} \tilde{\mathcal{P}}^d\big(\tilde{s}^d_{t-\tau+i+1} \big| \tilde{s}^d_{t-\tau+i}, s^a_{t-\tau+i}; \pi^d\big) d\tilde{s}^d_{t-\tau+1:t-1}
$$

초기 조건: $\tilde{s}^d_{t-\tau}$ 는 관측 $s^d_{t-\tau} = \phi(\tilde{s}^d_{t-\tau})$ 로부터 **kinematic 부분 (position, velocity)** 이 고정되고, attitude 부분 ($R^d, \omega^d$) 에 대한 prior 는 문제 세팅에 의존 (Appendix 참조). ✓

**Reward**: $R(s_t, a^a_t;, g_{\text{mission}}) = R(s_t^a, \tilde{s}^d_t, a^a_t;, g_{\text{mission}})$ . $s_t^a$ 는 $x_t^{\text{goal}}$ 에 직접 포함, $\tilde{s}^d_t$ 는 belief $b_\tau$ 로 표현, $g_{\text{mission}}$ 은 $x_t^{\text{goal}}$ 의 일부. $\rho_g$ **marginalization 없이** 모든 항이 $x_t^{\text{goal}}$ 조건부로 계산 가능. ✓

따라서 expected Bellman equation 이 $x_t^{\text{goal}}$ 만으로 closed form. $\square$

### Remark (Optimality Scope)

Prop 3.4 가 보장하는 "optimal policy" 는 **고정된 defender policy** $\pi^d$ 하에서의 optimum. §3.6 AMS-DRL 의 각 phase 내부에서 $\pi^d$ 가 stationary 로 간주되므로 위 정리가 각 phase 마다 적용된다. Phase 경계에서 $\pi^d$ 업데이트 시 attacker 도 re-optimize 해야 하며, alternating 과정의 수렴성은 §3.6.5 에서 Xiao & Feroskhan (2024) bipartite convergence 로 다룬다.

### Observation 3.5 (Information Advantage)

**Condition**: 두 POMDP $\mathcal{M}\tau^{\text{full}}$ *(§3.2) 과* $\mathcal{M}\tau^{\text{goal}}$ 이 **동일한 underlying Markov game** (동일 state space, transition, reward, defender policy $\pi^d$) 위에서 정의되고, **오직 관측 함수만 다르다**고 가정한다.

이 조건 하에서 동일 underlying state $s_t$ 에 대해:

$$
\mathbb{H}(\tilde{s}^d_t \mid x_t^{\text{full}}) \geq \mathbb{H}(\tilde{s}^d_t \mid x_t^{\text{goal}})
$$

즉 goal-delay information state 는 full-obs-delay 보다 **uncertainty 가 작거나 같다**. 따라서:

$$
V^{\mathcal{M}\tau^{\text{goal}}}(x_0;, g) \geq V^{\mathcal{M}\tau^{\text{full}}}(x_0;, g), \quad \forall x_0, g
$$

### Justification Sketch

(1) $x_t^{\text{full}} = (s_{t-\tau}, a^a_{t-\tau:t-1})$ 는 attacker 현재 상태 $s_t^a$ 를 **$\tau$-step forward 함수 composition** 으로만 유도 가능 — attacker 궤적이 $x_t^{\text{full}}$  에 explicit 하게 담겨있지 않음. (2) 반면 $x_t^{\text{goal}}$ 은 $s_t^a$ 를 직접 포함 → attacker-관련 uncertainty 항 제거. (3) 또한 $g_{\text{mission}}$ 이 $x_t^{\text{goal}}$ 에만 포함 → mission-relevant information 이 추가로 제공. (4) 동일 underlying Markov game 가정 하에 $V^*$ 비교가 well-defined (서로 다른 환경 간 비교가 아님). $\square$

### Remark (Information Advantage ≠ Algorithmic Advantage)

Observation 3.5 는 **정보 우위** 이지 **알고리즘 우위** 가 아니다:

- "Goal-delay 가 더 많은 정보를 가진다" ✓
- "따라서 optimal value 가 더 높다" ✓
- "따라서 우리 알고리즘이 자동으로 더 낫다" ❌

더 많은 정보를 **효율적으로 활용하는 알고리즘** 은 별도 설계가 필요 → §3.4 (direct trajectory forecaster), §3.5 (asymmetric critic) 으로 이어진다.

---

### 📎 §3.3 의 역할

"우리 문제 정의를 확립"하는 섹션. 기존 full-obs-delay POMDP 의 특수 케이스가 아니라 **통신 아키텍처의 비대칭에서 비롯된 구조적으로 다른 POMDP class** 임을 주장.

- **§3.3.1**: 현실적 motivation — 왜 비대칭이 필연적인가
- **§3.3.2 Definition 3.1**: 수학적 정식 정의 — tuple 외부 $\pi^d$ parametric, kinematic projection 관측
- **§3.3.3 Prop 3.4 + Obs 3.5**: 이론적 성질 — sufficient statistic 구체화 + information advantage 엄밀 명시

### 💡 §3.4 이후에 주는 토대

- **§3.4 Forecaster**: Prop 3.4 의 $x_t^{\text{goal}}$ 을 forecaster 입력으로 직접 사용. Attacker 궤적 $\bar{s}^a$ 가 명시적으로 $x_t^{\text{goal}}$ 에서 유도되므로, reactive defender 를 예측할 수 있음
- **§3.5 Asymmetric Critic**: Actor 는 $x_t^{\text{goal}}$ 유도된 관측 (forecasted $\hat{\tilde{s}}^d_t$) 사용, critic 은 privileged $\tilde{s}^d_t$ 직접 접근 — temporal + dimensional privilege
- **§3.6 AMS-DRL**: Def 3.1 의 tuple 외부 $\pi^d$ parametric 처리가 phase-wise fixed 가정과 자연스럽게 맞물림

---

# 3.4 Target Forecaster & Belief Representation

---

## 3.4.1 Goal-Delay Belief

§3.3 Proposition 3.4 의 sufficient statistic $x_t^{\text{goal}}$ 만으로 attacker 의 optimal policy 가 학습 가능하다는 것은 보였다. 그러나 실제 학습에서는 policy network 가 $x_t^{\text{goal}}$ 을 직접 입력받기보다, **현재 시점 defender state 의 추정치** 를 명시적으로 활용하는 것이 효과적이다 — 이는 (i) policy network 의 입력 차원을 줄이고, (ii) §3.5 의 asymmetric critic 설계와 자연스럽게 맞물린다.

### Definition 3.6 (Goal-Delay Belief)

Goal-Delay POMDP $\mathcal{M}_\tau^{\text{goal}}$ 에서 고정된 defender policy $\pi^d$ 하의 시점 $t$ **defender belief** 는 **현재 defender 의 kinematic projection $s^d_t = \phi(\tilde{s}^d_t) = (p^d_t, v^d_t)$ 에 대한 사후 분포** 로 정의한다:

$$
\boxed{b_\tau\big(s^d_t \big| x_t^{\text{goal}}; \pi^d\big) := \mathbb{P}\big( s^d_t \big| s_t^a, s^a_{t-\tau}, s^d_{t-\tau}, a^a_{t-\tau:t-1}, g_{\text{mission}}; \pi^d \big)}
$$

**중요**: Belief 는 attacker 의 통신 modality 와 일관되게 **kinematic projection $s^d$ 에 대해서만** 정의된다. Defender 의 attitude 부분 $(R^d, \omega^d)$ 은 attacker 의 의사결정에 사용되지 않으므로 belief 변수가 아니다. (§3.5 의 critic 은 이와 별개로 full $\tilde{s}^d$ 에 privileged access 를 가진다 — richer-than-observation privilege.)

### Proposition 3.7 (Belief Decomposition for Reactive Defender)

$§$ 3.2 의 closed-loop kernel $\tilde{\mathcal{P}}^d$ 와 $§$3.1.2 의 deterministic attacker dynamics $f^a$ 하에서, belief 는 다음 형태로 전개된다:

$$
b_\tau\big(s^d_t \big| x_t^{\text{goal}}; \pi^d\big) = \int \prod_{i=0}^{\tau-1} \tilde{\mathcal{P}}^d\big( \tilde{s}^d_{t-\tau+i+1} \big| \tilde{s}^d_{t-\tau+i}, s^a_{t-\tau+i}; \pi^d \big) \cdot p_{\text{init}}\big(\tilde{s}^d_{t-\tau} \mid s^d_{t-\tau}\big) d\tilde{s}^d_{t-\tau:t-1}
$$

여기서:

- ${s}^a_{t-\tau+i+1}$ — attacker 가 매 step 자기 상태를 관측·저장하므로 **buffer 에서 그대로 사용** ( $x_t^{\text{goal}}$ 에 포함된 $s^a_{t-\tau:t-1}$ 의 $i$ 번째 element). 별도 추정·복원 불필요.
- $p_{\text{init}}(\tilde{s}^d_{t-\tau} \mid s^d_{t-\tau})$ — 관측된 kinematic projection $s^d_{t-\tau}$ 에 대해 조건부로 정의되는 defender full state 의 initial distribution
- 마지막 단계에서 $\tilde{s}^d_t \to s^d_t$ projection: $s^d_t = \phi(\tilde{s}^d_t)$ 의 marginal 계산

즉 belief 는 **closed-loop kernel(defender)**

**을 $\tau$ 번 composition** 한 결과이며, 그 과정에서 **attacker 궤적 ${s}^a_{t-\tau:t-1}$ 가 매 step 의 conditioning** 으로 등장한다.

### Proof

Chain rule 과 Markov property 를 ($\tilde{s}^d, a^d$) 에 적용. Defender action 이력은 closed-loop kernel $\tilde{\mathcal{P}}^d$ 로 흡수되며, attacker 궤적은 deterministic 가정으로 forward propagation 으로 복원된다.

Initial distribution $p_{\text{init}}(\tilde{s}^d_{t-\tau} \mid s^d_{t-\tau})$ 는 다음과 같이 분해된다:(**관측된 정보 (kinematic) 는 delta (확신), 관측 안 된 정보 (attitude) 는 prior, 둘이 독립이라 곱으로 결합**)

$$
p_{\text{init}}(\tilde{s}^d_{t-\tau} \mid s^d_{t-\tau}) = \delta\big(p^d_{t-\tau} - p^d_{\text{obs}}\big) \cdot \delta\big(v^d_{t-\tau} - v^d_{\text{obs}}\big) \cdot \rho_{\text{att}}\big(R^d_{t-\tau}, \omega^d_{t-\tau}\big)
$$

즉 관측된 kinematic 부분 $(p^d_{\text{obs}}, v^d_{\text{obs}}) = s^d_{t-\tau}$ 는 delta 로 고정되고, 관측되지 않은 attitude 부분 $(R^d_{t-\tau}, \omega^d_{t-\tau})$ 만 시뮬레이터 prior $\rho_{\text{att}}$ 로 분포된다 (구체 분포는 Appendix 참조).

마지막 projection $s^d_t = \phi(\tilde{s}^d_t)$ 는 $\phi$ 의 결정성으로부터 직접 따른다 — full state belief 에서 attitude 부분을 marginalize 하여 kinematic 부분만 남긴 것. $\square$

### Remark 3.8 (Attacker 궤적 의존성 — 핵심 통찰)

Reactive defender 에 대한 belief 는 **attacker 궤적 전체** $\bar{s}^a_{t-\tau:t-1}$ 에 의존한다. Non-reactive (defender 가 attacker 와 무관하게 움직임) 케이스라면 belief 가 defender 자체 dynamics 만으로 전개되어 attacker 궤적이 불필요하지만, **reactive 케이스에서는 매 step 의 attacker 위치가 belief 의 필수 conditioning** 이다.

이 사실이 $§$3.2 의 full-obs-delay 방법과의 구조적 차이를 시사한다:

> **Full-obs-delay 방법 (§3.2) 은 이 구조를 직접 활용하도록 설계되지 않았다.** $x_t^{\text{full}}$ 에서는 attacker 자신도 $\tau$ step 지연 관측되므로, attacker 궤적을 복원하려면 학습된 dynamics model $\hat{\mathcal{P}}^a$ 의 forward propagation 이 필요하고, 이 과정에서 추가적인 model error 가 누적될 수 있다.
> 
> 
> 반면 **goal-delay 에서는 attacker 가 매 step 자기 상태를 즉시 관측·저장하므로**, attacker 궤적 $s^a_{t-\tau:t-1}$ 이 **buffer 에서 GT 값으로 그대로 사용** 된다 — 추정·복원 일절 불필요. Forecaster 는 이 깨끗한 attacker 궤적을 직접 conditioning 으로 받는다
> 
> 이 설계 차이가 asymmetric-delay 세팅에서 두 접근법의 성능 차이로 이어지는가는 $§5$ 에서 경험적으로 평가한다.
> 

---

## 3.4.2 Belief Approximation Strategies

Belief 의 정확한 계산은 Prop 3.7 의 적분이 일반적으로 intractable. 두 가지 근사 전략을 비교한다.

### Definition 3.9 (Recursive Belief Approximation — 기존 방식)

기존 방법 (D-SAC [Bouteiller 2020], BPQL [Kim 2023]) 이 채택하는 재귀적 근사. 학습된 closed-loop kernel approximation $\hat{\tilde{\mathcal{P}}}^d_\theta$ 를 도입하고, 이를 **$\tau$ 번 반복 적용**:

$$
\hat{b}*\tau^{\text{rec}}\big(\cdot \mid x_t^{\text{goal}}\big) = \big( \hat{\tilde{\mathcal{P}}}^d*\theta \circ \hat{\tilde{\mathcal{P}}}^d_\theta \circ \cdots \circ \hat{\tilde{\mathcal{P}}}^d_\theta \big) (s^d_{t-\tau}; {s}^a_{t-\tau:t-1})
$$

각 composition 에서 attacker 궤적의 해당 step 이 conditioning 으로 들어감.

**Single-step approximation error**:

$$
\epsilon^d := \sup_{\tilde{s}^d, s^a} \mathcal{W}_2\big( \hat{\tilde{\mathcal{P}}}^d\theta(\cdot \mid \tilde{s}^d, s^a), \tilde{\mathcal{P}}^d(\cdot \mid \tilde{s}^d, s^a; \pi^d) \big)
$$

$\mathcal{W}_2$ 는 Wasserstein-2 distance (Euclidean 기저 metric).

### Definition 3.10 (Direct Trajectory Forecaster)

학습된 parametric model $f_\psi$ 가 **한 번의 forward pass 로** 전체 trajectory 를 예측:

$$
\boxed{\hat{b}*\tau^{\text{dir}}\big(s^d*{t-\tau+1:t} \mid x_t^{\text{goal}}\big) = f_\psi\big( s^a_t, s^a_{t-\tau:t-1}, s^d_{t-\tau}, a^a_{t-\tau:t-1} \big)}
$$

Output 은 $\tau$-step 궤적 $(\hat{s}^d_{t-\tau+1}, \hat{s}^d_{t-\tau+2}, \ldots, \hat{s}^d_t)$ 의 sequence. **마지막 항 $\hat{s}^d_t$ 가 $§$ 3.5 actor 입력으로 사용된다** — intermediate trajectory 항들은 학습 시 auxiliary loss 로만 활용 (구체 구현은 §4 에서 다룸).

이 direct forecasting 접근은 **Wu et al. (2025) 의 DFBT (Directly Forecasting Belief Transformer) framework** 를 base 로 채택한다. DFBT 는 single-agent delayed RL 세팅에서 transformer attention 으로 belief 를 한 번에 forecast 하여 step-by-step recursive 의 compounding error 를 회피한다는 framing 을 제시한 대표 사례이며, 본 논문은 이 framework 를 채택하되 본 setting 의 특수성에 맞춰 확장한다 (구체 차별점은 §3.4.3 Remark 3.16).

**핵심 설계 결정 — forecaster 입력에 $g_{\text{mission}}$ 미포함**: Defender dynamics $\tilde{\mathcal{P}}^d$ 가 attacker 의 mission goal 에 직접 의존하지 않으므로, forecaster 도 $g_{\text{mission}}$ 없이 학습한다. 이는 의도적인 separation of concerns — forecaster 는 "kinematic 단서로부터 reactive defender 패턴 학습", policy 는 "$g_{\text{mission}}$ 으로부터 goal-directed action 결정".

**Approximation error (terminal marginal 기준)**:

$$
\epsilon^{\text{dir}}(\tau) := \sup_{x_t^{\text{goal}}} \mathcal{W}*2\big( \hat{b}*\tau^{\text{dir}}(s^d_t \mid x_t^{\text{goal}}), b_\tau(s^d_t \mid x_t^{\text{goal}}; \pi^d) \big)
$$

본 이론 분석은 **terminal belief $s^d_t$ 의 marginal** 정확도에 집중한다($s^d_t$ 가 defender 의 현재 위치이기 때문) — 이는 §3.5 actor 가 실제로 사용하는 quantity 이며, Theorem 3.12 의 recursive error bound $\mathcal{W}*2(\hat{b}^{\text{rec}}*\tau, b_\tau)$ 와 동일한 차원의 분포에 대한 비교를 가능하게 한다 (Cor 3.14 의 ratio 가 well-defined).

$\epsilon^{\text{dir}}(\tau)$ 의 $\tau$-dependence 는 학습된 **forecaster 의 capacity 와 학습 데이터 분포에 의존**하며, recursive 방식과의 비교는 §3.4.3 Theorem 3.13 / Corollary 3.14 에서 다룬다.

---

## 3.4.3 Compounding Error Analysis

### Assumption 3.11 (Lipschitz Continuity of Closed-Loop Defender Dynamics)

§3.6 AMS-DRL 의 각 phase 내에서 고정된 defender policy $\pi^d$ 의 closed-loop kernel $\tilde{\mathcal{P}}^d(\cdot \mid \tilde{s}^d, s^a; \pi^d)$ 가 다음 의미로 **$(L_d, L_a)$-Lipschitz** 라고 가정한다 (Wasserstein-2 distance, Euclidean 기저 norm $|\cdot|_2$):

$$
\mathcal{W}_2\big( \tilde{\mathcal{P}}^d(\cdot \mid \tilde{s}_1^d, s_1^a; \pi^d), \tilde{\mathcal{P}}^d(\cdot \mid \tilde{s}_2^d, s_2^a; \pi^d) \big) \leq L_d , |\tilde{s}_1^d - \tilde{s}_2^d|_2 + L_a , |s_1^a - s_2^a|_2
$$

**Lipschitz 상수 $L_d, L_a$ 의 성격**:

1. **Closed-loop 에 정의됨**: $\tilde{\mathcal{P}}^d$ 가 raw dynamics $f^d$ 와 defender policy $\pi^d$ 를 합친 결과이므로, $L_d, L_a$ 는 $\pi^d$ 의 reactivity 를 흡수한다. **§3.6 phase 경계에서 $\pi^d$ 가 업데이트되면 $L_d, L_a$ 도 재계산된다.**
2. **드론 setting 에서의 $L_d$ 추정**: 드론은 open-loop 불안정 시스템이며, reactive defender policy 가 공격적일수록 closed-loop $L_d$ 가 커지는 경향이 있다. 본 논문의 setting 에서 **실제 $L_d$ 값은 §5 에서 sample-based estimation 으로 경험적 측정** 한다.
3. **이론 분석의 일반성**: 이하 Theorem 3.12 / Corollary 3.14 는 $L_d, L_a$ 의 구체적 값에 대한 가정 없이 성립한다. $L_d > 1$ 의 non-contractive regime 에서 우월성이 두드러지며, $L_d \leq 1$ 의 contractive regime 에서는 recursive 방식도 유한 bound 를 가지나, 본 논문이 다루는 long-delay regime ($\tau \geq 100$) 에서는 작은 $L_d$ 라도 누적 오차가 의미 있는 수준에 도달할 수 있다.

### Theorem 3.12 (Compounding Error of Recursive Belief)

Assumption 3.11 과 single-step model error $\epsilon^d$ 하에서, recursive approximation 의 belief error 는 다음 bound 를 만족한다:

$$
\boxed{\mathcal{W}*2\big( \hat{b}*\tau^{\text{rec}}, b_\tau \big) \leq \frac{1 - L_d^\tau}{1 - L_d} \cdot \epsilon^d}
$$

(편의상 $L_d \neq 1$ 로 가정; $L_d = 1$ 인 경우 우변은 $\tau \cdot \epsilon^d$.)

**$L_d > 1$ regime 에서 error 가 $\tau$ 에 따라 기하급수적으로 증가**.

### Proof Sketch

Telescoping argument. $i$-th composition 에서의 error 를 $e_i := \mathcal{W}_2(\hat{b}^{\text{rec}}*i, b*\tau^{(i)})$ 로 두면:

$$
e_i \leq \epsilon^d + L_d \cdot e_{i-1}
$$

해석: 한 step composition 시 (i) 새로운 single-step error $\epsilon^d$ 가 추가되고, (ii) 직전까지의 누적 error 가 closed-loop kernel 의 Lipschitz factor $L_d$ 만큼 amplify 된다. Attacker 궤적 conditioning 은 deterministic 가정 하에 정확히 알려져 있으므로 $L_a$  term 은 기여하지 않음 (이게 goal-delay 의 이점).

$e_0 = 0$ 에서 시작해 $\tau$ 번 반복하면:

$$
e_\tau \leq \sum_{i=0}^{\tau-1} L_d^i \cdot \epsilon^d = \frac{1 - L_d^\tau}{1 - L_d} \cdot \epsilon^d \quad \square
$$

### Theorem 3.13 (Bounded Error of Direct Forecaster)

Direct trajectory forecaster $f_\psi$ 가 sufficient capacity 를 가지고 충분한 데이터로 학습되었다고 가정하면, belief error $\epsilon^{\text{dir}}(\tau)$ 는 다음 성질을 가진다:

$$
\boxed{\epsilon^{\text{dir}}(\tau) = \mathcal{O}\big(g(\tau)\big), \quad g(\tau) ;\text{growing strictly slower than}; L_d^\tau}
$$

즉 $\epsilon^{\text{dir}}(\tau)$ 는 $\tau$ 에 대해 **약하게 (subexponentially)** 의존하며, recursive 방식의 $L_d^\tau$ 의존성보다 점근적으로 우월하다.

### Proof Sketch

$\epsilon^{\text{dir}}(\tau)$ 는 forecaster 의 학습 generalization error 이며, 다음 두 성분으로 분해된다:

1. **Approximation error**: $f_\psi$ 의 model class 가 true belief mapping $x_t^{\text{goal}} \mapsto b_\tau$ 를 얼마나 잘 표현할 수 있는가. Universal approximation property — MLP 의 경우 [Cybenko 1989; Hornik 1991], attention-based sequence-to-sequence model 의 경우 [Yun et al. 2020] — 하에서 sufficient capacity 의 학습된 모델은 임의 정밀도로 근사 가능. $\tau$ 가 클수록 mapping 의 복잡도가 증가하므로 동일 capacity 에서 approximation error 는 $\tau$ 에 약하게 증가.
2. **Estimation error**: 유한 데이터 $N$ 에서 학습 시 발생하는 generalization error. PAC-style bound [Mohri et al. 2018] 하에서 $\mathcal{O}(\sqrt{1/N})$ 으로 수렴.

종합하면 $\epsilon^{\text{dir}}(\tau)$ 는 $\tau$ 에 대해 **다항식적 또는 $\log \tau$** 정도로 약하게 증가할 것으로 기대되며, **$L_d^\tau$ 의 지수적 증가보다 점근적으로 느리다**. 정확한 $\tau$-dependence 는 §5 에서 경험적 측정한다. $\square$

### Corollary 3.14 (Direct Beats Recursive in Long-Delay Regime)

Theorem 3.12 와 Theorem 3.13 으로부터, 두 방식의 belief error 비율은:

$$
\frac{\mathcal{W}*2(\hat{b}^{\text{rec}}*\tau, b_\tau)}{\mathcal{W}*2(\hat{b}^{\text{dir}}*\tau, b_\tau)} \approx \frac{(1 - L_d^\tau)/(1 - L_d) \cdot \epsilon^d}{\epsilon^{\text{dir}}(\tau)}
$$

$L_d > 1$ regime 에서 분자는 $\tau$ 에 대해 지수적으로 증가하고, 분모는 다항식적/로그적으로 증가하므로 ratio 는 **$\tau$ 가 커질수록 발산** 한다. 즉:

$$
\lim_{\tau \to \infty} \frac{\mathcal{W}*2(\hat{b}^{\text{rec}}*\tau, b_\tau)}{\mathcal{W}*2(\hat{b}^{\text{dir}}*\tau, b_\tau)} = \infty \quad (\text{when } L_d > 1)
$$

**핵심 메커니즘 (한 줄 요약)**:

> Direct forecaster 의 우월성은 단순히 "더 적은 error" 가 아니라, **closed-loop dynamics 의 Lipschitz amplification 을 구조적으로 회피** 한다는 데서 온다. Recursive 는 매 step 마다 Lipschitz 만큼 error 가 amplify 되지만, direct 는 한 번의 학습 task 의 generalization error 만 부담한다.
> 

**조건부 우월성 명시**: 상기 우월성은 forecaster 가 deployment 분포를 cover 하는 분포에서 학습되었을 때 성립한다 (Remark 3.15). 분포 mismatch 가 클 경우 두 방식 모두 성능 저하 가능. §5 에서 (i) $\tau$-scaling 비교 (Cor 3.14 의 점근적 우월성 검증), (ii) $L_d$ 의 sample-based estimation, (iii) $\epsilon^{\text{dir}}(\tau)$ 의 실측 $\tau$-dependence 를 모두 수행한다.

### Remark 3.16 (Relation to Wu et al. (2025) DFBT)

Wu et al. (2025) DFBT 는 single-agent delayed RL 세팅에서 direct vs recursive forecasting 의 framing 과 compounding error 의 점근적 우월성을 명시한 대표 사례이며, 본 논문 §3.4 의 forecasting framework 도 이를 base 로 채택한다. 단순한 인용 정정 차원을 넘어, 본 논문이 추가/확장하는 측면은 다음 네 가지로 정리된다:

1. **Single-agent → 2-player asymmetric Markov game**: DFBT 는 단일 환경의 transition kernel $\mathcal{P}$ 에 대해 belief 를 forecast 한다. 본 논문은 $(s^a, \tilde{s}^d)$ 의 Markov game 위에 정의된 **information-asymmetric POMDP** (§3.3 Def 3.1) 에서 belief 를 다루며, defender state 만이 belief 변수가 된다.
2. **Raw transition kernel → Closed-loop kernel Lipschitz**: DFBT 의 Lipschitz 가정은 raw dynamics kernel 에 대한 것이다. 본 논문은 defender policy $\pi^d$ 를 흡수한 **closed-loop kernel** $\tilde{\mathcal{P}}^d(\cdot; \pi^d)$ (§3.2.1) 에 대해 Lipschitz 를 정의한다 (Assumption 3.11). 이 차이로 인해 $L_d$ 가 $\pi^d$ 의 reactivity 를 반영하며, §3.6 phase 경계에서 $\pi^d$ 가 update 될 때마다 재정의된다.
3. **Reactive opponent belief 의 attacker 궤적 의존성 (Prop 3.7)**: DFBT 는 단일 환경 dynamics 에 대한 forecasting 이므로 "ego 궤적이 belief 의 conditioning 으로 들어가는 구조" 가 자명하지 않다. 본 논문은 reactive defender 가 attacker 위치에 반응하는 구조 자체에서 attacker 궤적 $s^a_{t-\tau:t-1}$ 이 forecaster 의 필수 입력임을 명시한다 (Remark 3.8). 이는 forecaster 입력 설계 ($x_t^{\text{goal}}$ 직접 사용) 와 직결되는 setting-특수적 통찰이다.
4. **Fixed environment → AMS-DRL non-stationary $\pi^d$ 와의 통합**: DFBT 는 fixed environment 가정 하에 학습된다. 본 논문은 §3.6 AMS-DRL 의 phase 별 $\pi^d$ 업데이트로 인한 distribution shift 를 명시적으로 다루며 (Remark 3.15), domain randomization 과 stage-weighted replay 라는 두 메커니즘으로 완화한다.

이상의 차별점은 본 논문의 main theoretical contribution 의 contribution boundary 를 명시한다. **Theorem 3.12 / 3.13 / Corollary 3.14 의 점근적 우월성 framing 자체는 Wu et al. (2025) 의 결과를 본 setting 에 specialize 한 것이며, 본 논문의 novelty 는 (i) goal-delay POMDP 정의 (§3.3 Def 3.1), (ii) reactive defender belief 분해 (Prop 3.7), (iii) closed-loop kernel Lipschitz 도입 (Assumption 3.11), (iv) AMS-DRL 와의 통합 (§3.6) 에 있다.**

### Remark 3.15 (Distribution Shift and Training Strategy)

Theorem 3.13 의 $\epsilon^{\text{dir}}(\tau)$ bound 는 forecaster $f_\psi$ 의 학습 분포와 deployment 분포가 일치한다는 가정 하에 유효하다. 본 논문 setting 에서 distribution shift 의 주요 원인은:

(i) **§3.6 AMS-DRL 의 phase 경계** — defender policy $\pi^d$ 가 update 되며 closed-loop kernel 분포가 변함. Forecaster 가 학습한 phase N 의 $\pi^d_N$ 분포와 phase N+1 의 $\pi^d_{N+1}$ 분포가 다르므로, naive 한 phase-별 학습은 forecaster 성능 저하.

(ii) **Attacker policy 진화** — attacker 가 더 공격적이 되면 defender 의 reactive 패턴도 변화, 새로운 trajectory 분포 등장.

(iii) **Sim-to-real transfer** — 실기체 배포 시 sensor noise, wind, motor response 차이.

본 논문은 두 가지 메커니즘으로 distribution shift 를 완화한다:

**메커니즘 1: Domain Randomization (§3.6 와 통합)**

Forecaster 는 §3.6 에서 정의된 domain randomization 분포 — randomized $\pi^d$ parameters, drone dynamics parameters, communication delay $\tau$ — 에서 학습된다. 이를 통해 deployment 분포의 광범위한 cover 를 확보하며, sim-to-real gap 도 동일 framework 로 처리된다.

**메커니즘 2: Stage-Weighted Replay Buffer**

AMS-DRL 각 phase 종료 시 forecaster 를 **누적 replay buffer 로 fine-tune** 한다. Buffer 는 모든 phase 의 rollout 을 보존하되, 샘플링 시 **최근 phase 에 더 큰 가중치** 를 부여한다 (예: phase-recency weighted sampling 또는 prioritized replay [Schaul 2016]). 이는:

- 최근 $\pi^d$ 분포에 forecaster 를 맞춤 (recency weighting)
- 이전 phase 분포의 catastrophic forgetting 완화 (buffer 보존)
- Single forecaster 유지로 구현 단순성 확보

구체적 weighting schedule 과 fine-tune frequency 는 §4 에서 다룬다.

**한계 및 검증**: 위 메커니즘으로도 cover 못 하는 distribution shift (예: deployment 시 학습 분포 밖의 새로운 defender 패턴, 또는 DR 분포가 cover 못 하는 sim-to-real shift) 는 본 논문의 한계다. **§5 ablation 에서 (i) DR 분포 밖 $\pi^d$ 에 대한 test-time evaluation, (ii) stage-weighted replay vs uniform replay 비교를 통해 본 전략의 유효 범위를 경험적으로 측정** 한다. Robust forecasting (uncertainty-aware prediction, hybrid direct+recursive ensemble 등) 은 future work 로 남긴다.

---

### 📎 §3.4 의 역할

§3.3 이 "$x_t^{\text{goal}}$ 가 sufficient" 를 보였다면, §3.4 는 **"$x_t^{\text{goal}}$ 으로부터 belief 를 어떻게 효율적으로 근사하는가"** 를 정량화. Wu et al. (2025) DFBT 의 direct forecasting framework 를 base 로 채택하되, 본 논문 setting 의 (i) 2-player Markov game 구조, (ii) closed-loop kernel Lipschitz, (iii) reactive defender belief 의존성, (iv) AMS-DRL 와의 통합 — 네 측면으로 specialize·확장한다 (Remark 3.16). 우월성 주장은 distribution match 조건 하에서이며, distribution shift 에 대한 방어는 §3.6 DR + stage-weighted replay 로 구성된다 (Remark 3.15).

### 💡 §3.5 이후에 주는 토대

- **§3.5 Asymmetric Critic**: Forecaster output $\hat{s}^d_t$ 가 actor 입력으로, full $\tilde{s}^d_t$ 가 critic 입력으로 — temporal + dimensional privilege
- **§3.6 AMS-DRL**: Phase 경계에서 $\pi^d$ 업데이트 → forecaster 도 stage-weighted replay 로 fine-tune → distribution shift 완화. 각 phase 에서 $L_d, L_a, \epsilon^d, \epsilon^{\text{dir}}$ 재계산. 단 Cor 3.14 의 메커니즘 (Lipschitz amplification 회피) 은 **phase 와 무관하게** 유효

---

# 3.5 Asymmetric Actor-Critic

---

## 3.5.1 Motivation

§3.4 까지 attacker forecaster $f_\psi$ 를 설계했다. 본 절에서는 이 forecaster 를 활용하는 policy $\pi_\phi$ 와 value function $Q_\theta$ 를 어떻게 학습할지 논한다.

### 문제 상황

Goal-Delay POMDP 에서 actor (policy) 는 forecast 된 defender state 만 볼 수 있다:

$$
\pi_\phi(a^a_t \mid x_t^{\text{goal}}), \quad \text{where}; \hat{s}^d_t ;\text{is obtained as the terminal output of}; f_\psi(x_t^{\text{goal}})
$$

$\hat{s}^d_t$ 는 §3.4 Theorem 3.13 의 학습 generalization error $\epsilon^{\text{dir}}(\tau)$ 만큼 실제 $s^d_t$ 와 차이가 있다. 이 오차가 그대로 **value estimation 에 반영되면** policy gradient 추정이 편향되어 학습이 불안정해진다.

### 핵심 통찰

**시뮬레이터 학습 시 ground-truth $\tilde{s}^d_t$ 는 알려져 있다.** 시뮬레이터가 defender 의 full physical state 를 매 step 유지하므로, 학습 framework 에서 이를 critic 에게 oracle 로 제공할 수 있다 — 단 actor 에는 주지 않는다. 배포 시에는 critic 자체가 사용되지 않으므로, 이 privilege 가 sim-to-real transfer 에 추가 부담을 주지 않는다.

이 비대칭이 **asymmetric actor-critic** 의 핵심 설계 — 학습 시간의 simulator privilege 와 배포 시간의 partial observability 를 분리.

---

## 3.5.2 Asymmetric Actor-Critic Design

### Definition 3.15 (Asymmetric Actor-Critic)

**Actor (배포용, partial observation 기반)**:

$$
\pi_\phi: \mathcal{X}^{\text{goal}} \to \Delta(\mathcal{A}^a)
$$

$$
\pi_\phi\big(a^a_t \big| x_t^{\text{goal}}\big)
$$

여기서 $x_t^{\text{goal}} = (s^a_t, s^a_{t-\tau:t-1}, s^d_{t-\tau}, a^a_{t-\tau:t-1}, g_{\text{mission}})$ 는 §3.3 의 sufficient statistic. 실제 구현에서는 $s^d_{t-\tau}$ 와 attacker 궤적 buffer 를 forecaster $f_\psi$ 에 통과시켜 terminal forecast $\hat{s}^d_t$ 를 얻은 후 actor network 에 입력한다.

**Actor signature 의 두 layer**: 엄밀히는 actor network 의 실제 구현 input 은 $(\hat{s}^d_t, s^a_t, a^a_{t-\tau:t-1}, g_{\text{mission}})$ 이며, 여기서 $\hat{s}^d_t = f_\psi(x_t^{\text{goal}})$ 의 terminal output 이다 (§3.4 Def 3.10). Sufficient statistic 수준에서는 $x_t^{\text{goal}}$ 만으로 충분 (Prop 3.4) 이지만, 차원 축소를 위해 forecaster 를 거친 표현을 입력으로 사용한다.

**Critic (학습용, ground-truth defender 포함)**:

$$
Q_\theta: \mathcal{S}^a \times \tilde{\mathcal{S}}^d \times \mathcal{A}^a \to \mathbb{R}
$$

$$
Q_\theta\big(s^a_t, \tilde{s}^d_t, a^a_t; g_{\text{mission}}\big)
$$

Critic 은 시뮬레이터의 privileged 정보를 직접 입력받는다:

- **현재 (non-delayed) defender full state** $\tilde{s}^d_t = (p^d, v^d, R^d, \omega^d)$ — actor 가 받지 못하는 attitude 차원도 포함
- 이는 actor 의 forecast $\hat{s}^d_t$ (projection 추정치) 와 비교할 때 **temporal privilege (delay 없음) + dimensional privilege (attitude 접근)** 의 양쪽을 포함한 **richer-than-observation privilege**

### Privilege 의 명확한 정의

| 측면 | Actor | Critic |
| --- | --- | --- |
| **Defender state 차원** | Projection $\hat{s}^d_t = (\hat{p}^d, \hat{v}^d)$ | Full $\tilde{s}^d_t = (p^d, v^d, R^d, \omega^d)$ |
| **Defender state 시점** | Forecast (학습된 $f_\psi$ 의 추정치) | Ground-truth (시뮬레이터 oracle) |
| **Mission goal** | $g_{\text{mission}}$ 직접 관측 | $g_{\text{mission}}$ 직접 관측 |
| **사용 시점** | 학습 + 배포 | 학습 only |

**중요**: Critic 은 **$\pi^d$ 자체나 defender 의 internal neural network parameter 는 받지 않는다** — physical state $\tilde{s}^d_t$ 만. 이는 §3.6 의 phase 경계 (다른 $\pi^d$ 로 update 시) 에서도 critic 의 입력 명세가 변하지 않음을 보장한다.

---

## 3.5.3 Theoretical Justification

본 설계의 정당성을 세 가지 이론적 framework — **Privileged POMDP** [Lambrechts 2025], **Informed AAC** [Ebi 2025], **BPQL belief projection** [Kim 2023] — 와 연결하여 establish 한다.

### Proposition 3.16 (Goal-Delay AAC as Privileged POMDP Instance)

Lambrechts et al. (2025) 의 **Privileged POMDP** framework 는 critic 이 actor 보다 더 많은 정보에 접근하는 POMDP class 를 정의한다. 핵심 이론 결과:

> Critic 이 ground-truth state 에 접근할 수 있을 때, value estimate 의 **agent state aliasing 으로 인한 error 항이 이론적으로 제거 가능** 하다 (Lambrechts et al. 2025 Appendix A 의 "Agent State Aliasing" 분석) → policy gradient 분산 감소 → 학습 안정화. 실용적으로는 학습된 critic 의 generalization error 만큼 잔여 aliasing 이 남으나, partial observation 만 사용하는 경우보다 본질적으로 작다.
> 

본 논문의 Goal-Delay POMDP 는 Privileged POMDP 의 자연스러운 인스턴스다:

| 요소 | Privileged POMDP 일반 | Our Goal-Delay Setting |
| --- | --- | --- |
| Actor 입력 | partial observation | $x_t^{\text{goal}}$ from forecaster |
| Critic 입력 | full state 또는 privileged signal | $(s^a_t, \tilde{s}^d_t)$ — ground-truth defender |
| Privilege 획득 | 시뮬레이터 oracle | 시뮬레이터 oracle (defender full state) |
| Aliasing 원인 | agent state aliasing (Lambrechts 2025) | $\tau$-step delay → terminal $s^d_t$ 부분정보 + forecast error $\epsilon^{\text{dir}}$ |

이 positioning 으로 Lambrechts et al. 의 aliasing 분석 결과를 본 setting 에 직접 import 하며, asymmetric critic 설계의 이론적 정당성을 확보한다.

### Proposition 3.17 (Unbiasedness via Informed AAC)

Ebi et al. (2025) 의 **Informed Asymmetric Actor-Critic** 은 critic 이 full state 를 받지 않더라도 **actor observation space 를 포함하는 임의의 privileged signal $i_t$** 만 받으면 policy gradient 가 unbiased 임을 증명한다:

$$
\nabla_\phi J(\pi_\phi) = \mathbb{E}\left[ \nabla_\phi \log \pi_\phi(a^a_t \mid o_t) \cdot Q(h_t, i_t, a^a_t) \right], \quad i_t \subseteq s_t
$$

**본 논문 specialization**:

- $o_t = x_t^{\text{goal}}$ — actor 가 사용하는 sufficient statistic
- $i_t = (s^a_t, \tilde{s}^d_t, g_{\text{mission}})$ — privileged signal (defender ground-truth + context)
- $i_t$ 는 시뮬레이터 oracle 이 제공하는 full physical state 의 subset → **Ebi's theorem 그대로 적용**

이는 본 논문의 critic 이 **defender policy $\pi^d$ 의 internal parameter 에 접근하지 않아도** unbiased policy gradient 를 유지함을 의미한다 — sim-to-real 시 defender 의 내부 표현 (e.g. neural network weights) 을 critic 에 injection 할 필요 없음.

### Proposition 3.18 (Goal-Delay as a Specialization of BPQL's Belief Projection)

Kim et al. (2023) BPQL 의 augmented state $\bar{s}_t$ 를 본 setting 의 sufficient statistic $x_t^{\text{goal}}$ (§3.3 Prop 3.4) 에 대응시키면, BPQL 의 belief projection 은 다음 형태로 simplify 된다.

BPQL 의 belief projection 은 augmented state Q-value 를 원래 state Q-value 로 project 한다:

$$
\bar{Q}^{\bar\pi}(\bar{s}*t, a_t) \approx \mathbb{E}*{s_t \sim \mathcal{P}(\cdot \mid \bar{s}*t)}\big[ Q*\beta^{\bar\pi}(s_t, a_t) \big]
$$

**Goal-delay setting** 에서 이 projection 은 자동으로 simplify 된다. §3.4 의 closed-loop kernel $\tilde{\mathcal{P}}^d$ 와 belief $b_\tau$ 를 사용하면:

$$
\bar{Q}^{\bar\pi}\big(x_t^{\text{goal}}, a^a_t\big) = \mathbb{E}*{\tilde{s}^d_t \sim b*\tau(\cdot \mid x_t^{\text{goal}}; \pi^d)}\big[ Q_\theta(s^a_t, \tilde{s}^d_t, a^a_t; g_{\text{mission}}) \big]
$$

즉:

- Attacker state $s^a_t$ 는 $x_t^{\text{goal}}$ 에 직접 포함되어 deterministic 하게 결정
- Defender state $\tilde{s}^d_t$ 만 belief $b_\tau$ 에 대해 expectation
- $g_{\text{mission}}$ 은 양변 conditioning 에 공통

### Proof Sketch

BPQL Eq. (11)-(12) 의 belief projection 을 goal-delay setting 에 적용하면, augmented state $x_t^{\text{goal}}$ 에 대한 belief $\mathcal{P}(s_t \mid x_t^{\text{goal}})$ 가 다음과 같이 분해된다:

$$
\mathcal{P}((s^a_t, \tilde{s}^d_t) \mid x_t^{\text{goal}}) = \delta(s^a_t - s^a_t) \cdot b_\tau(\tilde{s}^d_t \mid x_t^{\text{goal}}; \pi^d)
$$

첫 번째 component 는 deterministic 하게 결정 (attacker 자기 상태는 $x_t^{\text{goal}}$ 에 직접 포함), 두 번째 component 만 §3.4 closed-loop kernel 기반의 non-trivial belief. 따라서 BPQL projection 은 **defender state 에 대한 expectation** 으로 축소된다. $\square$

### Remark 3.19 (Relation to Prior Work)

| 선행연구 | 기여 | 본 논문 specialization |
| --- | --- | --- |
| **Lambrechts 2025** (Privileged POMDP) | Asymmetric AC 의 aliasing 분석 framework | Goal-delay 가 Privileged POMDP 인스턴스 (Prop 3.16) |
| **Ebi 2025** (Informed AAC) | Full state 없이도 privileged signal subset 으로 unbiased | Critic 이 defender ground-truth physical state 만 받고 unbiased 유지 (Prop 3.17) |
| **Kim 2023** (BPQL) | Uniform-delay 환경의 belief projection | Goal-delay 특수화 시 attacker state 가 deterministic 하게 분리 (Prop 3.18) |

**차별점**: 본 논문은 (1) goal-delay 문제 정의, (2) direct trajectory forecaster (Wu et al. 2025 DFBT framework 채택, §3.4 Remark 3.16 의 네 가지 측면으로 specialize), (3) communication architecture 자체에서 자연스럽게 유도되는 비대칭, (4) goal-conditioning 과 asymmetric privileged training 의 결합.

### Remark 3.20 (Reviewer Defense)

> *"Critic 에 privileged info 를 주는 건 cheating 아닌가?"*
> 

답변 4가지:

1. **학습 시에만 사용**: 배포 시 actor 만 사용하며 ground-truth 불필요. Critic 은 학습 끝나면 폐기됨.
2. **이론적 근거 완비**: Lambrechts 2025 (Privileged POMDP 의 aliasing 분석), Ebi 2025 (Informed AAC unbiasedness), Baisero 2022 (discrete POMDP unbiased AAC) 의 세 framework 가 각각 다른 측면에서 본 설계의 정당성을 보장.
3. **Privileged signal 의 자연성**: Critic 에 defender policy network 나 internal parameter 는 주지 않음 — **시뮬레이터가 자연스럽게 유지하는 physical state $\tilde{s}^d_t$** 만 oracle 로 제공. 이는 시뮬레이터의 inherent capability 이지 추가적 정보 주입이 아님.
4. **Sim-to-real 호환**: 시뮬레이터 학습에서 자연스럽게 얻을 수 있는 privileged information 이며, 현실 배포에 추가 요구사항 없음 (forecaster + actor 만 deploy).

---

### 📎 §3.5 의 역할

§3.4 가 forecaster 의 정확도를 정량화했다면, §3.5 는 **forecaster 를 활용한 policy 학습을 어떻게 안정화하는가** 를 다룬다. 핵심 메커니즘은:

- Actor: 배포에서 사용 가능한 정보 (forecast 된 projection) 만 사용
- Critic: 학습 시 시뮬레이터의 ground-truth 활용 → value estimation 안정화
- 두 framework 의 정합성: Privileged POMDP (Lambrechts 2025) + Informed AAC (Ebi 2025) + BPQL (Kim 2023) 으로 이론적 정당성 확보

### 💡 §3.6 이후에 주는 토대

- **§3.6 AMS-DRL**: 각 phase 에서 $\pi^d$ 가 update 되어도 critic 입력 명세 ($s^a_t, \tilde{s}^d_t, g_{\text{mission}}$) 는 변하지 않음 — Prop 3.17 의 unbiasedness 가 phase 경계에서도 유지
- **Forecaster 와의 통합**: §3.4 forecaster output $\hat{s}^d_t$ → actor 입력. Stage-weighted replay ($§$ 3.4 Remark 3.15) 가 forecaster 와 actor 양쪽에 적용

---

# 3.6 Adversarial Multi-Stage Co-Training of the Defender

본 섹션은 §3.1~§3.5 에서 고정된 defender policy $\pi^d$ 로 가정한 부분을 **AMS-DRL framework** [Xiao & Feroskhan 2024] 를 채택하여 학습하는 절차를 정의한다. 이는 attacker POMDP 의 transition kernel 이 의미 있는 challenge 를 제공하면서도 §3.1~§3.5 의 phase-wise stationary 가정과 정합적이도록 만들기 위한 필수 장치다.

---

## 3.6.1 Motivation — Why AMS-DRL?

§3.1.2 Remark 3.1 에서 defender 학습 체제를 (i) **warm-start: pretrained tracking policy** 와 (ii) **AMS-DRL: reactive learning** 으로 구분한 바 있다. Warm-start defender 만으로 학습된 attacker policy 는 단순 회피 정책으로 overfit 되기 쉬워, 실제 reactive learning-based defender 가 배포될 경우 일반화에 실패한다. AMS-DRL 채택의 근거는 세 가지:

1. **Overfitting 방지**: 학습된 defender $\pi^d_{\text{RL}}$ 은 attacker 의 취약점을 능동적으로 탐색 → attacker 를 더 다양한 challenge 에 노출
2. **Adversarial robustness**: Attacker 와 defender 가 교대로 강해지는 과정에서 양쪽 모두 robust 한 정책으로 수렴 (Xiao & Feroskhan 2024 의 AMS-DRL framework)
3. **Phase-wise stationarity 와의 정합**: 각 phase 내에서 한 쪽 policy 를 freeze 함으로써 §3.1~§3.5 의 가정 (transition kernel 의 stationarity) 을 자연스럽게 만족

---

## 3.6.2 Bipartite Training Schedule

### Defender Warm-up Pretraining

Bipartite alternating 시작 전에 defender 는 **RL 로 사전학습된 hovering / trajectory tracking 정책 $\pi^d_{\text{warm}}$** 으로 initialize 된다. 이 정책의 학습 절차:

- **Task**: 임의로 생성된 3D 좌표 trajectory 를 추적하며 hovering
- **Input**: 자기 full state $\tilde{s}^d_t$ + tracking 대상 좌표 $(p^a_{\text{ref}}, v^a_{\text{ref}})$ — 실제 attacker 자리에 임의 trajectory 를 대입
- **Action**: 4-rotor RPM (본 논문 main setting 과 동일)
- **Reward**: 기준 좌표에 대한 tracking error minimize
- **Output**: $\pi^d_{\text{warm}}$ — 주어진 좌표를 추적할 줄 아는 RL 정책

이 사전학습은 attacker 와 무관하며, 학습된 정책은 reactive 한 trajectory tracking 능력을 미리 확보한다. 구체 hyperparameter 와 절차는 §4 implementation 에서 다룬다.

### Bipartite Alternating

Initialize: $\pi^d \leftarrow \pi^d_{\text{warm}}$, $\pi^a \leftarrow \text{random}$, $f_\psi \leftarrow \text{random}$, $Q^a_\theta, V^d_\theta \leftarrow \text{random}$.

각 outer iteration 은 두 phase 로 구성:

**Phase A — Attacker Turn (Defender warm-start 활용)**:

- $\pi^d$ freeze, $f_\psi$ **freeze**
- $\pi^a, Q^a_\theta$ update via §3.5 Asymmetric AC
- 목적: 현재 defender (초기에는 $\pi^d_{\text{warm}}$, 이후 update 된 $\pi^d$) 에 대한 best response 학습
- Forecaster $f_\psi$ 가 Phase A 동안 freeze 되어 있어 actor 입력 ($\hat{s}^d_t = f_\psi(x_t^{\text{goal}})$ 의 terminal output) 의 분포가 stationary → attacker AC 학습이 안정

**Phase B — Defender Turn (Forecaster Co-Training)**:

- $\pi^a$ freeze
- $Q^a_\theta$ (attacker critic) 도 Phase B 동안 freeze — 구체 절차는 §4 implementation 에서 다룬다
- $\pi^d, V^d_\theta$ update via standard PPO (defender 측 학습 구조는 §3.6.3 참조)
- $f_\psi$ **update via stage-weighted replay** (§3.4 Remark 3.15)
- 목적: 현재 attacker 에 대한 best response defender 학습 + forecaster 를 새로운 $\pi^d$ 의 closed-loop kernel 분포에 적응

### Algorithm (Pseudocode)

```
# Pretraining (사전 단계, §4 에서 디테일)
π^d_warm ← train_hovering_tracker()

# Initialize for AMS-DRL
π^d ← π^d_warm
π^a, f_ψ, Q^a_θ, V^d_θ ← random init

# Bipartite Alternating
for outer in range(N_outer):
    # Phase A: Attacker turn
    freeze π^d, f_ψ
    for k in range(K_a):
        rollout episodes with π^a vs π^d (frozen)
        update (π^a, Q^a_θ) via §3.5 asymmetric AC

    # Phase B: Defender turn (with forecaster co-training)
    freeze π^a, Q^a_θ
    for k in range(K_d):
        rollout episodes with π^a (frozen) vs π^d
        update (π^d, V^d_θ) via standard PPO
        update f_ψ via stage-weighted replay (§3.4 Remark 3.15)

    if convergence_criterion(π^a, π^d):
        break
```

수렴 판정 기준 (e.g. reward plateau, empirical Nash gap, fixed iteration budget) 의 구체 정의는 §4 implementation 에서 다룬다.

---

## 3.6.3 Defender Side — Why Standard (Not Asymmetric) AC

Defender 측 학습은 §3.5 의 asymmetric actor-critic **이 아닌 standard symmetric AC (PPO)** 를 사용한다. 이는 본 논문의 정보 비대칭 구조에서 자연스럽게 따라온다.

### Privilege Gap 의 부재

§3.5 의 asymmetric AC 가 의미 있으려면 actor 와 critic 사이의 **privilege gap** 이 필요하다 — actor 가 partial / noisy / forecasted 정보를 사용할 때, critic 이 ground-truth privileged 정보로 가치 추정의 분산을 줄이는 것이 핵심. Attacker 측에서는:

- Actor: forecasted projection $\hat{s}^d_t$ 사용 (지연 + 차원 축소)
- Critic: ground-truth full $\tilde{s}^d_t$ 사용 (지연 없음 + 차원 풀)

→ **Temporal + dimensional privilege gap** 존재 → asymmetric AC 의 motivation 성립.

반면 defender 측에서는:

- §3.3.1 의 motivating observation 에 따라 defender 는 **ground radar + edge compute 로 attacker 를 실시간 관측** ($\tau^d \approx 0$)
- Defender 가 받는 attacker 정보는 kinematic projection $s^a_t = (p^a_t, v^a_t)$ 로, attacker 측 forecaster 의 출력과 **동일한 차원**
- 즉 actor 든 critic 이든 **같은 정보** ($\tilde{s}^d_t,, s^a_t$) 사용 가능

→ **Privilege gap 부재** → asymmetric critic 의 추가 정보가 없음 → standard symmetric AC.

### Defender 학습 구조

|  | Defender Actor | Defender Critic (V) |
| --- | --- | --- |
| **Input** | $(\tilde{s}^d_t, s^a_t)$ | $(\tilde{s}^d_t, s^a_t)$ |
| **Output** | $\pi^d(a^d \mid \tilde{s}^d_t, s^a_t)$ | $V^d_\theta(\tilde{s}^d_t, s^a_t)$ |
| **Privilege** | — | — (none beyond actor) |

**Mission goal $g_{\text{mission}}$ 비노출**: Defender 의 reward 와 policy 모두 $g_{\text{mission}}$ 에 비의존. Defender 는 단순히 attacker 추격·격추가 목표이며, attacker 의 임무 좌표는 모른다. 이는 defender 의 학습 task 를 단순화하고, 본 논문의 attacker-centric framing 과 정합적이다.

---

## 3.6.4 Domain Randomization Scope (개요)

모든 phase 에서 **Domain Randomization (DR)** 이 적용된다. 카테고리는 (구체 분포는 §4):

- **Attacker side**: 질량·관성 (SysID 기반, Chen et al. 2025), motor response 시정수, sensor noise
- **Defender side**: 동일 카테고리 (질량·관성, motor response, sensor noise)
- **Communication layer**: $\tau \in [\tau_{\min}, \tau_{\max}]$ — main contribution axis
- **Mission & arena**: $g_{\text{mission}}$ 초기화 분포 $\rho_g$, arena boundary, spawn positions

**DR 과 Stage-Weighted Replay 의 관계**: 보완적 메커니즘.

- **DR**: 매 episode 시작 시 환경 parameter 다양화 → forecaster/actor 가 다양한 분포 경험 (within-iteration diversity)
- **Stage-weighted replay** (§3.4 Remark 3.15): Buffer 에서 학습 sample 추출 시 최근 phase 가중 → catastrophic forgetting 완화 (across-iteration continuity)

---

## 3.6.5 Consistency with §3.1–§3.5

§3.6 의 schedule 은 §3.1~§3.5 의 분석 결과와 다음 방식으로 정합:

1. **Phase-wise stationarity**: 각 phase 내에서 한 쪽 policy 가 freeze 되므로 attacker POMDP 의 transition kernel $\tilde{\mathcal{P}}^d(\cdot; \pi^d)$ 가 stationary. Proposition 3.4–3.7, Theorems 3.12–3.13, Propositions 3.16–3.18 모두 각 phase 내부에서 그대로 성립.
2. **Phase-boundary non-stationarity**: Phase A → Phase B 전환 시 transition kernel 이 변한다. Attacker 와 forecaster 모두 re-optimization 이 필요하며, 본 논문은 이 alternating 과정의 수렴을 **Xiao & Feroskhan (2024) 의 AMS-DRL framework 채택** 으로 보장한다. 단 Xiao 2024 의 이론적 convergence 결과는 symmetric / fully observable Markov game 에 대한 것이며, **본 논문의 asymmetric goal-delay setting 에서의 convergence 는 §5 에서 empirical 검증** 한다.
3. **Forecaster co-adaptation**: $\pi^d$ 가 update 되면 closed-loop kernel $\tilde{\mathcal{P}}^d(\cdot; \pi^d)$ 의 분포도 변한다. Forecaster $f_\psi$ 는 **Phase B (Defender turn) 에서만 update** 되어, Phase A (Attacker turn) 에서 freeze 됨으로써 actor 입력 분포의 stationarity 를 확보한다. Stage-weighted replay (§3.4 Remark 3.15) 가 이전 phase 의 분포 일부를 유지하여 catastrophic forgetting 을 완화.
4. **Critic privilege 무결성**: Ebi (2025) Informed AAC unbiasedness (§3.5 Prop 3.17) 는 privileged signal $i_t$ 가 full state 의 subset 이기만 하면 성립. **$\pi^d$ 가 phase 마다 변해도 attacker critic 의 입력 명세 $(s^a_t, \tilde{s}^d_t, g_{\text{mission}})$ 는 보존** 되므로 unbiasedness 가 모든 phase 에서 유지된다. 단 $\tilde{s}^d_t$ 의 분포 shift 는 critic generalization 에 영향을 줄 수 있으며 (§3.4 Remark 3.15), stage-weighted replay 와 DR 이 이를 완화한다.

---

### 📎 §3.6 의 역할

"Defender 가 어떻게 얻어지는가" 를 framework level 에서 확립. §3.1~§3.5 의 formulation 이 **phase-wise stationary attacker POMDP** 로 well-defined 임을 보증하고, Xiao 2024 AMS-DRL framework 와의 관계를 명시. DR 카테고리, defender warm-up 의 구체 절차, 학습 hyperparameter 등은 §4 Method 에서 다룬다.

### 💡 §3.6 의 핵심 결정 (요약)

- **Defender warm-up**: RL 사전학습된 hovering/tracking 정책 $\pi^d_{\text{warm}}$
- **Phase 순서**: Attacker turn 먼저 (defender warm-start 활용) → Defender turn (forecaster co-training)
- **Forecaster update**: Phase B (Defender turn) 에서만, Phase A 에서 freeze
- **Defender side**: Privilege gap 부재 → **standard symmetric AC (PPO)**, V-critic 표기, $g_{\text{mission}}$ 비노출
- **Convergence**: Xiao 2024 framework 채택 + 본 setting 에서 empirical 검증 (§5)

---

# Section 3 — Problem Formulation

---

## 📌 3절 전체 스토리라인 (v8)

```
3.1 1v1 드론 pursuit-evasion 상태는 (attacker, defender)으로 자연 분해됨
    + mission goal g_mission은 attacker가 즉시 관측하는 conditioning variable
    + defender policy는 §3.6에서 AMS-DRL로 학습됨 (이 섹션에선 phase-wise fixed 가정)
         ↓
3.2 기존 delay RL: 모든 걸 지연시켜 상태 팽창 + 재귀 오차 누적
         ↓
3.3 우리: defender 정보만 지연 (goal-delay, asymmetric comm architecture)
    → information advantage (attacker knows own state + mission goal instantly)
         ↓
3.4 그 이점을 살리려면 forecaster를 direct trajectory prediction으로
    (Wu et al. 2025 DFBT framework 채택, 본 setting 에 specialize — Remark 3.16)
    + reactive defender 때문에 attacker ego 궤적이 forecaster 입력에 필수
         ↓
3.5 학습 안정화 위해 asymmetric critic (Privileged POMDP / Informed AAC 특수화)
    — training 시 defender full state privileged, deployment 시 forecasted state만
         ↓
3.6 Defender policy π^d는 고정 스크립트가 아닌 AMS-DRL로 공동 학습
    — §3.1~§3.5의 attacker POMDP가 phase-wise stationary transition kernel을 유지
```

---

# 📊 Section 3 정리 (v8)

## 각 결과 요약 표

| 결과 | Statement (한 줄) | Proof 방법 | 용도 | 한계 / 가정 |
| --- | --- | --- | --- | --- |
| **Prop 3.4** | $x_t^{\text{goal}} = (s_t^a,, s^a_{t-\tau:t-1},, s^d_{t-\tau},, a^a_{t-\tau:t-1},, g_{\text{mission}})$ 가 sufficient statistic | Bellman eq 가 $x_t^{\text{goal}}$ 만으로 closed form ($g_{\text{mission}}$ 조건부, $\rho_g$ marginalization 없음) | Architecture 정당화 (actor / critic / forecaster 입력 설계의 이론적 토대) | 고정 $\pi^d$, attacker 궤적은 buffer 에서 직접 사용 |
| **Obs 3.5** | Goal-delay 가 full-obs-delay 보다 informative — 동일 underlying Markov game 위에서 관측 함수만 다를 때 $V^*_{\text{goal}} \geq V^*_{\text{full}}$ | Entropy 비교 + Markov game 동일성 가정 | Information advantage 주장 (algorithmic advantage 와 구분) | 동일 underlying Markov game, 관측 함수만 다름 |
| **Prop 3.7** | Reactive defender belief 는 attacker 궤적 $s^a_{t-\tau:t-1}$ 에 의존, $g_{\text{mission}}$ 에 비의존; closed-loop kernel $\tilde{\mathcal{P}}^d$ 의 $\tau$-step composition | Chain rule + $\mathcal{P}^d$ 가 $g_{\text{mission}}$ 에 비의존 | Forecaster 입력 설계 근거 (attacker 궤적 buffer + $g_{\text{mission}}$ 미포함) | Closed-loop kernel 정의, attacker 궤적 buffer 가용 |
| **Thm 3.12** | Recursive error $\leq \frac{1-L_d^\tau}{1-L_d} \cdot \epsilon^d$ — $L_d > 1$ 시 $\tau$ 에 지수 증가 | Telescoping with closed-loop Lipschitz | 기존 recursive 방법의 점근적 약점 증명 | Closed-loop Lipschitz $(L_d, L_a)$, terminal marginal 기준 |
| **Thm 3.13** | Direct forecaster error $\epsilon^{\text{dir}}(\tau) = \mathcal{O}(g(\tau))$ — $L_d^\tau$ 보다 strictly 약하게 성장 (subexponential) | Universal approximation [Cybenko/Hornik/Yun] + sample complexity [Mohri] | 우리 방법의 점근적 우월성 증명 | Sufficient capacity, 충분한 학습 데이터, distribution match |
| **Cor 3.14** | $L_d > 1$ 인 long-delay regime ($\tau \to \infty$) 에서 direct/recursive ratio 발산 | Thm 3.12 + 3.13 | Direct forecasting 우월성 claim (Lipschitz amplification 회피) | Distribution match 조건 (Remark 3.15 참조). **Wu et al. 2025 DFBT 의 framing 을 본 setting 에 specialize (Remark 3.16)** |
| **Remark 3.15** | Distribution shift 완화: DR (within-iteration) + Stage-weighted replay (across-iteration) | 메커니즘 정의 + §5 ablation 검증 | Sim-to-real 및 phase 경계 robustness | 학습 분포 cover 못 하는 shift 는 한계 |
| **Remark 3.16** | Wu et al. (2025) DFBT 와의 차별점 4가지: (1) single-agent → 2-player Markov game, (2) raw → closed-loop kernel Lipschitz, (3) reactive opponent belief 의 attacker 궤적 의존성, (4) fixed env → AMS-DRL non-stationary $\pi^d$ 통합 | DFBT framework 채택 + 본 setting 특수화 명시 | Contribution boundary 명확화 — Wu 2025 의 framing 을 base 로, 본 논문 novelty 를 (i)~(iv) 로 재배치 | DFBT 의 single-agent / fixed env 가정과 본 setting 의 차이 명시 |
| **Prop 3.16** | Goal-delay AAC = Privileged POMDP [Lambrechts 2025] 의 인스턴스 — agent state aliasing 으로 인한 error 항이 이론적으로 제거 가능 | Framework matching | Asymmetric critic 의 이론적 토대 (main) | Lambrechts 2025 framework |
| **Prop 3.17** | Critic 이 ground-truth $\tilde{s}^d_t$ 만 받아도 policy gradient unbiased | Ebi (2025) Informed AAC theorem 의 specialization | Critic privilege 의 unbiasedness 보장 (defender policy parameter 비주입) | Informed AAC framework, $i_t \subseteq$ full state |
| **Prop 3.18** | Goal-delay setting 에서 BPQL belief projection 이 attacker state delta + defender belief 로 분해 (Kim 2023 augmented state $\bar{s}_t$ ↔ 본 setting $x_t^{\text{goal}}$ 대응) | BPQL projection 전개 + closed-loop kernel | Prior work (Kim 2023) positioning | BPQL framework |
| **§3.6 Bipartite Schedule** | Defender warm-up (RL pretrained) → Attacker turn ↔ Defender turn alternating (Phase A · B) | Xiao & Feroskhan 2024 AMS-DRL framework 채택 + 본 setting 에서 empirical convergence (§5) | Defender 획득 절차 정의 | Phase-wise stationarity, asymmetric setting 의 convergence 는 empirical |
| **§3.6.3 Defender Side** | Defender 측은 privilege gap 부재 → standard symmetric AC (PPO, V-critic), $g_{\text{mission}}$ 비노출 | Defender 의 attacker 관측이 실시간 → actor/critic 동일 정보 | 본 논문 비대칭성이 attacker side only 임을 명시 | Defender 가 attacker projection $s^a_t$ 를 실시간 관측 |
| **§3.6.5 Consistency** | §3.1–§3.5 의 결과가 각 phase 내부에서 그대로 성립 | Phase-wise $\pi^d$ (또는 $\pi^a$) fixed → transition kernel stationary | §3.1–§3.5 와 §3.6 의 논리적 정합성 | Phase 경계에서 re-optimization 필요, forecaster 는 Phase B 만 update |

---

## 🔧 보류 / 향후 확인

1. **$L_d$ 의 실제 값 — §5 sample-based estimation 결과**
    - Assumption 3.11 에서 $L_d \geq 1$ 단정 제거, §5 에서 측정 예정
    - 만약 $L_d < 1$ 으로 나오면 Cor 3.14 의 우월성 주장이 약화됨 → 결과 보고 framing 조정 필요
2. **$\epsilon^{\text{dir}}(\tau)$ 의 실제 $\tau$-dependence**
    - Thm 3.13 에서 "subexponential" 만 주장, 구체 값 (e.g. $\mathcal{O}(\sqrt{\tau})$, $\mathcal{O}(\log \tau)$) 은 §5 에서 측정
3. **Stochastic attacker/defender dynamics 확장**
    - 현재 deterministic transition 가정 (§3.2.1)
    - §3.6 Domain Randomization 으로 sim-to-real noise 처리, 이론은 deterministic 유지
    - Future work: stochastic transition 으로 이론 일반화
4. **Policy gradient bias under forecaster**
    - Ebi (2025) theorem 은 critic privilege 의 unbiasedness 만 보장
    - Forecaster 오차 $\epsilon^{\text{dir}}$ 이 actor 입력단에 있으므로 별도 분석 필요
    - Section 4 또는 Appendix 에서 다룸
5. **Distribution shift cover 검증** (Remark 3.15)
    - DR 분포 밖 $\pi^d$ 에 대한 test-time evaluation
    - Stage-weighted replay vs uniform replay 비교
    - §5 ablation 에서 다룸
6. **$\rho_g$ 선택이 성능에 미치는 영향**
    - 현재 $\rho_g$ = 특정 구조 (origin 근방 또는 uniform over bounded region)
    - Training 과 eval 의 $\rho_g$ 불일치 시 generalization 분석
    - §5 ablation 후보
7. **Bipartite convergence 의 empirical 검증**
    - Xiao 2024 의 이론적 보장은 symmetric setting → 본 논문 asymmetric goal-delay 에서는 §5 empirical 만
    - Reward plateau / Nash gap / fixed iteration budget 중 어느 것을 termination 으로 쓸지 §4 에서 결정
8. **Packet loss 확장 (보류)**
    - Gilbert-Elliott 2-state Markov 모델 적용 가능
    - Future work / §5 ablation 후보 — 시간 지연 실험 성공 후 추가 검증
9. **Low-level controller gain (보류)**
    - 현재 RPM action space 가정, low-level dynamics 는 simulator-provided
    - 실기체 배포 시 PWM-to-force/torque SysID 필요
    - SimpleFlight (Chen et al. 2025) procedure 적용, ablation candidate
10. **Defender 의 attitude 활용 ablation**
    - Attacker 가 defender attitude 를 관측하면 어떻게 되는지 (현재는 projection only)
    - §5 ablation candidate — projection vs full state observation
11. **Asymmetric AC 의 quadrotor empirical 선례 보강** (v8 신규)
    - 기존 Bauersfeld & Scaramuzza 2026 narrow pipe 71% 인용은 부정확 (71% 는 observation-level flow measurement 효과지 privileged critic 효과 아님) — v8 에서 제거
    - 진짜 quadrotor + asymmetric AC + privileged critic 사례 발굴이 향후 라운드 과제
    - 후보: Geles, Bauersfeld, Romero, Xing, Scaramuzza (RSS 2024) "Demonstrating Agile Flight from Pixels without State Estimation" — privileged critic 으로 quadrotor agile flight 성공 사례

---

## 🎯 Section 3 확정 후 To-Do

### Section 4 (Method) 구성

- [ ]  Forecaster $f_\psi$ architecture (Transformer-based seq2seq)
- [ ]  Actor / Critic network 구조 (§3.5 입출력 명세)
- [ ]  PPO loss 와 §3.5 asymmetric AC 의 통합
- [ ]  Defender warm-up pretraining 구체 절차 (3D trajectory tracking)
- [ ]  Stage-weighted replay buffer 의 weighting schedule
- [ ]  Domain Randomization 분포 (구체 hyperparameter)
- [ ]  SimpleFlight-style SysID 절차 (Chen et al. 2025)
- [ ]  Convergence criterion 정의 (reward plateau / Nash gap / fixed budget)
- [ ]  Phase B 에서 attacker critic $Q^a_\theta$ freeze 의 구체 implementation

### Section 5 (Experiments) 구성

- [ ]  **Cor 3.14 검증**: $\tau \in {10, 50, 100, 200, 500}$ 에 대한 prediction error vs $\tau$ — direct vs recursive 비교
- [ ]  **$L_d$ sample-based estimation**: Phase 별 closed-loop kernel 의 Lipschitz 측정
- [ ]  **$\epsilon^{\text{dir}}(\tau)$ 의 실측 $\tau$-dependence**: subexponential 주장 검증
- [ ]  **AMS-DRL Nash convergence plot**: Phase iteration 별 reward + policy stability
- [ ]  **Distribution shift ablation**: Stage-weighted replay vs uniform vs no-replay
- [ ]  **Critic privilege ablation**: Full $\tilde{s}^d$ vs projection only $s^d$
- [ ]  **Goal $\rho_g$ generalization**: Training $\rho_g$ vs eval $\rho_g$ 불일치 실험
- [ ]  **Defender attitude observability ablation**: Projection vs full state for attacker
- [ ]  **Packet loss extension**: Gilbert-Elliott 모델 (시간 지연 성공 후)

### Appendix 구성

- [ ]  **Appendix A**: Proposition 3.4 full proof (Bellman closure 의 엄밀한 전개)
- [ ]  **Appendix B**: Theorem 3.12 full proof + Wu et al. (2025) DFBT 와의 bound 비교
- [ ]  **Appendix C**: Theorem 3.13 의 sample complexity 분석 (universal approximation + PAC bound 결합)
- [ ]  **Appendix D**: Prop 3.16–3.18 full derivation (Privileged POMDP / Informed AAC / BPQL 인용 세부)
- [ ]  **Appendix E**: §3.6 AMS-DRL framework transfer argument (Xiao 2024 의 symmetric 결과를 asymmetric goal-delay 로 확장 시 가정 변화)
- [ ]  **Appendix F**: §3.3.1 통신 아키텍처 상세 (latency table 의 구체 출처, C-UAS 시스템 spec)
- [ ]  **Appendix G**: Defender warm-up pretraining 의 ablation (다양한 trajectory sampling 분포)

---

## 📦 Section 3 v8 최종 파일 목록

| 파일 | 내용 | 핵심 결정 (v7 → v8 변경) |
| --- | --- | --- |
| `section3_1_v8.md` | Contextual MDP + 상태 분해 + Markov game | K8: Stage 1/2/3 표기 제거 → "사전 단계 / Phase A · B alternating" 으로 통일 |
| `section3_2_v8.md` | CDMDP + Taxonomy + 한계 | K9: DFBT 인용 [Liotet 2022] → [Wu et al. 2025], 분류 (B)→(D) sequence-model 계열로 이동 |
| `section3_3_v8.md` | Goal-Delay POMDP + Prop 3.4 | K8: stage → phase 표기 통일 |
| `section3_4_v8.md` | Forecaster + Theorems 3.12/3.13 + Cor 3.14 | K9: Liotet 2022 인용 제거. K10: **Remark 3.16 신설** — Wu et al. 2025 DFBT 와의 차별점 4가지 명시 (Cor 3.14 직후 위치) |
| `section3_5_v8.md` | Asymmetric AC | K1: Definition 3.15 의 actor signature 두 layer 명시. K2: Prop 3.18 BPQL ↔ goal-delay mapping. K3: aliasing 표현 → "agent state aliasing" (Lambrechts 2025 원문). K11: Bauersfeld 71% 인용 통째 제거 (Remark 3.19 표 행 + Remark 3.20 항목 3) |
| `section3_6_v8.md` | AMS-DRL Co-Training | K4: Phase A forecaster freeze 표현 강화. K5: Phase B 의 attacker critic $Q^a_\theta$ freeze 명시. K6: Defender critic $Q^d_\theta(\cdot, a^d) \to V^d_\theta(\cdot)$ V-critic 표기로 PPO 호환. K7: §3.6.5 Item 4 분포 shift 한 문장 추가. K8: phase 표기 통일. K12: "PETN framework" → "AMS-DRL framework" |
| `section3_summary_v8.md` | 정리 표 | Remark 3.16 추가, K11 보류 항목 신설 (asymmetric AC quadrotor 선례 보강), Wu et al. 2025 DFBT 차별점 framing 명시 |

---

## 🔍 v7 → v8 핵심 변화 요약

**최우선 (인용 정확성 정정)**:

- **K9**: DFBT 인용 [Liotet 2022] → [Wu et al. 2025] (DFBT 의 본저자, ICML 2025 PMLR 267:67810-67832, arXiv:2505.00546). Liotet 2022 (DIDA, imitation 기반) 와 DFBT 는 무관.
- **K11**: Bauersfeld & Scaramuzza 2026 narrow pipe 71% 인용 제거. 71% 는 observation 에 flow measurement 추가한 효과지 privileged critic 효과가 아님 — 잘못된 attribution 회피.

**최우선 (Contribution boundary 재설정)**:

- **K10**: §3.4 에 Remark 3.16 신설. Wu et al. (2025) DFBT 가 direct vs recursive forecasting 의 framing 과 점근적 우월성을 이미 single-agent 세팅에서 명시했음을 정직하게 인정. 본 논문의 novelty 를 (i) goal-delay POMDP 정의, (ii) reactive defender belief 분해 (Prop 3.7), (iii) closed-loop kernel Lipschitz 도입, (iv) AMS-DRL 와의 통합 으로 명시적으로 재배치.

**보통 (minimal-diff 정정)**:

- K1: Actor signature 두 layer 명시
- K2: BPQL ↔ goal-delay mapping 한 줄 추가
- K3: aliasing 표현 → "agent state aliasing" (Lambrechts 2025 Appendix A 원문 표현)
- K4: Phase A forecaster freeze 표현 명료화
- K5: Phase B attacker critic freeze 명시
- K6: Defender critic V-critic 표기 (PPO 호환)
- K7: §3.6.5 Item 4 분포 shift 한 문장
- K8: Stage 1/2/3 표기 제거, "사전 단계 / Phase A · B" 로 통일
- K12: "PETN framework" → "AMS-DRL framework" (1 곳)

---

*📝 본 문서는 논문 본문 초고이자 Notion 학습 노트. 각 결과를 본인 말로 막힘 없이 설명할 수 있으면 Section 4 로 넘어갈 준비 완료.*

*v8 의 핵심 정직성 강화: (1) DFBT 의 본저자 (Wu et al. 2025) 정확한 귀속, (2) Wu 2025 와의 contribution boundary 명시 (Remark 3.16), (3) Bauersfeld 71% 잘못된 attribution 제거 — 인용 정확성 + 정직한 selfpositioning.*