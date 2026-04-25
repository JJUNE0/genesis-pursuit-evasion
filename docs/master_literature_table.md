# Master Literature Table — ICRA 2027 / RAL

> **논문 (working title)**: Asymmetric Partial-Observation RL for 1v1 Drone Pursuit-Evasion with Targeted Navigation
> **작성일**: 2026-04-17 (v1) → 2026-04-24 (v2, POV 전환 + AMS-DRL 반영 + Gemini 딥리서치 3회 통합)
> **용도**: Section 2 Related Work 작성 시 참고 + 논문 전반 인용 관리
> **Target venue**: ICRA 2027 (~9월 마감) 또는 RAL (rolling)

> **v1 → v2 주요 변경**:
> 1. POV 전환 (C-UAS defender → 공격 드론 attacker) 수용
> 2. Defender 학습 방식: scripted → AMS-DRL co-training 확정
> 3. 1v1 드론 pursuit-evasion 3편 tier 승격 (Sánchez, Xiao, Chen 2024 OPEN)
> 4. 신규 논문 13편 추가 (Gemini 딥리서치 3회 검증 후 할루시네이션 제거)
>    - Prompt #1 (AMS-DRL/AMSPB 세부): 3편
>    - Prompt #2 (comm asymmetry + 2025-2026 경쟁작): 6편
>    - Prompt #3 (이론 framing follow-ups): 4편
> 5. 할루시네이션 필터링: P3DP, MAGIC, NFSP-D3RN×2, "Observation 3.5=scheduling 연결", "Wang 2026 What Matters" 저자 오류 등 **10+편 제거**

---

## 📊 Quick-Access Summary Table (v2: 28편)

### T1 (본문 직접 비교 필수) — 6편

| # | Citation Key | Area | 핵심 키워드 | 인용 섹션 |
|---|---|---|---|---|
| 1 | **Sánchez Roncero, Cai et al. 2025** | 1v1 Drone PE | body-rate, AMSPB, KTH | Intro, §2.3, §5 (비교) |
| 2 | Li et al. 2025 | Drone Delay RL | TEM, async perception, sim-to-real | Intro, §2.1, §2.3 |
| 3 | Wu et al. 2025 (DFBT) | Delayed RL | Direct belief forecasting, transformer | §2.1, §3.4, §5 (baseline) |
| 13 | **Xiao & Feroskhan 2024 (AMS-DRL)** ⬆T3→T1 | Drone PETN | bipartite training, 3-stage AMS-DRL, Nash eq | **§2.3 (메인)**, §3.6 (framework 직접 채택) |
| 19 | **Xiang et al. 2025 (CI-HRL)** 🆕 | Multi-UAV PE | hierarchical consensus, MAPPO, coop evasion | §2.3 (가장 가까운 최신 경쟁, 2025년 TNNLS) |
| 25 | **Chen J., Yu C. et al. 2024/2025 (SimpleFlight)** 🆕 | Quadrotor Sim-to-Real | input design, SysID+DR, RA-L | §2.3, **§4 (method 직접 참고)** |

### T2 (이론 framing 핵심) — 10편

| # | Citation Key | Area | 핵심 키워드 | 인용 섹션 |
|---|---|---|---|---|
| 4 | Lambrechts et al. 2025 | Asymmetric AC | Privileged POMDP, aliasing elimination | §3.5 (메인 이론) |
| 5 | Ebi et al. 2025 | Asymmetric AC | Informed AAC, unbiased privileged signal | §3.5 (critic 정당화) |
| 6 | Pinto et al. 2017 | Asymmetric AC | AAC 원조, sim-to-real | §2.2, §3.5 |
| 7 | Baisero & Amato 2022 | Asymmetric AC | Unbiased AAC theory (tabular) | §2.2, §3.5 |
| 8 | Ong et al. 2010 | MOMDP | Factored observability 원조 | §2.2, §3.3 |
| 9 | Kim et al. 2023 (BPQL) | Delayed RL | Belief projection, continuous | §2.1, §3.5 |
| 14 | Chen J., Yu C. et al. 2024 (OPEN) | Drone PE | Multi-UAV, evader prediction, ICRA 2025 | §2.3 |
| 16 | **Pope et al. 2021** 🆕 | Aerial Combat RL | HRL air-to-air, ICUAS | §2.3 (adversarial aerial precedent) |
| 20 | **Sun Yihao et al. 2025** 🆕 | Multi-UAV PE (real) | KE-MATD3, Crazyflie, IROS | §2.3 (physical validation) |
| 21 | **Gong Xudong et al. 2025 (VVC-Gym)** 🆕 | Goal-Conditioned RL | fixed-wing velocity vector, multi-goal, ICLR | §2.2 (GCRL precedent) |

### T3 (background / context) — 12편

| # | Citation Key | Area | 핵심 키워드 | 인용 섹션 |
|---|---|---|---|---|
| 10 | Kaufmann et al. 2023 (Nature) | Drone RL | Champion racing, body rates | Intro, §2.3 |
| 11 | Katsikopoulos & Engelbrecht 2003 | Delayed MDP 이론 | CDMDP, augmented state | §3.2 |
| 12 | Zhang et al. 2023 (Game of Drones) | Pursuit RL | Multi-UAV, target prediction | §2.3 |
| 15 | Lee et al. 2023 (HOMDP) | Privileged Info | Hindsight observability | §2.2 |
| 17 | **Chen C. et al. 2025** 🆕 | Dogfight RL | AOS (angle of sight), Aerospace | §2.3 |
| 18 | **DeMay et al. 2022 (AlphaDogfight)** 🆕 | Aerial Combat RL | APL Tech Digest, DARPA ACE | §2.3 (DARPA 전개 언급 시) |
| 22 | **Selmonaj et al. 2025** 🆕 | Combat MARL | NATO, explainable, strategic | §2.3 |
| 23 | **Search G. 2025 (Tunnel)** 🆕 | Aerial Combat RL Env | open-source env, X-62A 대체 | §2.3 (env 언급 시) |
| 24 | **Scukins et al. 2023** 🆕 | BVR Combat RL | ICUAS, situational awareness | §2.3 |
| 26 | **Bauersfeld & Scaramuzza 2026 (Narrow Pipe)** 🆕 | Quadrotor Control | privileged critic + LSTM, T-RO | §3.5 (empirical 뒷받침) |
| 27 | **Zhou et al. 2025 (SWEET-RL)** 🆕 | LLM RL | asymmetric critic with training-time info | §3.5 (cross-domain remark) |
| 28 | **Park et al. 2025 (OGBench)** 🆕 | GCRL Benchmark | offline goal-conditioned, ICLR | §2.2 (GCRL context) |

**Tier**: T1 = 본문 직접 비교 필수, T2 = 이론 framing에 핵심, T3 = background / context

**통계**: T1 6편 / T2 10편 / T3 12편 = 총 **28편**

---

## 📚 Detailed Entries

각 entry 구조:
- **Full citation**
- **Venue**
- **Core contribution**
- **Relevance to us**
- **Distinction**
- **Citation sections**
- **Quote 재료** (필요 시 그대로 쓸 수 있는 한 줄)

---

### [1] Sánchez Roncero, Cai, Andersson, Ögren (2025) ⭐⭐⭐

**Full citation**:
Sánchez Roncero, A., Cai, Y., Andersson, O., & Ögren, P. (2025). Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games. *arXiv preprint arXiv:2506.02849.*

**Venue**: arXiv (KTH, WASP-funded). ICRA/IROS 투고 가능성 높음.

**Core contribution**:
- 1v1 quadrotor pursuit-evasion에 body-rate RL 적용
- AMSPB (Asynchronous Multi-Stage Population-Based) 훈련 알고리즘 제안 — catastrophic forgetting 완화
- Isaac Sim + AscTec Hummingbird, PPO 3-layer MLP
- Body-rate 정책이 velocity 정책보다 peak speed 12.90 m/s vs 10.49 m/s로 압도

**Relevance to us**:
- 시나리오, action space, motivation 모두 우리와 거의 동일 (1v1, body rates, quadrotor pursuit-evasion)
- **우리와 가장 가까운 경쟁작** — reviewer가 반드시 언급할 가능성 큼

**Distinction**:
- **POV 차이**: 그들은 **pursuer 학습 (attacker=evader scripted)** → **우리는 attacker 학습 (defender=pursuer AMS-DRL 학습)**
- **Orthogonal 축**: 그들은 AMSPB adversarial training 축, 우리는 comm delay 축
- 그들은 $\Delta p_{t-4:t}$를 즉시 관측 가정 → 우리는 defender 관측에 $\tau$ step 지연
- Forecaster 없음, asymmetric critic 없음, packet loss 없음, goal-conditioning 없음

**Citation sections**:
- **Introduction**: 1v1 quadrotor PE 분야의 "가장 최근 work, POV 반대 & comm delay orthogonal"
- **§2.3 Drone Pursuit-Evasion**: 그들의 AMSPB 아이디어와 Xiao의 AMS-DRL 비교
- **§5 Experiments**: 가능하면 direct comparison (그들 세팅 + 우리 delay/goal-conditioning 추가)

**Quote 재료**:
> "The most closely related work to ours is Sánchez Roncero et al. (2025), who address 1v1 quadrotor pursuit-evasion with body-rate RL policies and population-based adversarial training. Our work addresses two orthogonal axes: (1) the attacker's perspective with targeted navigation under partial observation, and (2) the communication dimension enforcing asymmetric $\tau$-step delay between combatants."

---

### [2] Li, Zhou, Li, Gong, Mei (2025) ⭐⭐⭐

**Full citation**:
Li, Y., Zhou, Z., Li, H., Gong, Y., & Mei, J. (2025). Agile in the Face of Delay: Asynchronous End-to-End Learning for Real-World Aerial Navigation. *arXiv preprint arXiv:2509.13816.*

**Venue**: arXiv (Harbin Institute of Technology, Shenzhen). ICRA 투고 형식.

**Core contribution**:
- High-frequency IMU + low-frequency LiDAR perception의 temporal mismatch 해결
- **Temporal Encoding Module (TEM)**: Age of Information (AoI)으로 policy conditioning
- Asynchronous RL framework (perception stream과 control loop 분리)
- 100Hz control, zero-shot sim-to-real 검증

**Relevance to us**:
- Drone RL + delay 조합의 SOTA — 우리 방법의 직접 경쟁작
- TEM 아이디어: "delay를 명시적 입력으로" 는 우리 forecaster와 유사 정신

**Distinction**:
- **Internal perception delay** (LiDAR 처리 lag) vs **우리 external comm delay** (target 정보 lag)
- Single-agent navigation through obstacles vs 우리 1v1 pursuit with reactive target
- TEM은 AoI encoding만 (예측 없음) vs 우리 direct trajectory forecaster

**Citation sections**:
- **Introduction**: 드론 RL에서 delay 다룬 최근 work 소개
- **§2.1 Delayed RL** 또는 **§2.3 Drone Pursuit-Evasion RL**
- Method 섹션 비교 시 "단순 AoI encoding vs forecaster"

**Quote 재료**:
> "Li et al. (2025) address internal perception latency via a Temporal Encoding Module. Our setting differs in that latency arises externally from adversarial target telemetry, requiring explicit state prediction rather than just delay-aware conditioning."

---

### [3] Wu et al. (2025) DFBT ⭐⭐⭐

**Full citation**:
Wu, Q., Wang, Y., Zhan, S. S., et al. (2025). Directly Forecasting Belief for Reinforcement Learning with Delays. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*.

**Venue**: ICML 2025

**Core contribution**:
- Recursive belief propagation → **direct multi-step forecasting** via Transformer
- Compounding error 방지 증명 (Theorem 5.5)
- DFBT-SAC 알고리즘, multi-step bootstrapping
- MuJoCo continuous control에서 state-of-the-art

**Relevance to us**:
- Direct forecasting 방법론의 원조 — 우리 forecaster 설계의 직접 기반
- Theorem 5.5의 compounding error bound 우리 Theorem 3.12의 출발점

**Distinction**:
- **Uniform delay** (모든 관측) vs **우리 asymmetric goal-delay** (타겟만)
- Single-agent non-adversarial vs 우리 1v1 reactive target
- 우리는 ego trajectory conditioning 추가 (reactive target 예측에 필요)

**Citation sections**:
- **Introduction**: Direct forecasting 아이디어 등장
- **§2.1 Delayed RL**: 메인 비교 (full-obs vs goal-delay)
- **§3.4 Forecaster**: Theorem 3.12 유도 시 DFBT Theorem 5.5 인용
- **§5 Experiments**: DFBT를 full-obs-delay baseline으로 포팅

**Quote 재료**:
> "DFBT (Wu et al., 2025) pioneered direct trajectory forecasting to avoid the compounding errors of recursive belief propagation. We extend this paradigm to the asymmetric goal-delay setting, where only target information is delayed while ego state remains instantaneous."

---

### [4] Lambrechts, Ernst, Mahajan (2025) ⭐⭐

**Full citation**:
Lambrechts, G., Ernst, D., & Mahajan, A. (2025). A Theoretical Justification for Asymmetric Actor-Critic Algorithms. *Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267, 32375–32405.* arXiv:2501.19116.

**Venue**: ICML 2025

**Core contribution**:
- **Privileged POMDP** framework 도입 ("full at training, partial at execution")
- Asymmetric AC의 finite-time convergence analysis
- **Aliasing error 제거** 이론 증명 — critic이 ground-truth 접근하면 agent state aliasing 해소
- Linear function approximator 기준, but framework는 일반화 가능

**Relevance to us**:
- **Section 3.5 asymmetric critic의 메인 이론 근거**
- 우리 세팅이 정확히 "Privileged POMDP": 학습 시 target ground truth 접근, 배포 시 forecaster만 사용
- "Temporal aliasing" 용어 — 우리 $\tau$-step delay가 agent state aliasing을 야기한다고 재프레이밍 가능

**Distinction**:
- 그들은 일반 POMDP 이론, 우리는 goal-delay 특수 케이스
- 우리는 trajectory forecaster를 추가로 설계 (그들은 actor architecture 자유)
- Linear approx 이론 → 우리는 deep neural network (empirical 검증)

**Citation sections**:
- **§2.2 Asymmetric Learning**: 메인 소개
- **§3.5 Asymmetric Actor-Critic**: 메인 이론 근거
- **§4 Method**: PPO critic 설계 시 근거

**Quote 재료**:
> "Our setting is a Privileged POMDP (Lambrechts et al., 2025): the simulator provides ground-truth target state $s_t^{\text{tgt}}$ during training, but the deployed policy observes only the delayed $s_{t-\tau}^{\text{tgt}}$ and its forecasted future. This asymmetry provably reduces temporal aliasing in the critic's value estimation."

---

### [5] Ebi, Lambrechts, Ernst, Böhm (2025) ⭐⭐

**Full citation**:
Ebi, D., Lambrechts, G., Ernst, D., & Böhm, K. (2025). Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access. *arXiv preprint arXiv:2509.26000.* (EWRL 2025 submission)

**Venue**: arXiv, EWRL 2025

**Core contribution**:
- Critic에 **full state가 아닌 arbitrary privileged signal만 제공해도 unbiased policy gradient** 증명
- Theorem 4.1: Informed history Q-function $Q(h_t, i_t, a_t)$로 일반화
- Baisero & Amato (2022)의 Corollary로 환원 ($i_t = s_t$ 특수 케이스)
- Informativeness 측정 기준: HSCIC (kernel-based), prediction-error based

**Relevance to us**:
- **우리 asymmetric critic 정당화의 핵심 이론**
- 우리 세팅에서 critic은 "target ground truth"만 받음 (full state 아님)
- 이 논문 없으면 "왜 critic에 full state를 안 주냐"는 reviewer 질문에 답하기 어려움

**Distinction**:
- 그들은 일반 POMDP에서 privileged signal 선택 문제
- 우리는 이미 정해진 privileged signal (target state) 활용, signal selection 안 함
- Informativeness criteria는 future work로만 언급

**Citation sections**:
- **§3.5**: Informed AAC framework로 우리 critic 정당화
- **Remark** 수준으로 reviewer defense ("full state 없어도 unbiased")

**Quote 재료**:
> "Following the informed asymmetric actor-critic framework (Ebi et al., 2025), our critic $Q_\theta(s_t^e, s_t^{\text{tgt, true}}, a_t)$ conditions on ground-truth target state as the privileged signal without requiring access to the full environment state, preserving unbiased policy gradient estimates."

---

### [6] Pinto, Andrychowicz, Welinder, Zaremba, Abbeel (2017) ⭐⭐

**Full citation**:
Pinto, L., Andrychowicz, M., Welinder, P., Zaremba, W., & Abbeel, P. (2017). Asymmetric Actor Critic for Image-Based Robot Learning. *Robotics: Science and Systems (RSS).*

**Venue**: RSS 2017

**Core contribution**:
- Asymmetric AC paradigm **원조 논문**
- Actor는 이미지 (partial) 관측, critic은 물리 상태 (full) — sim-to-real 효과
- Block stacking, picking 등 로봇 조작에서 symmetric 대비 월등

**Relevance to us**:
- Asymmetric AC의 기원을 언급하지 않으면 안 됨 — 무조건 인용

**Distinction**:
- 그들은 **sensor modality 비대칭** (image vs state)
- 우리는 **temporal 비대칭** (delayed vs current)
- 우리는 이론적 근거 추가 (Lambrechts 2025, Ebi 2025)

**Citation sections**:
- **§2.2**: Asymmetric learning 역사 소개
- **§3.5**: "following the seminal work of..."

---

### [7] Baisero & Amato (2022) ⭐⭐

**Full citation**:
Baisero, A., & Amato, C. (2022). Unbiased Asymmetric Reinforcement Learning under Partial Observability. *Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS), pp. 44–52.*

**Venue**: AAMAS 2022

**Core contribution**:
- Naive AAC가 policy gradient에 bias 발생 가능성 증명
- **History-state value function $Q(h_t, s_t, a_t)$** 정의 — unbiased AAC algorithm
- Theoretical bounds in discrete POMDPs

**Relevance to us**:
- Unbiased AAC 이론의 근거
- Ebi et al. 2025가 이것을 일반화 — 우리는 두 논문 함께 인용

**Distinction**:
- Discrete, tabular POMDP 중심
- 우리는 continuous deep RL

**Citation sections**:
- **§2.2**, **§3.5** (Ebi et al.과 함께)

---

### [8] Ong, Png, Hsu, Lee (2010) ⭐⭐

**Full citation**:
Ong, S. C. W., Png, S. W., Hsu, D., & Lee, W. S. (2010). Planning under Uncertainty for Robotic Tasks with Mixed Observability. *International Journal of Robotics Research, 29(8), 1053–1068.*

**Venue**: IJRR 2010

**Core contribution**:
- **MOMDP (Mixed Observable MDP) 원조**
- 상태를 $s = (x, y)$로 factorize — $x$ 완전 관측, $y$ 부분 관측
- Belief은 $y$에만 유지 → 계산 효율 향상

**Relevance to us**:
- **우리 goal-delay POMDP formulation의 구조적 기반**
- $x$ = ego ($s^e$), $y$ = target ($s^g$) 대응
- MOMDP 인용 없이 우리 formulation을 "새로운 POMDP"라 주장하면 reviewer가 바로 지적

**Distinction**:
- 그들의 $y$는 "순간 noise 또는 occlusion"
- 우리 $y$는 "지연된 관측" — 시간적 차원 추가
- 우리는 RL + deep learning, 그들은 exact planning (SARSOP)

**Citation sections**:
- **§2.2 Mixed Observability**: MOMDP 소개
- **§3.1 Preliminaries**: 상태 factorization 정당화
- **§3.3 Goal-Delay POMDP**: "MOMDP의 temporal 확장"이라 positioning

**Quote 재료**:
> "We formalize our problem as an instance of Mixed Observable MDPs (Ong et al., 2010), where the ego state $s_t^e$ is fully observable and the target state $s_t^g$ is partially observable due to communication delay. This temporal specialization of the MOMDP framework exposes structural advantages that we exploit in Sections 3.4 and 3.5."

---

### [9] Kim, Kim, Kang, Baek, Han (2023) — BPQL ⭐⭐

**Full citation**:
Kim, J., Kim, H., Kang, J., Baek, J., & Han, S. (2023). Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback. *Advances in Neural Information Processing Systems (NeurIPS), 36.*

**Venue**: NeurIPS 2023

**Core contribution**:
- Delayed RL에서 augmented state 폭발 회피 위한 **belief projection**
- Critic이 augmented state 아닌 원래 state space에서 Q-value 평가
- Long-delay continuous control에서 TD learning 안정화

**Relevance to us**:
- Delayed RL에서 asymmetric critic 아이디어의 선행
- 우리 Proposition 3.16 (BPQL의 goal-delay 특수화) 직접 근거

**Distinction**:
- Uniform delay 가정
- Global belief projection
- 우리는 target-only projection (훨씬 단순)

**Citation sections**:
- **§2.1**: Delayed RL 방법론 소개
- **§3.5**: "Our asymmetric critic is a natural specialization of BPQL's belief projection"

---

### [10] Kaufmann et al. (2023) ⭐

**Full citation**:
Kaufmann, E., Bauersfeld, L., Loquercio, A., Müller, M., Koltun, V., & Scaramuzza, D. (2023). Champion-level drone racing using deep reinforcement learning. *Nature, 620(7976), 982–987.*

**Venue**: Nature 2023

**Core contribution**:
- Swift: 실세계 드론 레이싱에서 인간 챔피언 격파
- Simulation에서 PPO로 학습 → zero-shot real drone
- Body-rate control + CNN visual encoder

**Relevance to us**:
- Drone RL이 실세계 성능 도달 가능함을 입증한 landmark
- Body-rate control 선택의 근거
- CoRL reviewer 모두가 알고 있음

**Distinction**:
- Static gate 레이싱 (pre-mapped track)
- 우리는 moving adversarial target

**Citation sections**:
- **Introduction**: Drone RL이 성숙한 분야임을 강조
- **§2.3 Drone Control**: Body-rate control 정당화

---

### [11] Katsikopoulos & Engelbrecht (2003) ⭐

**Full citation**:
Katsikopoulos, K. V., & Engelbrecht, S. E. (2003). Markov decision processes with delays and asynchronous cost collection. *IEEE Transactions on Automatic Control, 48(4), 568–574.*

**Venue**: IEEE TAC 2003

**Core contribution**:
- Delayed MDP를 augmented state MDP로 reduce (CDMDP theorem)
- Constant & random delay 모두 커버
- Delayed RL 분야의 foundational theoretical result

**Relevance to us**:
- 우리 §3.2 full-obs-delay formulation의 출발점
- 이걸 인용 안 하면 delayed RL 이론 부족해 보임

**Distinction**:
- Discrete, finite state spaces
- 우리는 continuous + neural approximation

**Citation sections**:
- **§3.2 Full-Observation-Delay POMDP**: CDMDP 이론 근거

---

### [12] Zhang, Zong, Zhang, Dou, Tian (2023) — Game of Drones ⭐

**Full citation**:
Zhang, R., Zong, Q., Zhang, X., Dou, L., & Tian, B. (2023). Game of Drones: Multi-UAV Pursuit-Evasion Game With Online Motion Planning by Deep Reinforcement Learning. *IEEE Transactions on Neural Networks and Learning Systems, 34(10), 7900–7909.*

**Venue**: IEEE TNNLS 2023

**Core contribution**:
- Multi-quadcopter vs single target pursuit
- **CBC-TP Net**: MADDPG + target prediction network
- 도시 환경 시뮬레이션 (PES framework)

**Relevance to us**:
- Multi-UAV pursuit + prediction network 사용 — 우리 forecaster 개념과 유사
- C-UAS 분야 내 positioning에 유용

**Distinction**:
- Multi-agent (여러 pursuer), 우리는 1v1
- Prediction network는 MADDPG 내부 module — 우리처럼 standalone forecaster 아님
- Comm delay, packet loss 없음

**Citation sections**:
- **§2.3 Drone Pursuit-Evasion RL**: 기존 prediction-augmented 방법 소개

---

### [13] Xiao & Feroskhan (2024) — AMS-DRL ⭐⭐⭐ (T3→**T1 승격**, v2)

**Full citation**:
Xiao, J., & Feroskhan, M. (2024). Learning Multi-Pursuit Evasion for Safe Targeted Navigation of Drones. *IEEE Transactions on Artificial Intelligence.*

**Venue**: IEEE TAI 2024

**Core contribution**:
- **AMS-DRL** (Asynchronous Multi-Stage DRL): bipartite 학습 graph로 pursuer/evader 동시 학습
  - Stage 1: evader fixed, pursuer 학습
  - Stage 2: pursuer fixed, evader 학습
  - Stage 3: bipartite alternating until near-Nash
- Nash equilibrium 수렴 보장 (bipartite alternating theoretical guarantee)
- "Targeted Navigation" 개념 도입 (PETN): evader가 단순 생존이 아니라 **목적지 도달 + 회피** 동시 최적화
- Real quadrotor (Crazyflie) 비행 검증

**Relevance to us — T1 승격 이유**:
- **우리 §3.6 AMS-DRL 프레임워크의 직접 채택 대상** (v2 pivot 이후 핵심 방법론)
- "Targeted Navigation" 용어와 구조가 우리 논문 제목 "Targeted Navigation"과 정확히 일치 — 메인 base 논문
- 우리 attacker (공격 드론) 역할 = 그들의 **evader with targeted navigation** (재분류 후 동일 역할)
- Bipartite co-training이 우리 §3.6 학습 절차의 토대

**Distinction** (우리가 AMS-DRL을 확장하는 축):
- 그들: Multi-pursuer, perfect info 가정 (delay 없음, packet loss 없음), deterministic evasion
- 우리: **1v1** (more tractable for delay analysis), **asymmetric $\tau$-delay** (defender 관측에 지연), **direct forecaster** for target state prediction, **asymmetric critic** (privileged training)
- 그들은 targeted navigation을 motivation으로만 다룸 — 우리는 $g_{\text{mission}}$을 **명시적 conditioning variable**로 formulation에 포함

**Citation sections**:
- **Introduction**: "PETN task 원조 & 우리 framework의 출발점"
- **§2.3 Drone Pursuit-Evasion RL**: AMS-DRL을 중심 문헌으로 소개
- **§3.6 Adversarial Co-training**: framework 직접 채택 + 우리 확장 설명
- **§5 Experiments**: 기본 AMS-DRL을 baseline 중 하나로 구현

**Quote 재료**:
> "Our work builds directly on the AMS-DRL framework of Xiao and Feroskhan (2024), who introduce pursuit-evasion with targeted navigation (PETN) and propose bipartite alternating training for Nash convergence. We extend their perfect-information formulation along two orthogonal axes: (i) asymmetric communication delay in attacker observations of the defender, and (ii) direct transformer-based state forecasting integrated with an asymmetric critic trained under privileged full-state access."

---

### [14] Chen, Yu et al. (2024) — OPEN ⭐

**Full citation**:
Chen, J., Yu, C., Li, G., Tang, W., Yang, X., Xu, B., Yang, H., & Wang, Y. (2024). Multi-UAV Pursuit-Evasion with Online Planning in Unknown Environments by Deep Reinforcement Learning. *arXiv preprint arXiv:2409.15866.*

**Venue**: arXiv (Tsinghua)

**Core contribution**:
- Multi-UAV pursuit with **evader prediction-enhanced network (EPN)**
- Adaptive environment generator (curriculum)
- CTBR (collective thrust + body rates) + real quadrotor deployment
- Zero-shot sim-to-real

**Relevance to us**:
- Multi-UAV pursuit의 최신 작업 + evader prediction
- Sim-to-real 성공 — 우리 future work 비교

**Distinction**:
- Multi-agent 3 vs 1
- EPN은 MAPPO 내 module — 우리 standalone forecaster 아님
- Comm delay, packet loss 없음

**Citation sections**:
- **§2.3**: 최신 multi-UAV pursuit

---

### [15] Lee, Agarwal, Dann, Zhang (2023) — HOMDP ⭐

**Full citation**:
Lee, J. N., Agarwal, A., Dann, C., & Zhang, T. (2023). Learning in POMDPs is Sample-Efficient with Hindsight Observability. *Proceedings of the 40th International Conference on Machine Learning (ICML).*

**Venue**: ICML 2023

**Core contribution**:
- **Hindsight Observable MDP (HOMDP)** framework
- 학습 시 latent state 공개, 실행 시 감춰짐 — 시뮬레이터 기반 RL의 수학적 모델
- Polynomial sample complexity bounds

**Relevance to us**:
- 우리 시뮬레이터 기반 학습 세팅의 이론적 근거
- Privileged information의 sample-efficiency 보장

**Distinction**:
- Episodic hindsight (에피소드 끝에 공개)
- 우리 setting: 실시간 simulator access + rolling window
- 일반 POMDP vs 우리 MOMDP 특수화

**Citation sections**:
- **§2.2** 또는 **§3.5**: Privileged info 활용의 이론적 배경

---

## 📚 Detailed Entries — 신규 13편 (v2 추가)

---

### [16] Pope, Ide, Mićović, Diaz, Rosenbluth, Ritholtz, Twedt, Walker, Alcedo, Javorsek (2021) ⭐⭐

**Full citation**:
Pope, A. P., Ide, J. S., Mićović, D., Diaz, H., Rosenbluth, D., Ritholtz, L., Twedt, J., Walker, T. T., Alcedo, K., & Javorsek, D. (2021). Hierarchical Reinforcement Learning for Air-to-Air Combat. *International Conference on Unmanned Aircraft Systems (ICUAS).* arXiv:2105.00990.

**Venue**: ICUAS 2021 (Lockheed Martin + DARPA ACE 계열)

**Core contribution**:
- HRL로 air-to-air 1v1 dogfight 해결 — high-level tactical manager + low-level skill policies
- AlphaDogfight Trials 우승 아이디어 기반
- League/self-play로 안정적 adversarial 학습

**Relevance to us**:
- 1v1 adversarial aerial RL의 **foundational precedent**
- Self-play / league training이 우리 AMS-DRL 선행
- 우리 "Drone PE as adversarial aerial combat" 맥락 연결

**Distinction**:
- Fixed-wing fighter (kinematic 차이), 우리는 quadrotor
- Perfect info 가정, 우리는 asymmetric delay
- Tactical HRL structure, 우리는 flat body-rate

**Citation sections**: §2.3 (adversarial aerial RL precedent, AMS-DRL 도입 전 historical link)

---

### [17] Chen, Wu, Shi, Liu (2025) — AOS Dogfight ⭐

**Full citation**:
Chen, C., Wu, Y., Shi, Y., & Liu, H. (2025). Close-range air combat maneuver decision based on deep reinforcement learning and sight angle observation. *Aerospace, 12(3), 265.*

**Venue**: Aerospace (MDPI) 2025

**Core contribution**:
- Angle-of-Sight (AOS) 기반 관측 설계로 dogfight 결정
- PPO + curriculum

**Relevance to us**: Adversarial aerial RL의 최근 예, 관측 설계(angle-only) 아이디어 참고

**Distinction**: Fighter aircraft, perfect info, 1v1

**Citation sections**: §2.3 (관측 설계 다양성 예시)

---

### [18] DeMay, Moriarty, Schreiber, Duffy (2022) — AlphaDogfight ⭐

**Full citation**:
DeMay, B., Moriarty, J., Schreiber, R., & Duffy, J. (2022). AlphaDogfight: Transforming Autonomous Tactical Behaviors with Deep Reinforcement Learning. *Johns Hopkins APL Technical Digest, 36(2).*

**Venue**: Johns Hopkins APL Technical Digest 2022

**Core contribution**:
- DARPA AlphaDogfight Trials 결과 정리
- Human pilot 격파한 RL 에이전트
- AlphaStar-style league training

**Relevance to us**: Drone/fighter adversarial RL의 성숙도 증거 — Intro에서 맥락 언급 시 유용

**Citation sections**: Intro (aerial combat RL의 실용성), §2.3 배경

---

### [19] Xiang, Wang, Zhang, Liu, Du (2025) — CI-HRL ⭐⭐⭐

**Full citation**:
Xiang, Y., Wang, Y., Zhang, X., Liu, F., & Du, S. (2025). Decentralized Consensus Inference-Based Hierarchical Reinforcement Learning for Multiconstrained UAV Pursuit-Evasion Game. *IEEE Transactions on Neural Networks and Learning Systems.* arXiv:2506.18126.

**Venue**: IEEE TNNLS 2025 (최근 대형 ML 저널)

**Core contribution**:
- Consensus Inference HRL (CI-HRL): high-level consensus aggregation + low-level MAPPO
- Multi-constrained UAV PE (formation coverage + evasion)
- Communication module로 local state → global consensus

**Relevance to us — T1 근거**:
- **2025년 TNNLS에 나온 가장 최근 UAV PE + RL + communication 관련 논문** — 우리 "comm asymmetry" 주장의 direct benchmark
- Reviewer가 "이것과 뭐가 다르냐" 물을 가능성 큼 — 차별화 설명 필수
- Communication constraint를 명시적으로 다룬 최신 연구

**Distinction**:
- 그들: Multi-UAV swarm, **cooperative consensus** (팀 내부 통신)
- 우리: **1v1, adversarial, asymmetric τ-step delay** between combatants
- 그들은 packet loss/delay 모델 없음, consensus 알고리즘이 포커스
- 그들은 formation coverage, 우리는 goal-conditioned targeted navigation

**Citation sections**:
- **§2.3 Drone Pursuit-Evasion**: "가장 최근 PE + comm 관련 문헌, orthogonal 축"
- 차별화 설명 필요 — §2.3 말미 문단으로 deliberate positioning

**Quote 재료**:
> "The most recent related work, Xiang et al. (2025), addresses cooperative consensus communication within a pursuer swarm. Our work is orthogonal: we formalize asymmetric communication delay between adversarial agents (attacker vs defender) rather than consensus within a team."

---

### [20] Sun Yihao, Zhang, Liu, et al. (2025) — KE-MATD3 ⭐⭐

**Full citation**:
Sun, Y., Zhang, C., Liu, H., et al. (2025). Emergent Cooperative Strategies for Pursuit-Evasion in Cluttered Environments: A Knowledge-Enhanced Multi-Agent Deep Reinforcement Learning Approach. *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).*

**Venue**: IROS 2025 (NUDT, National University of Defense Technology)

**Core contribution**:
- KE-MATD3 (Knowledge-Enhanced MATD3) + Improved Artificial Potential Field (IAPF)
- Team reward로 emergent cooperation
- **Physical Crazyflie 검증** — sim-to-real

**Relevance to us**:
- Physical drone PE 검증 precedent — 우리 hardware 실험 방향과 일치
- IROS 2025 fresh publication

**Distinction**:
- Multi-pursuer cooperative (그들), 1v1 adversarial (우리)
- Perfect info 가정
- Goal-conditioning 없음

**Citation sections**: §2.3 (physical drone PE precedent), Experiment section에서 Crazyflie 선택 근거 보강

---

### [21] Gong Xudong et al. (2025) — VVC-Gym ⭐⭐

**Full citation**:
Gong, X., et al. (2025). VVC-Gym: A multi-goal long-horizon Reinforcement Learning environment. *International Conference on Learning Representations (ICLR).*

**Venue**: ICLR 2025

**Core contribution**:
- Fixed-wing UAV velocity vector control 환경
- Multi-goal, long-horizon GCRL benchmark
- Multi-quality demonstration sets

**Relevance to us**:
- **Goal-Conditioned RL for UAV의 최신 benchmark**
- 우리 goal-conditioned 공격 drone의 academic precedent
- "GCRL for aerial vehicles" framing 강화

**Distinction**:
- Single-agent (그들), adversarial (우리)
- Fixed-wing (그들), quadrotor (우리)
- Delay 없음

**Citation sections**: §2.2 (GCRL precedent), §2.3 (UAV GCRL context)

---

### [22] Selmonaj, Hostettler et al. (2025) — Explainable Combat MARL ⭐

**Full citation**:
Selmonaj, A., Hostettler, D., et al. (2025). Explaining Strategic Decisions in Multi-Agent Reinforcement Learning for Aerial Combat Tactics. arXiv:2505.11311. *NATO Science and Technology Organization (STO) Journal (Spring 2026).*

**Venue**: arXiv 2025 / NATO STO 2026

**Core contribution**:
- Combat MARL의 strategic-level explainability
- NATO 계열 공식 연구

**Relevance to us**: 군사 항공 RL 맥락의 최근 연구 — Intro/§2.3에서 분야 성숙도 언급 시

**Distinction**: Strategic/campaign level, 우리는 tactical 1v1

**Citation sections**: §2.3 (adversarial aerial RL 분야 정당화)

---

### [23] Search et al. (2025) — Tunnel ⭐

**Full citation**:
Search, G. (2025). Tunnel: A Reinforcement Learning Environment for High Performance Air Combat. arXiv:2505.01953.

**Venue**: arXiv 2025 (USAF Test Pilot School, 452 FLTS)

**Core contribution**:
- X-62A VISTA 대체 open-source RL env
- High-performance air combat agent training

**Relevance to us**: Open benchmark precedent for adversarial aerial RL

**Citation sections**: §2.3 (environment/benchmark 언급 시)

---

### [24] Scukins et al. (2023) — BVR Situational Awareness ⭐

**Full citation**:
Scukins, E., Klein, M., Kroon, M., & Ögren, P. (2023). Enhancing Situation Awareness in Beyond Visual Range Air Combat with Reinforcement Learning-based Decision Support. *International Conference on Unmanned Aircraft Systems (ICUAS).*

**Venue**: ICUAS 2023

**Core contribution**:
- BVR combat RL 의사결정 지원
- Threat/formation flexibility prediction
- **KTH Ögren 연구실** — Sánchez Roncero와 같은 그룹

**Relevance to us**:
- 같은 그룹 (Ögren)의 BVR precedent — Sánchez Roncero 맥락 강화
- RL + predictive inference in adversarial aerial

**Citation sections**: §2.3 (BVR combat RL precedent)

---

### [25] Chen Jiayu, Yu Chao, Xie Yuqing, et al. (2024/2025) — SimpleFlight ⭐⭐⭐

**Full citation**:
Chen, J., Yu, C., Xie, Y., Gao, F., Chen, Y., Yu, S., Tang, W., Ji, S., Mu, M., Wu, Y., Yang, H., & Wang, Y. (2025). What Matters in Learning a Zero-Shot Sim-to-Real RL Policy for Quadrotor Control? A Comprehensive Study. *IEEE Robotics and Automation Letters.* arXiv:2412.11764.

**Venue**: IEEE RA-L 2025 (Tsinghua thu-uav group, **[14] OPEN 저자와 같은 그룹**)

**Core contribution**:
- **SimpleFlight**: PPO + quadrotor sim-to-real의 5대 필수 요소
  1. Actor input에 velocity + rotation matrix 포함
  2. Critic input에 time vector 추가
  3. Action difference regularization
  4. Selective SysID + DR
  5. Large batch size
- Crazyflie 2.1 zero-shot 검증
- 기본 PPO 대비 trajectory tracking error **50%+ 감소**

**Relevance to us — T1 근거**:
- **우리 §4 method의 직접 참고서** — Crazyflie sim-to-real 세팅 완전 동일
- Asymmetric actor-critic input 설계의 **실용 best practice** (우리 이론 framing의 empirical counterpart)
- Open-source Omnidrones 통합 — 우리 구현 시 직접 활용 가능

**Distinction**:
- Single-agent trajectory tracking (그들), adversarial PE (우리)
- 그들의 asymmetric input은 "time vector를 critic에만"인 간단한 수준 → 우리는 delay-induced structural asymmetry (전체 defender state hidden/forecasted)
- 그들은 delay 없음, 우리는 τ-step delay central

**Citation sections**:
- **§2.3** (quadrotor sim-to-real precedent)
- **§4 Method** (우리 asymmetric input 설계 영감, SimpleFlight 5대 요소 따름)
- **Experiment setup**: Crazyflie + Omnidrones 사용 정당화

**Quote 재료**:
> "We build our policy architecture on the quadrotor sim-to-real best practices identified by Chen et al. (2025), specifically integrating velocity and rotation-matrix representations in the actor and incorporating task-relevant temporal features in the critic. Our asymmetric critic extends their principle by granting access to the full, non-delayed defender state during training."

---

### [26] Bauersfeld & Scaramuzza (2026) — Narrow Pipe ⭐

**Full citation**:
Bauersfeld, L., & Scaramuzza, D. (2026). Low-Latency Event-Based Velocimetry for Quadrotor Control in a Narrow Pipe. *IEEE Transactions on Robotics.* arXiv:2507.15444.

**Venue**: IEEE T-RO 2026 (UZH RPG)

**Core contribution**:
- LSTM-PPO quadrotor policy in narrow pipe
- **Privileged critic** (full state including CFD-derived disturbance wrenches) during training
- Actor는 noisy positional data + temporal history만
- Lateral maneuvers에서 **71% overshoot 감소**

**Relevance to us**:
- Privileged critic 실증 최신 사례 — §3.5 theoretical framing의 empirical 뒷받침
- Recurrent actor + privileged critic 조합이 우리 구조와 유사 정신

**Distinction**: Disturbance rejection (그들), adversarial PE (우리). 그들은 정적 환경 + 공기역학 비대칭.

**Citation sections**: §3.5 (privileged critic의 최근 empirical 성공 사례)

---

### [27] Zhou, Jiang, Tian, Weston, Levine, Sukhbaatar, Li (2025) — SWEET-RL ⭐

**Full citation**:
Zhou, Y., Jiang, S., Tian, Y., Weston, J., Levine, S., Sukhbaatar, S., & Li, X. (2025). SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks. arXiv:2503.15478.

**Venue**: arXiv 2025 (FAIR, UC Berkeley)

**Core contribution**:
- Multi-turn LLM RL에 asymmetric critic 적용
- Critic에 training-time information 접근 허용 → step-level reward
- LLM actor는 delayed/restricted view

**Relevance to us**:
- **Cross-domain precedent**: asymmetric critic이 robotics/POMDP 범위를 넘어 LLM 분야까지 확장
- "Privileged training-time info" 원리의 general applicability 강조 — §3.5 Remark 수준

**Distinction**: LLM 도메인, 우리는 continuous control. 구조만 공유, 구체 구현 다름.

**Citation sections**: §3.5 Remark (cross-domain generality), 선택적 인용

---

### [28] Park, Frans, Eysenbach, Levine (2025) — OGBench ⭐

**Full citation**:
Park, S., Frans, K., Eysenbach, B., & Levine, S. (2025). OGBench: Benchmarking Offline Goal-Conditioned RL. *International Conference on Learning Representations (ICLR).* arXiv:2410.20092.

**Venue**: ICLR 2025 (Seohong Park, UC Berkeley)

**Core contribution**:
- GCRL의 표준 benchmark suite (8 envs, 85 datasets)
- Stitching / long-horizon reasoning / stochasticity / high-dim 고유 도전 분리
- 대표 offline GCRL 6개 알고리즘 reference 구현

**Relevance to us**:
- GCRL 분야의 standard reference — §2.2에서 GCRL context 제시 시
- 우리는 online + goal-conditioned이지만, GCRL 분야 성숙도 언급 필요

**Distinction**:
- Offline (그들) vs online (우리)
- Single-agent (그들) vs adversarial (우리)

**Citation sections**: §2.2 (GCRL context / benchmark 언급 시)

---

## 🗂️ 논문 간 관계도 (v2, POV 전환 + AMS-DRL 반영)

```
              [8] Ong 2010 (MOMDP 원조)
                     │
                     ├─── 특수화 ──→ Our Goal-Delay POMDP (§3.3)
                     │             (attacker observes self+goal, defender delayed τ)
                     │
       ┌─────────────┼─────────────┐
       │                           │
[11] Katsikopoulos 2003           [9] Kim 2023 (BPQL)
    (CDMDP foundation)              │
       │                            │
       └──→ [3] Wu 2025 (DFBT) ──┼──→ Our Direct Forecaster f_ψ (§3.4)
                                  │    (predicts defender state without g_mission)
                                  │
[6] Pinto 2017 (AAC 원조)         │
       │                          │
       └──→ [7] Baisero 2022 ─────┼──→ Our Asymmetric Critic (§3.5)
                │                 │    (training-time full defender access)
                ├──→ [4] Lambrechts 2025 (Privileged POMDP theory)
                │         │
                │         └──→ [5] Ebi 2025 (Informed AAC)
                │
                └──→ [26] Bauersfeld 2026 Narrow Pipe (empirical 사례)
                └──→ [27] Zhou 2025 SWEET-RL (cross-domain 일반성)

[15] Lee 2023 (HOMDP) ──→ 학습 시 privileged access 정당화

[10] Kaufmann 2023 (Nature) ──┐
[25] Chen 2025 SimpleFlight ──┤──→ Quadrotor sim-to-real foundation
                              │     (actor/critic input design 직접 참고)
                              │
[21] Gong 2025 VVC-Gym     ───┤──→ GCRL for UAV precedent
[28] Park 2025 OGBench     ───┤    (offline GCRL context)
                              │
                              ↓
═══════════════════════════════════════════════════
Adversarial Drone PE (우리 main context)
═══════════════════════════════════════════════════

[16] Pope 2021 HRL Air-to-Air    ┐
[18] DeMay 2022 AlphaDogfight    ├─→ Adversarial aerial RL 분야 성숙도
[17] Chen C. 2025 AOS Dogfight   │
[22] Selmonaj 2025 Combat MARL   │
[23] Search 2025 Tunnel          │
[24] Scukins 2023 BVR SA         ┘

[12] Zhang 2023 Game of Drones   ┐
[20] Sun Y. 2025 KE-MATD3 IROS   ├─→ Multi-UAV PE (우리 1v1의 배경)
[14] Chen Y. 2024 OPEN           │
[19] Xiang 2025 CI-HRL TNNLS     ┘    ← 2025 최신 최근접 경쟁

★★★ Core adversarial thread ★★★
[13] Xiao 2024 AMS-DRL ─────→ §3.6 adversarial co-training framework 직접 채택
              │                 (defender=pursuer 학습, PETN 개념)
              │
[1] Sánchez Roncero 2025 KTH ─→ 1v1 quadrotor body-rate RL (POV 반대, orthogonal)
              │
              └──→ Our Contribution (3 axes):
                   (i) attacker POV + goal-conditioned targeted navigation
                   (ii) asymmetric τ-step comm delay (attacker observes delayed)
                   (iii) direct forecaster + asymmetric privileged critic

[2] Li 2025 Agile Face of Delay ──→ Drone + delay (internal perception, orthogonal)
```

---

## 🎯 Section 2 구조 제안 (v2, POV 전환 반영)

```
2. Related Work

2.1 Reinforcement Learning under Delays
    - [11] Katsikopoulos 2003 — foundation (CDMDP)
    - [9]  Kim 2023 BPQL — belief projection
    - [3]  Wu 2025 DFBT — direct forecasting (우리 main baseline)
    - [2]  Li 2025 — drone + internal perception delay

2.2 Asymmetric Learning, Mixed Observability & Goal-Conditioning
    - [8]  Ong 2010 — MOMDP 원조
    - [6]  Pinto 2017 — AAC 원조
    - [7]  Baisero & Amato 2022 — unbiased AAC
    - [4]  Lambrechts 2025 — Privileged POMDP theory
    - [5]  Ebi 2025 — Informed AAC
    - [15] Lee 2023 — Hindsight Observability
    - [26] Bauersfeld 2026 — privileged critic 실증 (empirical)
    - [27] Zhou 2025 SWEET-RL — cross-domain 일반성 (Remark)
    - [21] Gong 2025 VVC-Gym — GCRL for UAV precedent
    - [28] Park 2025 OGBench — GCRL benchmark context

2.3 Drone Pursuit-Evasion and Adversarial Aerial RL
    ─ Quadrotor RL foundation
    - [10] Kaufmann 2023 — drone RL foundation, body rates
    - [25] Chen 2025 SimpleFlight — sim-to-real best practices

    ─ Adversarial aerial combat RL (historical)
    - [18] DeMay 2022 AlphaDogfight — DARPA ACE
    - [16] Pope 2021 HRL Air-to-Air — HRL precedent
    - [24] Scukins 2023 BVR — KTH BVR combat
    - [17] Chen C. 2025 AOS — 최근 dogfight
    - [22] Selmonaj 2025 — combat MARL (NATO)
    - [23] Search 2025 Tunnel — aerial combat RL env

    ─ Multi-UAV Pursuit-Evasion
    - [12] Zhang 2023 Game of Drones — multi-UAV prediction
    - [14] Chen 2024 OPEN — multi-UAV real drone
    - [20] Sun Y. 2025 KE-MATD3 — cooperative emergent (IROS)
    - [19] Xiang 2025 CI-HRL — 최신 TNNLS, consensus 기반

    ─ 1v1 / Targeted Navigation (★ core comparison ★)
    - [13] Xiao 2024 AMS-DRL — **PETN base framework, direct adoption**
    - [1]  Sánchez Roncero 2025 — 1v1 quadrotor body-rate (가장 가까움, POV 반대)

    ─ 우리 논문의 positioning (§2.3 말미)
    - Axis (i): attacker POV + goal-conditioned PETN (extending Xiao)
    - Axis (ii): asymmetric τ-step comm delay (orthogonal to all above)
    - Axis (iii): direct forecaster + asymmetric privileged critic
```

---

## 🔖 자주 쓰는 citation key (BibTeX용, v2)

```bibtex
# Tier 1 (6편)
@article{sanchezroncero2025learned}        # [1] Sánchez Roncero, Cai, Andersson, Ögren 2025
@article{li2025agile}                       # [2] Li 2025
@inproceedings{wu2025dfbt}                  # [3] Wu 2025 DFBT
@article{xiao2024amsdrl}                    # [13] Xiao & Feroskhan 2024 AMS-DRL (T1 승격)
@article{xiang2025cihrl}                    # [19] Xiang 2025 CI-HRL (TNNLS)
@article{chen2025simpleflight}              # [25] Chen 2025 SimpleFlight (RA-L)

# Tier 2 (10편)
@inproceedings{lambrechts2025theoretical}   # [4] Lambrechts 2025
@article{ebi2025informed}                   # [5] Ebi 2025
@inproceedings{pinto2017asymmetric}         # [6] Pinto 2017
@inproceedings{baisero2022unbiased}         # [7] Baisero 2022
@article{ong2010planning}                   # [8] Ong 2010
@inproceedings{kim2023bpql}                 # [9] Kim 2023 BPQL
@article{chen2024open}                      # [14] Chen 2024 OPEN
@inproceedings{pope2021hrl}                 # [16] Pope 2021 HRL air-to-air
@inproceedings{sun2025kematd3}              # [20] Sun Y. 2025 IROS
@inproceedings{gong2025vvcgym}              # [21] Gong 2025 VVC-Gym

# Tier 3 (12편)
@article{kaufmann2023champion}              # [10] Kaufmann 2023 Nature
@article{katsikopoulos2003markov}           # [11] Katsikopoulos 2003
@article{zhang2023game}                     # [12] Zhang 2023 Game of Drones
@inproceedings{lee2023learning}             # [15] Lee 2023 HOMDP
@article{chenc2025aos}                      # [17] Chen C. 2025 AOS
@article{demay2022alphadogfight}            # [18] DeMay 2022 AlphaDogfight APL
@article{selmonaj2025explainable}           # [22] Selmonaj 2025
@misc{search2025tunnel}                     # [23] Search 2025 Tunnel
@inproceedings{scukins2023bvr}              # [24] Scukins 2023 BVR
@article{bauersfeld2026narrow}              # [26] Bauersfeld 2026 T-RO
@misc{zhou2025sweetrl}                      # [27] Zhou 2025 SWEET-RL
@inproceedings{park2025ogbench}             # [28] Park 2025 OGBench
```

---

## ✅ 세션 마감 상태 (v2, 2026-04-24)

**v1 완료** (2026-04-17):
- [x] 15개 기존 논문 정리
- [x] 3개 핵심 논문 존재 검증 (Lambrechts, Ebi, Li)
- [x] 의심 논문 검증 (Pierre "MAGNET" 할루시네이션 확인, Zhang "Game of Drones" 실존)
- [x] Sánchez Roncero 2025 PDF 정독 → 직접 경쟁작 차별화 전략 수립

**v2 완료** (2026-04-24):
- [x] POV 전환 수용 (C-UAS defender → attacker drone)
- [x] Defender 학습 방식: scripted → **AMS-DRL co-training** 확정
- [x] Gemini 딥리서치 3회 진행 & 할루시네이션 필터링
  - [x] Prompt #1 (AMS-DRL 세부): 3편 신규
  - [x] Prompt #2 (comm asymmetry + 2025-26 경쟁작): 6편 신규
  - [x] Prompt #3 (이론 framing): 4편 신규
  - [x] 할루시네이션 10+편 제거 (P3DP, MAGIC, NFSP-D3RN×2, "Observation 3.5=scheduling", "Wang 2026 What Matters" 저자 오류 등)
- [x] [13] Xiao AMS-DRL T3 → **T1 승격**
- [x] [1] Sánchez Roncero 저자에 Yixi Cai 추가
- [x] 신규 13편 상세 정리
- [x] 관계도 v2 작성
- [x] Section 2 구조 v2 작성

**Scoop risk**: ✅ 3개 프롬프트 모두 독립 검증 결과 **no scoop** — 1v1 drone PE + asymmetric τ-delay + goal-conditioned attacker 조합은 2026-04 기준 unoccupied

**다음 세션 할 일**:
- [ ] Section 3 Problem Formulation 재작성 (A1/A2/A3 결정 반영)
  - 단일 attacker POV POMDP formulation
  - s^e → s^a, s^g → s^d 전면 sed
  - g_mission을 observable conditioning variable로 Proposition 3.4/3.7 재증명
  - §3.6 AMS-DRL framework 신설 (bipartite training, DR scope, consistency with §3.1-§3.5)
  - **🚨 Obs 3.5 justification은 기존 엔트로피 argument만 사용 — scheduling theory 연결 절대 금지**
- [ ] Section 2 Related Work 초고 작성 (v2 구조 기반)
- [ ] Sánchez Roncero + Xiao 조합으로 direct comparison 실험 설계

---

*📝 이 문서는 논문 작업 내내 옆에 두고 참고하는 마스터 파일이다. 새 논문 발견 시마다 여기에 업데이트.*