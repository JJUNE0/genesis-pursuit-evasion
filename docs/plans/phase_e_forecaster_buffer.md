# Phase E — Forecaster + Asymmetric Critic + τ-delay (논문 main novelty)

> **상태**: 사용자 plan 작성 단계 (2026-05-08). 코드 변경 큼 — 사용자 승인 후 구현.
> **선행**: v16 baseline (a_check_max 0.559, Nash 균형 X), v17/v18 reward tuning + FP로 Nash 도달 못함 (vanilla alternating PPO 본질 한계).
> **사용자 결정**: Forecaster pretraining 사용 안 함. Replay buffer 기반 학습만.

---

## 1. 동기 (Why)

### 1.1 v16~v18 결과 요약 (vanilla AMS-DRL 한계)

| version | a_check_max | Nash | 비고 |
|---|---:|---|---|
| v16 (Set A + existential) | **0.559** | ❌ | best, 진동 |
| v17 (R_mission_win 20) | 0.470 | cycle 6~7 \|Δ\|=0.14 (일시) | 균형 잠시 |
| v18 (FP, vanilla) | 0.142 | ❌ | FP 도리어 악화 (batch dilution + mixed signal) |

**결론**: vanilla alternating PPO + reward tuning + vanilla FP로는 Nash 안정 수렴 불가. **논문 핵심 메커니즘 (CLAUDE.md §3, §7, §8) 도입 필수**.

### 1.2 논문 main novelty 3가지 (현재 미구현)

| spec | 위치 | 현재 |
|---|---|---|
| **τ-delay** (`τ` step communication delay) | CLAUDE.md §3 | `action_latency_steps=0` 미사용 |
| **Forecaster `f_ψ`** (defender state τ-step 예측) | CLAUDE.md §8 | 없음 |
| **Asymmetric Critic** (privileged obs) | CLAUDE.md §7 | actor=critic obs 동일 |

→ **Phase E에서 3개 모두 도입**.

---

## 2. 구현 순서 — 3 sub-phase (additive only, rollback 가능)

### Phase E.1 — Asymmetric Critic (τ=0, no forecaster)

**목적**: Critic 가치 추정 정확 → policy update 안정 → Nash 균형 가능성. v16 reward 그대로.

**변경**:
- `envs/pe_1v1/env.py`: `critic_priv` obs key 추가 (defender ground truth velocity, ang_vel, attitude)
- `envs/pe_1v1/cfgs.py`: `CRITIC_PRIV_DIM` 정의
- `agents/ppo_runner.py`: `obs_groups = {"actor": ["policy"], "critic": ["policy", "critic_priv"]}`
- Train yaml: `obs_groups` 명시
- CLI flag: `--use_asymmetric_critic` (기본 False, backward compat)

**기대**: critic 가치 정확 → PPO advantage 정확 → policy gradient 안정 → cycle별 진동 ↓.

**시간**: 구현 ~5~8h, 검증 학습 ~6h.

---

### Phase E.2 — Forecaster + Replay Buffer (τ=0, 사전학습 X)

**목적**: forecaster 모듈 학습. τ=0이라 ground truth와 prediction 거의 같음 → sanity check 단계. 본격 효과는 E.3에서.

**변경**:
- 신규 `agents/forecaster/`:
  - `model.py` — Transformer (작은 모델: 2 layer, 64 hidden)
  - `replay_buffer.py` — FIFO buffer (1M transitions ≈ ~500MB GPU)
  - `trainer.py` — supervised loss + Adam optimizer
  - `__init__.py`
- `agents/ppo_runner.py`: forecaster trainer instance + step hook (rollout 후 buffer push + N번 update)
- `scripts/train_attacker.py`: `--use_forecaster` flag, forecaster config path
- `configs/forecaster/default.yaml`: model/training hyperparameters

**Forecaster spec** (논문 §8):
- 입력: `[s^a_{t-K:t}, s^d_{t-τ-K:t-τ}, action_{t-K:t-1}]` (history K=10)
- 출력: `ŝ^d_{t-τ+1:t+H}` (multi-step prediction, H=5)
- Loss: `MSE(ŝ^d, s^d_true)` (env에서 ground truth 제공)
- **PPO loss와 분리** — gradient stop 후 actor obs에 전달 (CLAUDE.md §8)

**Replay Buffer spec** (사용자 명시):
- Capacity: 1M transitions
- 매 PPO rollout 후 new transitions push (FIFO if full)
- 매 forecaster step 256~512 batch random sample
- **Cycle 누적** — cycle 1 ~ K 데이터 모두 buffer에 → catastrophic forgetting 해결

**기대**: forecaster MSE 학습 수렴 (τ=0이라 trivial). buffer 동작 검증.

**시간**: 구현 ~10~15h, 검증 ~6h.

---

### Phase E.3 — τ-delay 점진 도입 (full novelty)

**목적**: 부분 관측 problem 해결. Forecaster가 attacker에게 미래 정보 제공.

**변경**:
- `envs/pe_1v1/env.py`: 
  - τ-delay buffer (sliding window of defender states, 길이 τ_max+H)
  - actor obs에 `s^d_{t-τ}` (delayed) + forecaster output `ŝ^d_{t-τ+1:t+H}` 사용
- `agents/forecaster/trainer.py`: τ를 입력 받아 multi-step prediction
- yaml: `tau_max: 10`, `tau_curriculum: [0, 3, 5, 10]`
- CLI flag: `--tau_max`, `--tau_curriculum`

**Curriculum schedule** (τ 점진 증가):
```
Stage 4 cycles 1~3:   τ=0  (forecaster sanity)
Stage 4 cycles 4~6:   τ=3  (15ms delay)
Stage 4 cycles 7~10:  τ=10 (50ms delay, 논문 spec)
```

**기대**: τ=0에서 vanilla mission rate (v16 baseline) 도달 → τ 증가에 따라 forecaster 보완으로 mission rate 유지. 진정한 paper contribution.

**시간**: 구현 ~10h, 검증 ~10h (τ별 ablation).

---

## 3. Replay Buffer 상세 spec (사용자 핵심 요청)

### 3.1 구조

```python
# agents/forecaster/replay_buffer.py
class ForecasterReplayBuffer:
    """FIFO replay buffer for forecaster supervised learning.
    
    Stores transitions: (history_window, future_window).
    Uniform random sampling. Catastrophic forgetting 방지를 위해 cycle 누적.
    """
    def __init__(self, capacity: int, history_K: int, horizon_H: int, device):
        self.capacity = capacity  # 1M default
        self.history_K = history_K  # 10 (input window)
        self.horizon_H = horizon_H  # 5 (output window)
        # Stored fields:
        self.s_a_history = ...      # (cap, K+1, attacker_state_dim)
        self.s_d_history = ...      # (cap, K+1, defender_state_dim)
        self.action_history = ...   # (cap, K, 4)
        self.s_d_future = ...       # (cap, H, defender_state_dim)  ← target
        self.size = 0
        self.ptr = 0
    
    def add(self, transition_batch):
        # PPO rollout 끝나면 호출. (B, K+H+1) windows 추출 후 push.
        ...
    
    def sample(self, batch_size: int) -> dict:
        # uniform random
        ...
```

### 3.2 Catastrophic Forgetting 방지 메커니즘

```
Cycle 1: PPO rollout (1024 envs × switch_every iters) → buffer.add (~수만 transitions)
Cycle 2: defender 정책 다름 → 새 transitions push, but cycle 1 데이터 유지
...
Cycle K: buffer = mix of cycle 1 ~ K transitions
        forecaster 학습 시 random sample → 모든 cycle 데이터에 robust
```

→ FP의 PPO는 진동했지만, **forecaster는 supervised learning이라 buffer 누적이 자연스럽게 작동**.

### 3.3 Capacity 선택

- `1M transitions`:
  - 1024 envs × ~1000 steps/episode × 50 episodes/cycle ≈ 50M / cycle (overkill)
  - 실제: stride sampling + history window K=10 → ~5M storage / cycle
  - 1M cap이면 cycle 후반에 FIFO 동작 (early cycle 일부 잃음)
  - 또는 10M (~10GB) — 더 안전, 단 메모리 ↑

**추천**: 시작 1M, 부족하면 키움.

### 3.4 학습 schedule

```python
# 매 PPO iter 후
ppo_runner.update_policy()
forecaster_trainer.collect_from_rollout()  # buffer push
for _ in range(forecaster_steps_per_ppo):  # 10~50
    batch = buffer.sample(256)
    forecaster_trainer.train_step(batch)
```

PPO iter당 forecaster N step → forecaster가 PPO보다 빠르게 학습 → forecaster 정확 → attacker obs 신뢰 가능.

---

## 4. 영향 범위 및 신규/수정 파일

### 신규

| 파일 | 줄수 추정 | 용도 |
|---|---:|---|
| `agents/forecaster/__init__.py` | 5 | 모듈 export |
| `agents/forecaster/model.py` | ~150 | Transformer (2 layer, 64 hidden) |
| `agents/forecaster/replay_buffer.py` | ~150 | FIFO buffer + sampling |
| `agents/forecaster/trainer.py` | ~200 | Train loop + collect_from_rollout |
| `tests/test_forecaster.py` | ~150 | 단위 테스트 (model shape, buffer FIFO, trainer step) |
| `configs/forecaster/default.yaml` | ~30 | hyperparameters |
| `docs/plans/phase_e_forecaster_buffer.md` | (이 파일) | plan |

### 수정 (additive only — flag 미사용 시 기존 동작)

| 파일 | 변경 |
|---|---|
| `envs/pe_1v1/env.py` | `critic_priv` obs (E.1), τ-delay buffer (E.3) |
| `envs/pe_1v1/cfgs.py` | `CRITIC_PRIV_DIM` |
| `agents/ppo_runner.py` | `obs_groups`, forecaster trainer integration |
| `scripts/train_attacker.py` | `--use_asymmetric_critic`, `--use_forecaster`, `--tau_max` |
| `configs/train/ppo_attacker_*.yaml` | `obs_groups` |

### 미변경

- defender 학습 측 (forecaster 사용 안 함, defender는 ground truth obs 그대로)
- reward 구조 (v16 = Set A + existential 0.0002 그대로)
- AMS-DRL alternating wrapper (`train_ams_drl.py`) — flag 추가만

---

## 5. 검증 / 테스트 계획

### 5.1 단위 테스트 (각 sub-phase 별 추가)

- **Phase E.1**:
  - `test_critic_priv_obs.py`: env가 critic_priv key 반환 + actor에서 미사용 (deploy mode 검증)
  - `test_obs_groups_routing.py`: rsl-rl에 obs_groups 전달 시 actor/critic 분리 동작
- **Phase E.2**:
  - `test_forecaster_model.py`: Transformer forward shape (B, K, D) → (B, H, D)
  - `test_replay_buffer.py`: FIFO 동작, sample shape, capacity overflow
  - `test_forecaster_trainer.py`: 1 step train loop, MSE loss 감소
- **Phase E.3**:
  - `test_tau_delay_buffer.py`: env의 sliding window 정확성

### 5.2 통합 검증

- **Phase E.1 검증 (Asymmetric Critic 효과)**:
  - v16 reward 그대로 + asymmetric critic ON 학습
  - 비교: cycle별 진동 폭, a_check_max, |Δ| 평균
  - 기준: v16 cycle 16~20 collapse 패턴이 줄어드는지

- **Phase E.2 검증 (Forecaster sanity, τ=0)**:
  - forecaster MSE 시계열 (cycle별 buffer growing 동안 수렴 확인)
  - τ=0이라 prediction = ground truth (trivial) — MSE 거의 0
  - attacker mission rate: forecaster output 사용 vs 미사용 비교 (τ=0이라 차이 없어야)

- **Phase E.3 검증 (τ-delay)**:
  - τ별 ablation: τ ∈ {0, 3, 5, 10}, 각 학습 + a_check_max 비교
  - forecaster MSE: τ 클수록 어려움 → MSE ↑
  - attacker mission rate: τ 증가에도 baseline 수준 유지하는지 (forecaster 보완 효과)

### 5.3 Smoke (cpu, num_envs=16)

- E.1: 5 iter 학습 + critic_priv obs flow 정상
- E.2: 5 iter + buffer 100 transitions push + forecaster 5 train step
- E.3: τ=3 환경에서 5 iter + delay buffer 동작

---

## 6. Rollback Plan

각 sub-phase 독립 flag (`--use_*`):
- `--use_asymmetric_critic`: False면 v16 그대로 동작 (E.1 비활성)
- `--use_forecaster`: False면 forecaster trainer 미실행 (E.2 비활성)
- `--tau_max 0`: τ-delay 없음 (E.3 비활성)

→ flag 모두 False면 v16 reproduce 가능. **CLAUDE.md §1 "기존 동작 보존" 준수**.

각 sub-phase fail 시:
- E.1 fail: asymmetric critic만 비활성, v16 baseline 유지
- E.2 fail: forecaster 비활성, E.1까지만 사용
- E.3 fail: τ=0 유지 (E.1+E.2 효과만)

---

## 7. 시간 / 비용 추정

| sub-phase | 구현 | 학습 | 합계 |
|---|---:|---:|---:|
| E.1 Asymmetric Critic | 5~8h | 6h | 11~14h |
| E.2 Forecaster + Buffer (τ=0) | 10~15h | 6h | 16~21h |
| E.3 τ-delay (τ ablation 4개) | 10h | 24h (4×6h) | 34h |
| **합계** | 25~33h | 36h | **61~69h** |

---

## 8. 열린 질문 (사용자 결정 필요)

1. **Forecaster 모델 크기**: 2 layer × 64 hidden (작음, fast) vs 4 layer × 128 hidden (정확도 ↑, 학습 느림)?
2. **Replay buffer capacity**: 1M (default) vs 10M (안전, 메모리 10GB)?
3. **Forecaster optimizer step ratio**: PPO 1 step당 forecaster N step? (default N=10)
4. **τ curriculum**: stage 4 내에서 점진 증가 vs stage별 다른 τ vs 처음부터 τ=10 일정?
5. **Asymmetric critic priv obs 내용**:
   - 최소: defender ground truth velocity (3D)
   - 표준: + ang_vel (3D) + attitude (3D quat)
   - 최대: + 미래 trajectory (논문 §3.5)
6. **Forecaster + asymmetric critic 동시 활성**: actor obs에 forecaster output, critic obs에 ground truth → 일관성?

---

## 9. TODO 작업 순서 (사용자 승인 후)

### Phase E.1 (Asymmetric Critic)
1. (review) `agents/ppo_runner.py` `obs_groups` 지원 확인 — rsl-rl ≥ 5.0 spec
2. `envs/pe_1v1/env.py` critic_priv obs 추가 (additive)
3. `envs/pe_1v1/cfgs.py` schema
4. `tests/test_critic_priv_obs.py`
5. `scripts/train_attacker.py` `--use_asymmetric_critic` flag
6. Smoke (cpu, 5 iter)
7. v16 reward + Asymmetric Critic 본 학습 (curriculum_v13)
8. 결과 분석: cycle별 진동 vs v16 비교

### Phase E.2 (Forecaster + Buffer)
1. `agents/forecaster/model.py` Transformer
2. `agents/forecaster/replay_buffer.py` FIFO
3. `agents/forecaster/trainer.py` collect + train_step
4. `tests/test_forecaster.py`
5. `agents/ppo_runner.py` forecaster integration hook
6. `scripts/train_attacker.py` `--use_forecaster` flag
7. `configs/forecaster/default.yaml`
8. Smoke (cpu)
9. τ=0 학습 + forecaster MSE 시계열 검증

### Phase E.3 (τ-delay)
1. `envs/pe_1v1/env.py` τ-delay buffer
2. `tests/test_tau_delay_buffer.py`
3. `--tau_max` flag + curriculum
4. τ ∈ {0, 3, 5, 10} ablation
5. 최종 결과 비교 (vs v16 baseline)

---

## 10. 학술적 의미 (논문 contribution 측정)

각 sub-phase 결과로 **clean ablation** 구성:

| 구성 | 의미 |
|---|---|
| Vanilla AMS-DRL (v16) | Baseline (alternating PPO만) |
| + Asymmetric Critic (E.1) | Critic privileged info 효과 |
| + Forecaster (E.2, τ=0) | Forecaster module sanity |
| + τ=3 delay (E.3) | Mild partial observability |
| + τ=10 delay (E.3) | Full paper setting |

각 단계의 a_check_max / Nash 균형 / mission rate 비교 → **paper §4 main results table**.

---

*이 plan을 승인하면 §9 Phase E.1 TODO부터 즉시 시작.*
