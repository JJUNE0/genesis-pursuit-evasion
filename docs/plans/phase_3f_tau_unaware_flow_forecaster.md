# Phase 3f — τ-Unaware Flow Matching Forecaster

> **상태**: 계획 (2026-05-14)
> **선행 의존**: Phase 3a (τ-delay env), Phase 3b (data), Phase 3c (τ-aware forecaster), `data/forecaster_data_ams_v9_c{7,8,14,17,20}/`, `logs/forecaster/ams_v9_multi_v1/`
> **CLAUDE.md §1**: 기존 forecaster 파일 (`agents/forecaster/{model.py, dataset.py, rectified_flow.py, trainer.py}`) **수정 X**. 새 모듈 `agents/forecaster/tau_unaware/` 디렉토리에 작성.

---

## 1. Motivation — 변경 동기

### 1.1 현재 ams_v9_multi forecaster 의 한계

기존 `ForecasterModel` (Phase 3c) 은 **τ 를 explicit input** 으로 받음:

```python
forward(x_t, t, context, tau)
                          ↑↑↑ τ 를 sinusoidal embed → encoder context 에 concat
```

→ 학습 시 매 batch random τ ∈ [1, 50] sample (Pattern 2). Inference 시 deployment 가 정확한 τ 알아야 함. **현실 setting (cellular relay + crypto + air interface) 에서는 τ 를 정확히 모름** — packet jitter, retransmit, queueing 등으로 instantaneous τ 가 stochastic.

### 1.2 새 contribution — τ-unaware forecaster

**Forecaster 가 τ 모르고 prediction**:
- Input: fixed-length history (K=50 step attacker self) + 마지막 observed defender state (1 step, staleness 모름)
- Output: 현재 시점 defender state estimate
- → 어떤 τ 에서도 동일 모델로 작동, deployment 시 τ measurement 불필요

이게 paper main contribution 강화 — "asymmetric goal-delay POMDP 의 sufficient statistic 을 τ 명시 없이 학습 가능"

---

## 2. Architecture Spec

### 2.1 Encoder

```
Input:
  s^a_history: (B, K, 18)        # K=50 step attacker self (real-time, no delay)
  s^d_last:    (B, 1, 6)          # 가장 최근에 observed defender (staleness 모름)
  a^a_history: (B, K, 4)          # attacker action history

Per-modality projection → d_model = 256
Concat: (B, 2K + 1, d_model) — NO τ embedding, NO timestamp
Self-attention transformer encoder (4 layers, pre-norm GELU)
→ context_vector c_t: (B, 2K+1, d_model)
```

**Critical**: τ 명시 input 0개. Position encoding 만 sequence 위치 (1..K) 표시 — 이는 "K step 전부터 1 step 전까지 attacker history" 의미만.

### 2.2 Velocity Field (Flow Matching head)

```
v_θ(x, t_flow; c_t):
  x: (B, 6)                  # current denoising state of target s^d_t
  t_flow: (B, 1)             # flow matching time ∈ [0, 1]
  c_t: encoder context

Decoder (cross-attention):
  trajectory_proj(x) + sinusoidal(t_flow)        # (B, 1, d_model)
  → cross-attend to c_t
  → 4 decoder layers (self + cross attention)
  → output_head: (B, 1, d_model) → (B, 6)
```

**Output**: 6D defender state (rel_pos + rel_vel) at current time t, in t-attacker body frame + scaled.

(기존 Phase 3c forecaster 와 차이: target sequence length 1 (single step) vs τ — τ 가 없으니 multi-step prediction 도 의미 없음. 단일 "현재 추정" 만.)

### 2.3 Hyperparameters (default)

| | 값 |
|---|---|
| K (history length) | **50** (max τ 와 같음) |
| d_model | 256 |
| n_heads | 8 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| FFN ratio | 4 |
| Dropout | 0.1 |

---

## 3. Data Pipeline — 기존 데이터 재사용

기존 `data/forecaster_data_ams_v9_c{7,8,14,17,20}/` (5 cycle × 4000 episode) 그대로 사용.
- 추가 데이터 수집 0
- 새 데이터셋 class 만 작성 (τ field strip + last-observed defender single step)

### 3.1 Sample 정의

매 학습 sample:
```python
1. Sample τ ~ U[1, 50]                   # τ varies but model never sees it
2. Sample t ∈ [τ, L-1]
3. Extract:
   Input:
     s^a_window: episode['s_a_world'][t-K : t]    # (K, 18) world raw, K=50
                  (NOTE: K과 τ 무관. 항상 K step.)
     s^d_last:   episode['s_d_world'][t-τ : t-τ+1]  # (1, 18) world raw
     a_a_window: episode['a_a'][t-K : t]           # (K, 4)
   Target:
     s^d_now:    episode['s_d_world'][t : t+1]     # (1, 18) world raw
4. Frame transform → t-attacker body frame (build_sample 패턴 재사용)
5. Output: 6D defender (rel_pos+rel_vel)
```

**중요**: `s^a_window` 는 K=50 step 항상 같은 길이. τ 가 변해도 window 같음. **τ leak 차단**.

### 3.2 Critical — τ leak 차단 검증

`s^d_last` 를 받는 시점이 τ step 전이라는 정보는:
- Input feature 에 없음 ✓
- Position encoding 에 없음 ✓ (s^d_last 는 항상 별도 token)
- Sequence length 에 없음 ✓ (K + 1 fixed)
- Encoder mask 에 없음 ✓ (전부 attendable)

→ Model 은 "s^d_last 가 t-1 시점인지 t-50 시점인지 알 방법 없음".

### 3.3 Frame Convention (기존 dataset 와 동일)

`build_sample()` (기존 `agents/forecaster/dataset.py`) 의 frame transform 그대로 재사용 — **t-attacker body frame, scaled**. 단 target 이 6D × 1 (= 단일 step) 으로 reduce.

---

## 4. Training — Conditional Flow Matching Loss

### 4.1 Loss

기존 RF (rectified flow) 와 사실상 동등 — straight-line probability path:
```python
def cfm_loss(model, batch):
    target = batch['target']                       # (B, 6) — single step defender
    B = target.shape[0]
    
    noise = torch.randn_like(target)               # (B, 6)
    t_flow = torch.rand(B, 1, device=device)       # (B, 1) ∈ [0, 1]
    
    x_t = (1 - t_flow) * noise + t_flow * target   # (B, 6)
    v_target = target - noise                      # (B, 6)
    
    context = {
        'attacker_history': batch['attacker_history'],  # (B, K, 18)
        'defender_last':    batch['defender_last'],     # (B, 1, 6)
        'action_history':   batch['action_history'],    # (B, K, 4)
    }
    v_pred = model(x_t=x_t, t=t_flow, context=context)  # NO tau arg
    return F.mse_loss(v_pred, v_target)
```

차이 vs 기존 `rf_loss`:
- target shape `(B, τ, 6)` → `(B, 6)` (single step)
- model.forward 에 `tau` arg 제거

### 4.2 Training Schedule

| 항목 | 값 (기존 setup 동일) |
|---|---|
| batch_size | 256 |
| num_training_steps | 100K |
| LR | 1e-4 → 1e-5 cosine, warmup 1K |
| Optimizer | AdamW (WD 1e-4) |
| bf16 mixed precision | yes (no GradScaler) |
| Val every | 5K |
| Save every | 10K |

---

## 5. Inference Protocol (Deployment)

### 5.1 단일 sample (deployment 시)

```python
@torch.no_grad()
def deploy_predict(model, history, num_euler_steps=10):
    """
    history: dict with attacker_history, defender_last, action_history
    Returns: (B, 6) — current defender estimate
    """
    B = history['attacker_history'].shape[0]
    x = torch.randn(B, 6, device=device)
    dt = 1.0 / num_euler_steps
    for step in range(num_euler_steps):
        t_flow_val = step / num_euler_steps
        t_flow = torch.full((B, 1), t_flow_val, device=device)
        v = model(x_t=x, t=t_flow, context=history)
        x = x + v * dt
    return x  # (B, 6)
```

**Default**: Euler 10 step (paper inference 비용 낮춤). Train 시 20 step val 도 OK.

### 5.2 Multi-sample averaging (선택)

만약 단일 sample variance 너무 큼 → N (e.g. 4) sample 의 median 으로 stabilize. Wall-clock cost N×.

**결정**: Default 단일 sample (1 forward pass cost, deployment realism). Multi-sample 은 ablation.

---

## 6. Comparison Set — Paper Main Table

| # | Forecaster | τ knowledge | Architecture | Note |
|---|---|---|---|---|
| 1 | No FC | — | — | Lower bound (raw delayed obs) |
| 2 | Naive const-vel | implicit | None (math) | Trivial baseline |
| 3 | τ-aware deterministic | knows τ | MLP regression head | Existing reference |
| 4 | τ-aware flow (Phase 3c) | knows τ | Flow matching | `ams_v9_multi_v1` 그대로 사용 |
| 5 | **τ-unaware flow (3f, main)** | **NO τ** | Flow matching | **Main contribution** |
| 6 | τ-unaware deterministic | NO τ | MLP regression head | Flow matching 의 필요성 ablation |

→ **6-way comparison**. 각각 4 pair (a5/a6/a7/a8 × d_{i+1}) × τ scenarios → ablation table.

### 6.1 어느 baseline 추가 학습 필요?

| baseline | 학습 필요? | 시간 |
|---|---|---|
| 1 No FC | ✗ (env 만 사용) | 0 |
| 2 Naive const-vel | ✗ (수식만) | 0 |
| 3 τ-aware deterministic | **신규 학습** | ~2시간 |
| 4 τ-aware flow | ✗ (기존 ams_v9_multi_v1) | 0 |
| 5 τ-unaware flow (main) | **신규 학습** | ~2시간 |
| 6 τ-unaware deterministic | **신규 학습** | ~2시간 |

→ **신규 학습 3종 × 2시간 = ~6시간**. 1일 안 됨.

---

## 7. Devil's Advocate — 위험 3가지

### 위험 1 — τ-unaware 가 τ-aware 보다 너무 떨어짐
- **Symptom**: τ-unaware val MSE 가 τ-aware 의 3× 이상
- **Mitigation**: Flow matching 의 stochastic 성격이 dependable point estimate 보다 actor 한테 좋을 수 있음 (uncertainty 자연 표현). Eval 로 검증.
- **Fallback**: 만약 τ-unaware 가 너무 약하면 baseline 으로만 두고 paper main 은 τ-aware

### 위험 2 — Plug-in pair variance 그대로 (cherry-picking)
- **Symptom**: a5/a7 잘 됨, a8/a9 안 됨 의 패턴 반복
- **Mitigation**: Honest reporting. "Defender strength gating" finding 자체를 paper 발견으로 framing. Limitation 명시.

### 위험 3 — "Flow matching 진짜 필요?" 공격
- **Mitigation**: Comparison row #6 (τ-unaware deterministic) 추가 → flow matching 의 필요성 직접 ablation.
- 만약 τ-unaware deterministic ≈ τ-unaware flow → flow contribution 약함. 정직 인정.
- 만약 flow > deterministic → "stochastic prediction 이 multi-modal future 처리에 우월" claim.

---

## 8. 영향 범위 (CLAUDE.md §1)

### 8.1 변경 X (read-only)
- `envs/` 전체
- `agents/forecaster/{model.py, dataset.py, rectified_flow.py, trainer.py}` (Phase 3c τ-aware forecaster)
- `data/forecaster_data_ams_v9_c{7,8,14,17,20}/` (기존 데이터 재사용)
- `logs/forecaster/ams_v9_multi_v1/` (기존 ckpt)
- `agents/forecaster_legacy/` (이미 deprecated)

### 8.2 신규 작성
- `agents/forecaster/tau_unaware/__init__.py`
- `agents/forecaster/tau_unaware/dataset.py` — τ field strip + single-step target
- `agents/forecaster/tau_unaware/model.py` — encoder-decoder without τ input
- `agents/forecaster/tau_unaware/flow.py` — CFM loss + Euler sampling (no tau)
- `agents/forecaster/tau_unaware/trainer.py` — train loop helpers
- `scripts/train_tau_unaware_forecaster.py`
- `configs/forecaster/tau_unaware_default.yaml`
- `tests/test_tau_unaware_forecaster.py`

### 8.3 코드 변경 영향 (Phase 3c τ-aware 작동 그대로)
- 기존 `train_forecaster.py`, `eval_forecaster_compare.py`, `plot_forecaster_*.py` 모두 그대로 작동
- 기존 ams_v9_multi_v1/best.pt 그대로 사용 가능 (τ-aware reference)

### 8.4 Rollback
- 신규 디렉토리 통째 삭제 → Phase 3c 시점 복귀
- Git revert 가능 (단일 PR)

---

## 9. 작업 순서 (1-1.5일)

| # | Step | 시간 |
|---|---|---|
| 1 | `agents/forecaster/tau_unaware/` 디렉토리 + 4 module 작성 | 2-3시간 |
|  | - dataset.py: τ-strip collator, K-fixed history slice | |
|  | - model.py: encoder-decoder, NO τ embedding | |
|  | - flow.py: CFM loss, Euler sampling | |
|  | - trainer.py: train loop, validate (single-step MSE) | |
| 2 | `scripts/train_tau_unaware_forecaster.py` + yaml | 1시간 |
| 3 | `tests/test_tau_unaware_forecaster.py` (5 tests) | 1시간 |
|  | - shape consistency | |
|  | - τ leak 검증 (input 에 τ 정보 없는지 unit test) | |
|  | - frame transform parity | |
|  | - CFM loss decreases (overfit smoke) | |
|  | - sampling shape | |
| 4 | Smoke training (5K step) + sanity check | 30분 |
| 5 | Full training (100K step) | 2시간 |
| 6 | Eval comparison (6-way table on a5/a7) | 2-3시간 |
| | - τ-unaware deterministic baseline 학습 (#6) | (이 step 안에 포함, ~2시간) |
| | - Naive const-vel implementation (수식만) | 0.5시간 |
| | - Plug-in eval matrix | 0.5시간 |

**Total**: 8-12시간 ≈ **1-1.5일**

---

## 10. Test Plan (CLAUDE.md §12)

### 10.1 Unit tests (5개, `tests/test_tau_unaware_forecaster.py`)
1. **τ-strip collator shape**: `(attacker_history (B, 50, 18), defender_last (B, 1, 6), action_history (B, 50, 4), target (B, 6))`
2. **τ leak unit test**: 같은 (s^a, s^d_last, a^a) 입력에 다른 τ 학습 데이터 sample → input tensor byte-equal 검증
3. **Frame transform parity**: 기존 `build_sample` 의 t-attacker body frame 변환과 일치 (단 target 만 다른 shape)
4. **CFM loss overfit**: 단일 (context, target) 복제 → 500 step 학습 → loss < initial × 0.5
5. **Euler sampling shape**: `(B, 6)` 출력, finite

### 10.2 Integration sanity
- Smoke training (5K step) 후 val MSE 가 τ-aware ams_v9_multi_v1 대비 ≤ 3× 이내인지 확인 (위험 1 의 mitigation gate)

---

## 11. Acceptance Criteria

- [ ] 5 unit tests pass
- [ ] Smoke training 5K step 정상 수렴 (val MSE < 0.05)
- [ ] Full training 100K step 정상 수렴 + best ckpt 저장
- [ ] τ-unaware deterministic baseline 학습 + ckpt 저장
- [ ] 6-way comparison eval (a5/a7 pair 우선) plot + 수치 표
- [ ] 기존 Phase 3c τ-aware forecaster 동작 회귀 0 (기존 7 tests pass 유지)
- [ ] 기존 env / data 코드 수정 0

---

## 12. Out of Scope

- From-scratch attacker variant 학습 (Phase 3e — 본 plan 후 진행)
- ams_drl 재학습 (현 ams_v9 그대로)
- BPQL/DFBT 외부 baseline (structural argument 로 처리, paper Appendix)
- τ-unaware multi-step prediction (single step 만; future work)
- Reflow (CFM 1-pass 만)

---

## 13. Paper Implication

### 새 contribution 정리
> "We propose a **τ-unaware flow-matching forecaster** that estimates the current defender state from a fixed-length history without observing the actual delay τ. This matches deployment realism where instantaneous τ is unknown due to packet jitter, retransmit, and network queueing. Our 6-way ablation shows that (i) flow-matching is essential vs deterministic regression for handling multi-modal defender futures, and (ii) τ-unaware variant achieves competitive performance with τ-aware reference, making the system deployable without delay measurement infrastructure."

### Headline claim 강화
- **Before**: "Forecaster recovers 60-86% of delay penalty (plug-in)"
- **After**: "**τ-unaware** forecaster recovers comparable performance to τ-aware reference, **without delay measurement at deployment**"

→ Paper contribution 한 단계 ↑.
