# PR-O — Squashed Gaussian Policy (SAC-style tanh squashing for PPO)

작업일: 2026-04-30
선행: PR-N (wrench saturation), AMS-S2 P0/Phase A/B/C 처방 사이클 (logs/pe1v1_v0/ams_s2_v[1-4]).

## 변경 동기 — Policy Distribution Collapse 발견

ams_s2_v4 (Phase C wrench saturation) 학습 결과 **c14 |Δ|=0.085로 Nash 균형
근접**. 그러나 viewer (`scripts/eval.py --show_viewer`)에서 사용자 보고
**"프로펠러 회전 안 보임"**.

`scripts/test/diag_rpm_trace.py` 진단 — c14 attacker_v14 + defender_v14 head-
to-head, 400 step rollout:

| metric | deterministic | stochastic |
|---|---:|---:|
| `|action|_max per step` saturation_frac | **1.000** | **1.000** |
| RPM std median | 4520 | 4411 |
| sat_max (RPM≈cap) | 0.13 | 0.22 |

**stochastic도 100% saturate** — sample noise로도 boundary 영역 못 들어감.

actor 직접 inspection (`actor(td)` raw forward 결과):
- raw mean = `[24.84, 15.17, 31.85, -18.42]`  ← boundary [-1, 1]의 **32× 밖**
- stddev = `[14.21, 16.33, 14.57, 9.02]`  ← boundary scale의 14×

`mean ± 2·std` 95% CI 분석:
- T_norm: [-3.6, 53.3] — 일부만 boundary 통과
- wy: [2.7, 61.0] — 거의 안 통과
- wz: [-36.5, -0.4] — 거의 안 통과

즉 **정책 분포 mean이 [-1, 1] 영역 밖으로 비가역 push**. 4 차원 중 2개는
sample도 절대 boundary 영역 안 들어옴.

### 메커니즘 — PPO + clamp의 알려진 pathology

`PretrainedAttacker/Defender.step`이 `action.clamp(-1, 1)` 강제 → environment
응답은 `|raw|>1` 영역에서 동일. → reward gradient가 mean을 boundary 밖으로
자유롭게 push (no penalty for going further out). → 학습 진행할수록 mean
단조 발산. → eval 시 `μ → clamp = ±1` always saturate.

Phase A (R_opp_crash=0), Phase B (action_smooth + omega clamp), Phase C
(wrench saturation handler) 모두 **environment-domain 처방** — distribution-
domain (mean/std parameters) 못 건드림. 7번 try-and-error 누적 후 발견.

## 처방 — Squashed Gaussian (SAC 표준)

```
actor MLP forward → μ_raw ∈ ℝ⁴ (unconstrained)
                 → action = tanh(μ_raw + σ·ε)        # stochastic (학습)
                 → action = tanh(μ_raw)              # deterministic (eval)
log_prob(a) = log p(u) − Σ log(1 − tanh²(u))         # u = atanh(a)
```

**구조적 강제**: μ_raw가 어떤 값이든 출력은 `tanh(·) ∈ (-1, 1)`.
- mean이 boundary 밖으로 못 빠져나감
- reward gradient는 항상 boundary 안에서 작용
- `clamp(-1, 1)`은 사실상 no-op (수치 safety)

## 변경 사항

### 1. `agents/squashed_gaussian.py` 신규

`SquashedGaussianDistribution(GaussianDistribution)` subclass:
- `sample()`: parent sample → `tanh()`
- `deterministic_output(mlp_output)`: `tanh(mlp_output)`
- `log_prob(outputs)`: `atanh` invert (EPS clamp) + tanh Jacobian (stable form `2(log 2 - u - softplus(-2u))`)
- `as_deterministic_output_module()`: nn.Module wrapper for ONNX/JIT export
- `mean` / `std` / `entropy` / `params` / `kl_divergence`: parent 그대로 (pre-squash 영역, PPO KL 계산용)

CLAUDE.md §1 — 기존 rsl_rl 코드 무수정. 새 클래스만 추가.

### 2. `configs/train/ppo_attacker_squash.yaml` 신규
### 3. `configs/train/ppo_defender_squash.yaml` 신규

`ppo_attacker.yaml` / `ppo_defender.yaml` 복제 + 단일 변경:
```yaml
actor:
  distribution_cfg:
    class_name: agents.squashed_gaussian.SquashedGaussianDistribution
```

rsl_rl 5.x의 `class_name` dispatch가 dotted path를 module import로 처리.
Smoke test (5 iter)로 검증 완료.

### 4. `scripts/train_ams_drl.py` 수정 (additive)

`--attacker_train_yaml`, `--defender_train_yaml` CLI 인자 추가 (default
기존 yaml 유지). cmd_a/cmd_b에 `--train_yaml` 자식으로 forward. PR-N 패턴
재사용.

## 검증 결과

`SquashedGaussianDistribution` unit test:
- ✓ deterministic output ∈ [-1, 1]
- ✓ sample ∈ [-1, 1]
- ✓ log_prob(sample / boundary / ±1 saturated) 모두 finite (EPS clamp 작동)
- ✓ KL divergence (pre-squash) finite

5-iter smoke train (`train_defender.py --train_yaml ...squash.yaml`):
- ✓ rsl_rl OnPolicyRunner가 dotted-path class_name 정상 dispatch
- ✓ PPO 학습 정상 진행 (5 iter)
- ✓ "Mean action std: 1.00" — init_std=1.0 그대로
- ✓ "Mean entropy loss: 5.67" — entropy 정상 (squashing으로도 entropy OK)
- ✓ actor ckpt 저장 + alias 갱신

## 영향 범위

| 영역 | 영향 |
|---|---|
| obs / action shape | 무변경 |
| **체크포인트 호환성** | **BROKEN** — μ_raw scale 의미 다름. fresh init 필수 |
| 기존 yaml (Phase 1.0/1.5/A/B/C) | 무영향 — distribution_cfg.class_name 그대로면 동일 동작 |
| Phase 1.0/1.5/A/B/C 학습 ckpt 보존 | OK — `logs/pe1v1_v1_squash/`로 분리 |
| `PretrainedAttacker/Defender.step` | clamp는 squashed action 영역에서 사실상 no-op (안전 그대로) |

## 학습 plan (사용자 승인 후)

### Phase 1.0 (squashed defender, stationary attacker 상대)

```bash
.venv/bin/python scripts/train_defender.py \
    --train_yaml configs/train/ppo_defender_squash.yaml \
    --env_yaml configs/envs/pe_1v1_default.yaml \
    --num_envs 1024 --max_iterations 5000 --seed 0 --logger wandb \
    --logs_root logs/pe1v1_v1_squash/defender_v0
```
약 30-40분 GPU. 통과 기준: stationary attacker 대상 winrate > 0.9
(이전 baseline과 동일).

### Phase 1.5 (squashed attacker, pretrained squashed defender 상대)

```bash
.venv/bin/python scripts/train_attacker.py \
    --train_yaml configs/train/ppo_attacker_squash.yaml \
    --env_yaml configs/envs/pe_1v1_phase15.yaml \
    --defender pretrained \
    --defender_ckpt logs/pe1v1_v1_squash/defender_v0.pt \
    --num_envs 1024 --max_iterations 10000 --seed 0 --logger wandb \
    --logs_root logs/pe1v1_v1_squash/attacker_v0
```
약 1-1.5시간 GPU. 통과 기준: pretrained defender_v0 대상 winrate > 0.5.

### ams_s2_v5 (Phase C yaml + squashed actors)

```bash
.venv/bin/python scripts/train_ams_drl.py \
    --attacker_ckpt logs/pe1v1_v1_squash/attacker_v0/<run>/attacker_v0.pt \
    --defender_ckpt logs/pe1v1_v1_squash/defender_v0/<run>/defender_v0.pt \
    --attacker_resume logs/pe1v1_v1_squash/attacker_v0/<run>/model_<n>.pt \
    --defender_resume logs/pe1v1_v1_squash/defender_v0/<run>/model_<n>.pt \
    --env_yaml configs/envs/pe_1v1_phase_c.yaml \
    --attacker_train_yaml configs/train/ppo_attacker_squash.yaml \
    --defender_train_yaml configs/train/ppo_defender_squash.yaml \
    --num_envs 1024 --switch_every 50 --max_cycles 30 \
    --n_h2h_episodes 400 --convergence_window 3 --convergence_eps 0.10 \
    --draw_max 0.2 --seed 0 --logger wandb \
    --logs_root logs/pe1v1_v0/ams_s2_v5
```

## 통과 기준

`scripts/test/diag_rpm_trace.py`로 검증 (Phase 1.5 squashed ckpt 또는
ams_s2_v5 c5 ckpt):
- **deterministic eval `|action|_max` saturation_frac < 0.5** (현재 1.000)
- raw mean (actor(td) 결과)가 [-1, 1] 영역에 머무름 (tanh로 강제)
- |ω| mean < π = 3.14 (controller 추종 영역 내)

## Devil's Advocate

- **squashing이 학습 신호 일그러뜨림 위험**: SAC에서 잘 작동하나 PPO에서는
  덜 흔한 패턴. 학습 안정성 monitoring 필요. 만약 mean_std collapse → 0
  또는 발산 → init_std 또는 entropy_coef 조정.
- **KL pre-squash vs post-squash**: KL을 pre-squash z 영역에서 계산. 의미
  있으나 actual action distribution KL과 다름. PPO desired_kl=0.01 그대로
  유지 — 학습 결과 보고 조정.
- **재학습 비용 ~3시간 GPU**: Phase 1.0 + Phase 1.5 + ams_s2_v5 5 cycle smoke.
  실패 시 fallback: rsl_rl entropy_coef 강화 + reward L2 on raw mean (D2b).

## Follow-up (별도 PR)

- **PR-J/K 회귀 테스트** with squashed actor: PR-I PID 게인이 squashed action
  영역에서도 정상 PD 추종하는지 검증.
- **mean_std monitoring**: tensorboard에 log_std 시계열 추가, std collapse
  alert.
- **Sim-to-real 검증** (Phase 7 진입 시): 실제 모터의 RPM rate limit과 squashed
  policy의 action_diff 분포 비교.
