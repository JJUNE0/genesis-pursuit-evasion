# Phase 1 — Body-rate PD inner-loop controller gain tuning (PR-I)

작업일: 2026-04-28
선행 PR: PR-H (BodyRateController 인프라)
후행: 단계 3 (`action_mode: "rate"` 전환 후 RL 재학습)

## 변경 동기

PR-H에서 `BodyRateController`를 도입했으나 초기 게인은 placeholder
(`Kp=[0.05]×3`, `Kd=[0.001]×3`). RL 학습 진입 전 step response 측정으로
실제 트래킹 성능을 검증하고 게인을 확정해야 한다.

검증 후 발견: 초기 게인은 **discrete-time 불안정**.
1-step 후방차분 derivative 형태에서 stability 필요조건 `|Kd / I| < 1`이
roll(I_xx = 3.35e-4)에서 위반 (`Kd = 0.001 / 3.35e-4 = 2.99 > 1`).
따라서 plant 모델 위에서 closed-loop이 발산하는 것이 정상.

## 작업 범위

### 1. Step response 측정 인프라

`scripts/tune_rate_pid.py` 신규 (~390 LOC).

- 단일 드론 + plane, `num_envs=16`, dt=0.005s, gravity 활성, 외력 없음.
- 시작: hover state, identity quat, 위치 z=1.0m. 매 trial reset.
- 0.5s zero-action stabilize → 1.0s step input `action[:, axis_idx+1]=+1`
  (= +π rad/s 명령).
- 매 trial별로 fresh `BodyRateController`를 axis-별 Kp/Kd로 구성.
- 측정: 매 step 후 `drone.get_ang()[:, axis_idx]` 기록 (env-mean over 16 envs).

### 2. Metric 정의 (compute_metrics)

평가 윈도우는 step input 시작 후 **첫 0.3s만** (METRIC_WINDOW_S).
이유: 1.0s 전체 평가하면 gravity로 drone이 ~0.45s에 z=0 floor 도달,
지면 충돌 contamination이 ang_vel 측정에 들어와 metric 오염.

- `rise_time`: w_mean이 처음 `0.9·ω_ref`에 도달하는 시간 (없으면 inf)
- `overshoot`: `max(0, peak_signed - ω_ref) / ω_ref × 100%`
- `ess`: 윈도우 끝 0.1s mean의 ω_ref 대비 오차 (%)
- `diverged`: `peak_abs > 3·|ω_ref|` 또는 NaN/Inf

### 3. Selection 로직 (tiered)

Body-rate 트래킹은 `ess`가 가장 중요. 단순 weighted score는 빠른 rise를
가진 ess 높은 combo를 잘못 고른다 (1차 시도에서 실제로 그랬음).
4단 priority로 변경:

1. **Tier 1**: 모든 targets 만족 (rise<0.1s, os<20%, ess<5%, no div)
   → 최저 score (= rise + 0.005·os + 0.003·ess) 선택.
2. **Tier 2**: ess<5% & os<20% → rise 최단.
3. **Tier 3**: ess<10% & os<30% → `rise + 0.005·ess` 최소.
4. **Tier 4**: 비발산 fallback → `ess + 0.5·os + 0.5·rise·100` 최소.

### 4. Grid search 진화

`또는 manual sweep` 사용자 허용을 활용해 3차에 걸쳐 grid 조정.

| 차 | KP_LIST | KD_LIST | 결과 |
|---:|---|---|---|
| 1 (사용자 spec) | `[0.02, 0.05, 0.1, 0.2]` | `[5e-4, 1e-3, 5e-3, 1e-2]` | roll/pitch ess > 70%, yaw best Kp=0.1 Kd=1e-3 (ess=1.78%, rise=0.22s) |
| 2 (Kp 작게/Kd 크게) | `[0.005, 0.01, 0.02, 0.05, 0.1]` | `[1e-3, 5e-3, 1e-2, 2e-2, 5e-2]` | roll best Kp=0.01 Kd=5e-3 (ess=74%) — 여전히 발산 |
| 3 (안정성 영역) | `[0.02, 0.05, 0.1, 0.15, 0.2]` | `[5e-5, 1e-4, 2e-4, 3e-4]` | **전 축 targets 만족 ✓** |

3차 grid는 이론 분석 후 고정:
`Kd_max < min(I_x, I_y, I_z) = 3.35e-4` → Kd ≤ 3e-4.
첫-step ω 변화 `Kp·π·dt/I < 1`을 만족하려면 Kp < I/dt — roll에선 0.067,
pitch 0.124, yaw 0.176. 안전 margin 위해 Kp ≤ 0.2.

### 5. 최종 결과 (60 trials, env-mean over 16 envs)

| axis | Kp | Kd | rise(s) | overshoot | ess | meets all targets |
|---|---:|---:|---:|---:|---:|:---:|
| roll  | 0.05 | 1.0e-4 | 0.010 | 0.00% | 0.02% | ✓ |
| pitch | 0.10 | 5.0e-5 | 0.010 | 0.00% | 0.01% | ✓ |
| yaw   | 0.10 | 1.0e-4 | 0.025 | 0.00% | 0.01% | ✓ |

Targets: rise<0.1s, overshoot<20%, ess<5%, no divergence — **전 축 ✓**.

rise_time = 0.010s = 1 step (200Hz)이라는 것은 실제로 첫 측정에서 이미
0.9·π를 넘었다는 뜻 — 첫 dt 동안 plant `I·ω̇=τ`가 `ω = (Kp·π/I)·dt`만큼
변화. pitch (Kp=0.1, I_y=6.22e-4)에서 `0.1·π·0.005/6.22e-4 = 2.53` rad/s
(즉 0.81·π에 도달). 다음 step에서 0.9·π 통과. 사용자 "더 빨라도 되긴해"
지침에 부합.

## 영향 범위

- `scripts/tune_rate_pid.py` — 신규 (튜닝 entrypoint + grid search + plotting + report writer)
- `configs/envs/rate_controller.yaml` — 신규 (튜닝 결과 standalone artifact)
- `outputs/rate_pid_tuning/` — 신규
  - `step_response.png` (3 axes × 20 combos overlay)
  - `pareto.png` (rise vs overshoot scatter, best ★)
  - `tuning_report.md` (per-axis 표 + 추천 + 미해결)
- `configs/envs/pe_1v1_default.yaml` — `action.rate_controller.kp_rate / kd_rate` 갱신
- `configs/envs/pe_1v1_phase15.yaml` — 동일

코드 (`utils/rate_controller.py`, `envs/base_pe_env.py`) 변경 없음 —
yaml만의 변경이므로 ckpt 호환성에 영향 없음 (단계 3에서 RL 재학습 시
action 의미 바뀌므로 어차피 transfer 불가, fresh init 필요).

## 물리/이론 sanity check

### Discrete-time stability

Plant: `I·ω̇ = τ`, PD with derivative-on-output:
`τ[k] = Kp·(ω_ref - ω[k]) - Kd·(ω[k] - ω[k-1])/dt`

Forward-Euler integration:
`ω[k+1] = ω[k] + (dt/I)·τ[k]`
`     = (1 - a - b)·ω[k] + b·ω[k-1] + a·ω_ref` (a=Kp·dt/I, b=Kd/I)

Characteristic poly: `z² - (1-a-b)·z - b = 0`. roots' product = `-b`.
양 root이 `|z| < 1`이려면 `|product| = b < 1` 필수, 즉 **`Kd < I`**.

Nova 검증:
- I_xx = 3.35e-4 → Kd ≤ 3.35e-4 (선택 1e-4: ✓)
- I_yy = 6.22e-4 → Kd ≤ 6.22e-4 (선택 5e-5: ✓)
- I_zz = 8.81e-4 → Kd ≤ 8.81e-4 (선택 1e-4: ✓)

### Continuous-time response

CL transfer (Plant·C / (1+Plant·C), derivative on output):
`(I + Kd)·ω̇ + Kp·ω = Kp·ω_ref` → 1차계, time constant `τ = (I+Kd)/Kp`.

Steady-state: `G_cl(0) = Kp/Kp = 1` → `ess = 0` 이론치. 실측 ess가
0.01~0.02%로 미세 잔류는 motor saturation 비선형성과 sampling
discretization에서 비롯.

### Motor saturation 분석

Step input 직후 첫 commanded torque `Kp·π`:
- roll  Kp=0.05: 0.157 N·m
- pitch Kp=0.10: 0.314 N·m
- yaw   Kp=0.10: 0.314 N·m

physical max torque (`f` = max_thrust_per_motor 4.4N 활용):
- roll/pitch: 4·L·(f - f_hover) ≈ 4·0.10607·2.435 ≈ 1.033 N·m
- yaw: 4·c·(f - f_hover) ≈ 4·0.01239·2.435 ≈ 0.121 N·m (가장 작음)

yaw에서만 saturation 가능하지만 yaw torque 명령 0.314 > 0.121이라
saturation. 그럼에도 tracking은 잘 되는 이유: yaw는 reactive torque로
구동되므로 saturation 후에도 정상 동작 (4 motors 비대칭 활용).

## Rollback plan

- yaml만의 변경이므로 `git revert <commit>`로 단일 커밋 되돌림.
- ckpt 호환성 영향 없음 (rpm-mode default 유지).
- `outputs/rate_pid_tuning/` 산출물은 보존 (참고 자료, 학습엔 영향 없음).

## 테스트 계획

1. `pytest tests/` — **40/40 그린**. 회귀 0.
2. `action_mode='rate'` smoke test (4 env × 20 step zero-action)
   → 호버 안정 (`|ang_vel|≈1e-7`, finite). ✓
3. **단계 3 진입 시 추가** (사용자 GPU 액션):
   - `pe_1v1_default.yaml` `action_mode: "rpm"` → `"rate"` 전환.
   - stationary defender 대상 attacker 5k iter (logs_v2_rate/, fresh init).
   - 기준: `|ω|` late, mean_episode_length, attacker_win_rate를 RPM baseline과 비교.
   - episode 5초 (= max_steps 1000) 동안 발산 없음 / 게인 추가 조정 필요 여부.

## 미해결 / 향후 검토

### 1. Metric window 0.3s의 한계

현재 step input test는 0.3s까지만 평가하는데, 실제 RL episode는 5초.
정책이 다양한 step 명령을 분산 출력하지만 0.3s 윈도우는 long-horizon
dynamics를 검증하지 못함. 단계 3 학습 진입 시 실제 episode 동안
발산하지 않는지 확인 필요.

발산 시 candidate 조정:
- Kd 추가 감소 (현재 1e-4 → 5e-5 등)
- max_body_rate 감소 (현재 ±π → ±π/2)
- step input 크기를 학습 시 정책 출력 분포에 맞게 재테스트

### 2. Cascade controller 검토 (Phase 7 sim-to-real 진입 전)

현재 single-loop body-rate PD. 사용자 spec "no cascade"에 따른 결정.
실기 적용 (Phase 7) 시 attitude (quaternion) → body-rate outer loop가
강건성에 도움이 될 가능성. 단순 RL이 attitude 자체를 자체 학습하므로
이번 phase 범위 외.

### 3. Derivative low-pass filter (선택, 5 LOC)

backward-difference derivative는 sensor noise / RL action spike에 민감.
discrete instability를 회피했더라도 학습 중 행동 noise spike에
oscillation 일어날 가능성. 1st-order LPF on ω̇ 추가는 5줄 변경:

```python
ang_acc_raw = (ang_vel - self.last_ang_vel) / self.dt
self.ang_acc_filt = alpha * ang_acc_raw + (1 - alpha) * self.ang_acc_filt
torque = self.kp * rate_err - self.kd * self.ang_acc_filt
```

`alpha = dt / (tau_filter + dt)`. 학습 중 instability 관찰 시 도입.

### 4. Display rounding bug (cosmetic)

`scripts/tune_rate_pid.py` console output에서 `f"{kd:.4f}"`가
`5e-5`와 `1e-4`를 모두 `0.0001`로 출력. yaml/report에는 정확히
기록되므로 단순 디스플레이 문제. 단계 3 진입 전 fix:
`f"{kd:.5f}"` 또는 scientific notation `f"{kd:.2e}"`.

## 비고

이번 PR은 **인프라 + 튜닝값**만 갱신. RL 학습 정책은 변경 없으며,
`action_mode` 기본값 `"rpm"` 유지로 기존 학습 흐름 무영향. 단계 3에서
`"rate"` 전환 시 비로소 본 튜닝값이 사용됨 (사용자 GPU 액션 + fresh init
필요).

게인은 결정적으로 picked한 값이지만 단계 3 학습 결과에 따라 미세 조정
가능성 열어둔 상태로 yaml 주석에 출처(`PR-I 2026-04-28 grid search`)와
근거(`outputs/rate_pid_tuning/tuning_report.md`)를 기록.
