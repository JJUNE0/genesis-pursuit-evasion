# PR-J — BodyRateController hover stability 검증

작업일: 2026-04-28
선행: PR-I (PD gain tuning, 단계 2)
후행: 단계 3 (`action_mode: "rate"` RL 학습)

## 발견 의문

PR-I 튜닝 결과 `rise<10ms / overshoot=0% / ess<0.05%`로 너무 깨끗했고,
`scripts/visualize_rate_controller.py`에서 step phase 동안 drone z가 떨어지는
모습이 관찰됨. 두 신호가 모순되어 보였다:

> "PR-I 튜닝 자체가 자유낙하 중 measurement였다면, RL 학습 진입 시 발산 위험"

확인이 필요했다. 가설 4종:
- A: Mixer 비대칭 분배 (hover thrust 균등 X)
- B: kf 단위 불일치 (RPM ↔ rad/s)
- C: PD 첫 step transient
- D: BodyRateController 자체 버그 (mixer 곱셈/clamp/sign)

## Phase 1 — 진단 instrumentation

`scripts/diagnose_hover.py` 신규 (PR-J 일회성). 두 직교 테스트:

**Test A — BodyRateController + action=[0,0,0,0] for 5s.**
PD가 뽑은 RPM을 모터에 전달하는 정상 경로.

**Test B — Direct hover_rpm (BodyRateController bypass) for 5s.**
controller 우회. Genesis sim 자체 hover stability 베이스라인.

추가로 mixer matrix sanity check (M, M_inv, hover wrench → per-motor thrust).

### 결과

#### Mixer

```
M shape: (4, 4)
M = [[ 1,        1,        1,        1      ],
     [-0.10607,  0.10607,  0.10607, -0.10607],
     [-0.10607,  0.10607, -0.10607,  0.10607],
     [-0.01239, -0.01239,  0.01239,  0.01239]]

hover_wrench = [m·g, 0, 0, 0] = [7.860, 0, 0, 0]
thrusts_at_hover = [1.965, 1.965, 1.965, 1.965]   # 균등 ✓
max_dev_from_uniform = 0
hover_rpm (mixer)   = 8013.70
hover_rpm (formula) = 8013.70   # 동일 ✓
```

→ 시나리오 A (mixer 비대칭) 부정.
→ 시나리오 B (단위 불일치) 부정 — 8013.7 RPM이 single_drone_hover.py 기준치와 일치.

#### Test A — BodyRateController hover

```
k=   0  z=1.0000m  ang_vel=[0,0,0]  rpm=[8013.7]×4
k=   1  z=1.0000m  ang_vel=[0,0,0]  rpm=[8013.7]×4
k=   2  z=1.0000m  ang_vel=[0,0,0]  rpm=[8013.7]×4
...
k= 999  z=1.0000m  ang_vel=[0,0,0]  rpm=[8013.7]×4    (5s, 1000 step)

z_drop over 5s   = 0.0000m
max |ang_vel|    = 0.0000 rad/s
z_drop @ 0.3s    = 0.0000m
z_drop @ 1.0s    = 0.0000m
```

→ **BodyRateController hover 5초 완벽 안정.**
→ 시나리오 C (첫 step transient) 부정.
→ 시나리오 D (controller bug) 부정.

#### Test B — Direct hover_rpm

```
z_drop over 5s = 0.0000m   # Test A와 동일
```

→ Genesis sim baseline도 안정. 두 경로 일치.

### Phase 2 결론 — 시나리오 모두 ❌

**BodyRateController는 hover 명령에서 정확히 hover_rpm을 4 모터 모두 균등하게
출력하며, drone은 z=1.0 영구 유지한다.** 어떤 가설도 성립 안 함.

그럼 visualize에서 본 "fall"은 무엇이었는가?

## 진짜 원인 — 정상 물리

`visualize_rate_controller.py` step phase에 `action=[0, ω_x_ref, 0, 0]` 1초간
sustained 명령:
- t=0~0.5s: stabilize phase (action=0) — 위 검증대로 z=1.0 유지
- t=0.5s 이후: ω_x = π rad/s 트래킹 → drone roll 시작
- 0.3s 후 roll 각도 ≈ π·0.29 = 52° (PR-I metric window 끝)
- 52° tilt 시 thrust 수직 성분 = m·g·cos(52°) = 0.62·m·g
- 순중력 가속도 ≈ 0.38·g = 3.7 m/s² 하향
- 예상 z_drop @ 0.3s ≈ ½·3.7·0.3² ≈ 0.17m

이것은 **버그가 아니라 정확한 quad 비행 물리**: body-rate 명령으로 회전을
sustained하면 thrust 방향이 수직에서 멀어져 떨어진다. PD가 정확히 트래킹할수록
이 효과는 정직하게 나타난다.

다만 RL 정책은 sustained max body rate 명령을 거의 안 준다 (Gaussian smooth
policy). 따라서 학습 중에는 mild한 거동이며 발산 위험 낮음.

## Phase 3 — 수정 (필요 없음)

진단 결과 controller / mixer / scene 모두 정상이라 수정 사항 없음.
대신 향후 진단 가능성을 위해 instrumentation 영구화:

- `scripts/tune_rate_pid.py` `run_trial`: ω 외에 **z_history도 측정**.
- `compute_metrics`: **z_drop metric** 추가.
- console 출력: 각 trial line 끝에 `z_drop=Xm` 표시.
- `scripts/visualize_rate_controller.py` `run_axis_trial`: z_history 측정 +
  첫 axis(roll) trial 시 `debug_stabilize=True`로 stabilize 첫 10 step 출력.
- `plot_axes`: 2-row → **3-row layout** (zoom + full + z position).
- 두 스크립트 모두 console에 Kd를 `:.2e`로 출력 (PR-I display rounding bug
  수정 — 5e-05 vs 1e-4 구분).

`scripts/diagnose_hover.py`는 일회성 진단이지만 PR-J 증거로 보존.

## Phase 4 — 재검증

### 4-1. Hover-only test (`diagnose_hover.py`)

```
z_drop over 5s        = 0.0000m   ✓ < 0.01m 기준
max |ang_vel| over 5s = 0.0000    ✓ < 0.01 rad/s 기준
rpm0 == hover_rpm     = 8013.7    ✓
```

### 4-2. Step response 재측정 (`tune_rate_pid.py`)

z_drop metric 추가 후 60 trial 재실행. 모든 best combo:

| axis | Kp | Kd | rise(s) | os(%) | ess(%) | **z_drop(m)** | meets |
|---|---:|---:|---:|---:|---:|---:|:---:|
| roll  | 0.05 | 1.0e-4 | 0.010 | 0.00 | 0.02 | **0.028** | ✓ |
| pitch | 0.10 | 5.0e-5 | 0.010 | 0.00 | 0.01 | **0.029** | ✓ |
| yaw   | 0.10 | 1.0e-4 | 0.025 | 0.00 | 0.01 | **-0.007** | ✓ |

- 성공 기준 #3 (`z_drop < 0.05m`) — 셋 다 28-29mm로 PASS.
- yaw는 음수 (오히려 z 약간 상승) — yaw rotation은 thrust 방향 변경 안 시키므로
  떨어지지 않음. saturation 시 일부 motor 보상 thrust로 미세 상승.

### 4-3. Visualize plot 재생성

`outputs/rate_pid_tuning/tuned_response.png` — 3-row × 3-axis layout:
- row 0: zoom 0~60ms (rise 또렷)
- row 1: full metric window 0~0.3s (settling 평탄)
- row 2: z position 0~1.0s (호버 안정 증명)

Plot의 z row가 step phase 1초 동안 drone이 어떻게 떨어지는지를 명시적으로 보여줌:
- roll/pitch: t=0.3s에서 z≈0.97, 이후 0.5~1.0s에서 가파른 하강 (tilt 누적)
- yaw: z 거의 1.0 유지 (tilt 없음)

## Phase 5 — PR-I 결과 재평가

PR-I → PR-J 게인 비교 (z_drop metric 추가 전후):

| axis | PR-I Kp/Kd | PR-J Kp/Kd | 변경 |
|---|---|---|:---:|
| roll  | 0.05 / 1.0e-4 | **0.05 / 1.0e-4** | 동일 |
| pitch | 0.10 / 5.0e-5 | **0.10 / 5.0e-5** | 동일 |
| yaw   | 0.10 / 1.0e-4 | **0.10 / 1.0e-4** | 동일 |

게인 변경 없음. PR-I 결과 valid 확정. yaml 갱신 불필요.

`outputs/rate_pid_tuning/tuning_report.md`는 z_drop 컬럼 자동 추가됨 (재실행 시).

## 변경 파일

- `scripts/tune_rate_pid.py` — z_history 측정 + z_drop metric + Kd display fix.
- `scripts/visualize_rate_controller.py` — z 측정 + stabilize debug + 3-row plot.
- `scripts/diagnose_hover.py` — 신규 (1회성 진단, 향후 회귀 시 재사용 가능).
- `outputs/rate_pid_tuning/step_response.png` — z 컬럼 영향 (재생성).
- `outputs/rate_pid_tuning/pareto.png` — 영향 없음 (재생성).
- `outputs/rate_pid_tuning/tuning_report.md` — z_drop 컬럼 추가 (재생성).
- `outputs/rate_pid_tuning/tuned_response.png` — 3-row layout (재생성).

## 변경 안 한 파일

- `utils/rate_controller.py` — controller 정상.
- `utils/drone_params.py` — quad 형상 정상.
- `configs/envs/rate_controller.yaml` — 게인 변경 없음.
- `configs/envs/pe_1v1_default.yaml`, `pe_1v1_phase15.yaml` — 게인 변경 없음.

## 단계 3 진입 가능 여부

✓ **가능.** PR-J 검증 결과:
- BodyRateController hover 5초 완벽 안정 (z_drop=0)
- Step response valid (mixer 균등, body-rate 측정 invariant)
- z_drop @ metric window 28-29mm (성공 기준 통과)
- 게인 변경 불필요 → yaml 그대로 사용

**잔여 우려 (Phase 7 sim-to-real까지 재검토):**
- 5초 episode 동안 RL 정책이 cumulative tilt를 만들면 ground 충돌 가능. 단계 3
  학습 중 실시간 감시 (mean_episode_length, |ω| late, position out-of-bound 비율).
- 학습 중 발산 시 fallback: action_smooth coef 강화 또는 max_body_rate 한계 ±π/2로 축소.

## 비고

이번 검증은 **반증법으로 BodyRateController가 정상임을 확인**한 작업. 결과는
"문제 없음"이지만 그 증거 자체가 단계 3 진입 결심에 필요한 객관적 근거.
즉 작업 시간 1시간이 RL 학습 5k iter (수 시간) 이전에 안전하게 진행 가능
판정을 가져왔다는 점에서 ROI 양호.
