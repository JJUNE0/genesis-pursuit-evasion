# PR-N — Wrench saturation (PX4-style control allocation)

작업일: 2026-04-29
선행: PR-I/J/K (rate controller PID 튜닝, hover stability, trajectory tracking),
      PR-L (rate-mode RL 검증), PR-M (obs body-frame), AMS-S2 P0 / Phase A / B
      reward 처방 사이클 (logs/.../ams_s2, ams_s2_v2, ams_s2_v3).

## 변경 동기 — Reward 7번 시도 후 발견된 controller bug

Phase 1.5 학습 → AMS-S2 → Phase A (P0 R_opp_crash=0) → Phase B (action_smooth
per-side + omega clamp) 모두 attacker collapse 또는 defender fragility 미해결.

ams_s2_v3 c1 ~ c6에서 **defender 자체 추락 22-42%** 폭증. defender_v10 기준
fragility 진단 (`scripts/test/diag_defender_fragility.py`):
- stationary 상대 def_crash 0.82
- action sat_frac 0.99 (4축 모두 ±1 saturate)
- |ω| mean 5.18 rad/s (max_body_rate=π=3.14의 1.65×)

처음엔 가설 A (bang-bang) + C (intrinsic fragile)로 reward 처방 (Phase B)
정밀화. 효과 부족.

사용자 controller-layer 직관 진단 (`scripts/test/diag_rate_controller.py`):
- bang-bang 명령 (T_norm=±1, ω_ref=±π) 시 4 motor의 thrust 분배 [-7.5, +7.5] N
  범위로 분산
- max_thrust_per_motor=4.4 N, min=0 N clamp 후 4 motor 중 **2개 0 RPM,
  2개 11999 RPM** (saturate)
- **명령 wrench의 65%가 motor saturation으로 손실** (`wrench_loss_frac=0.653`)

PR-M body-frame 가설도 검증 (`scripts/test/diag_get_ang_frame.py`): 90° pitch
+ body-yaw torque test → Genesis `get_ang()`은 world frame, `rotate_to_body_frame`
변환 정확함. **PR-M body-frame mismatch 가설 = REJECTED**.

**진정한 root cause**: Reward shaping이 controller layer wrench loss bug를 가린 것.
정책이 saturated rate command 학습해도 controller가 명령의 35%만 실제 wrench로
적용 → 정책에 일관된 reward 신호 안 감 → bang-bang trap.

## 변경 사항

### 1. `utils/rate_controller.py` — wrench saturation handler 추가 (additive)

`BodyRateController.__init__`에 `enable_wrench_saturation: bool = False`
인자 추가. Default off → 모든 기존 yaml (Phase 1.0/1.5/A/B) 무영향.

`step()` 내부 mixer 적용 직전에 분기:
```python
if self.enable_wrench_saturation:
    wrench = self._saturate_wrench(T_total, torque)
else:
    wrench = torch.cat([T_total, torque], dim=1)
```

신규 메서드 `_saturate_wrench` — 단일 global scale 알고리즘 (priority:
T_total preserved > τ scaled):

1. T-only contribution per motor → `thrusts_t = wrench_t @ M_inv_T`
2. Per-motor headroom:
   - `headroom_pos = max_thrust - thrusts_t` (위로 가용)
   - `headroom_neg = thrusts_t.clamp_min(1e-6)` (아래로 0까지)
3. Torque-only contribution: `thrust_tau = thrusts_full - thrusts_t`
4. Per-env max violation:
   - `pos_v = (thrust_tau / headroom_pos).clamp_min(0).max(dim=1)`
   - `neg_v = (-thrust_tau / headroom_neg).clamp_min(0).max(dim=1)`
   - `max_v = max(pos_v, neg_v)`
5. Scale ∈ (0, 1]: `scale = 1 / max_v.clamp_min(1.0)` — 만 줄이고 절대 amplify 안 함
6. `torque_scaled = torque * scale`, return `[T_total, torque_scaled]`

### 2. `envs/base_pe_env.py` — yaml flag passthrough

`_build_rate_controllers()`에서 `rcfg.get("enable_wrench_saturation", False)`
읽어 controller 두 개 (attacker / defender)에 전달.

### 3. `configs/envs/pe_1v1_phase_c.yaml` — Phase B + 신규 키

`pe_1v1_phase_b.yaml` 복제 + 한 줄:
```yaml
action:
  rate_controller:
    enable_wrench_saturation: true
```

## 검증 결과 (`scripts/test/diag_wrench_saturation.py`)

| case | T_cmd (N) | enable=False | enable=True |
|---|---:|---|---|
| hover (a=0) | 7.86 | sat 0/0, loss 0% | sat 0/0, loss 0% (no-op ✓) |
| bang_+all (T=2g, τ=max) | 15.72 | sat **0.50/0.50** | sat **0.25/0.00** ✓ |
| bang_-all (T=0, τ=max) | 0.00 | sat 0.50/0.50 | sat **0.00**/0.50 |
| bang_T0 (T=0, +max τ) | 0.00 | sat 0.50/0.50 | sat **0.00**/0.50 |
| bang_yaw (T=hover, yaw=max) | 7.86 | sat 0.50/0.50 | sat **0.00**/0.50 |

**핵심 metric — motor saturation (0 RPM + max RPM 비율)**: bang-bang 모두 0.50→0.00
또는 0.25로 감소. wrench_loss% 자체는 92-100%지만 이는 **물리 한계의 정확한 반영**
(T_total + 큰 τ는 drone이 4.4N×4 = 17.6N 한계 외부 → 토크 자동 클립). 정책 학습
관점에서는 **"saturate 영역 = 토크 응답 0"라는 일관된 신호**가 핵심 — RL gradient가
정책을 자연스럽게 saturate 회피 영역으로 밀어낸다.

## 영향 범위

| 영역 | 영향 |
|---|---|
| obs / action shape | 무변경 |
| 체크포인트 호환성 | actor weight transfer 가능. controller 내부 변경만 — 첫 PPO iter에서 자연 적응 |
| Phase 1.0/1.5/A/B yaml 학습 재현성 | default off → 무영향 ✓ |
| Phase J/K (PR-J hover stability, PR-K trajectory) | 회귀 테스트 보류 (`scripts/visualize_rate_controller.py`에 `--enable-saturation` 옵션 추가는 별도 작업) |

## 회귀 / Sanity

- `tests/sanity/test_rate_controller.py` (기존): default off로 통과 — 변경 없음.
- `scripts/test/diag_rate_controller.py` (기존): enable=False 동작 그대로.
- `scripts/test/diag_wrench_saturation.py` (신규): enable on/off 비교.
- `scripts/test/diag_defender_fragility.py` (기존): ams_s2_v4 학습 후 v5 ckpt로
  재실행 시 sat_frac 0.99 → < 0.70 기대.

## 학습 영향 예상 — ams_s2_v4 5 cycle smoke

### 통과 기준 (사전 정의)

**c1 mismatch 진단** (Phase 1.0/1.5 ckpt 그대로 사용 — controller만 다름):
- attacker_winrate ≥ 0.50 (mismatch 미미)
- def_crash count ≤ 200 (saturation handler 효과)

**5 cycle 후 — defender_train phase**:
- def_crash_rate < 0.02 (Phase B 수준 유지)
- captured_rate > 0.65 (s2_v3 c5 0.69 비교)

**5 cycle 후 — attacker_train phase (가장 결정적, inference fragility)**:
- def_crash_rate < 0.10 (s2_v3 c5 0.244 → 회복 목표)

**defender_v5 deterministic eval** (`diag_defender_fragility.py`):
- sat_frac < 0.70 (s2_v3 v10 0.99)
- |ω| mean < 4 (s2_v3 v10 5.18)
- def_crash < 0.30 (s2_v3 v10 0.82)

### 시나리오 분기

| 시나리오 | 조건 | 다음 step |
|---|---|---|
| A (성공) | det eval def_crash<30% AND attacker_train def_crash<10% | 본격 학습 (60 cycle) |
| B (부분) | 둘 중 하나만 만족 | handler 재설계 (PX4 yaw-first) 또는 max_body_rate ±π/2 |
| C (실패) | 둘 다 미달 | controller 외 의심 (PPO config, action space) |

## Follow-up (별도 PR)

- **PX4 yaw-first priority handler**: 단일 global scale 대신 yaw 먼저 줄이고
  roll/pitch 보존. smoke 부분 성공 시 검토.
- **`visualize_rate_controller.py --enable-saturation`**: PR-J/K 회귀 테스트
  자동화. 별도 작업.
- **mixer cond(M)=80 재검토**: Nova drone params (arm_length, motor 위치,
  torque-thrust ratio) 합리성. 현재 borderline이지만 별개 이슈.
- **hover env frame bug**: `envs/hover/env.py:384` raw `get_ang()` →
  `rotate_to_body_frame` 적용 (사용자 viewer "모터 까딱까딱" 신호 fix).

## 비고

ams_s2_v3 collapse 분석 사이클이 처방 7번 시도했지만, **reward shaping은
controller layer bug를 가리는 것이 아니라 그 위에서만 효과가 있다**는 사실을
입증한 사이클이었다. 사용자의 controller-layer 직관 진단 ("프로펠러 안 도는
현상은 saturation 또는 mixer 문제 신호일 수 있음")이 7번째 처방의 결정적
방향 전환이었다.
