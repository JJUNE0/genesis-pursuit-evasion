# Phase 1 — `BasePursuitEvasionEnv` + 1v1 PE Env 사전 plan

> CLAUDE.md §13 RPIR "Plan" 단계 산출물. TODO.md Phase 1 (2026-04-26 개정판) 1:1 반영.
> 변경 정책: CLAUDE.md §1 — 기존 동작 코드 미수정, 새 기능은 새 파일.
> 본 plan 승인 시 PR‑A → PR‑B → PR‑C 순으로 구현 시작.

---

## 1. 변경 동기

논문 §3.1.2 Goal‑Delay POMDP의 $\tau=0$ **perfect‑info baseline**을 작성한다. Phase 2 (naive delayed obs), Phase 3 (forecaster $f_\psi$), Phase 4 (asymmetric privileged critic) 의 회귀 비교 reference이며, Phase 5 AMS‑DRL co‑training 의 attacker 측 warm‑start ckpt를 만들어내는 단계.

**Defender 철학 (TODO.md §Phase 1):** Defender는 *mission 저지* 에이전트이며 capture 에이전트가 아니다. 목적은 attacker가 `g_mission`에 도달하는 것을 막는 것 — capture는 그를 위한 수단이지 목적이 아님. 따라서 timeout은 defender win으로 처리된다 (attacker 가 시간 내 도달 실패 = mission 좌절).

Phase 0 sanity 4종 (V1 / V1.5 / V2 / V3) 이 모두 PASS:
- V1·V1.5 ([tests/sanity/two_drones_in_one_env.py](../../tests/sanity/two_drones_in_one_env.py), [tests/sanity/dual_drone_differential_control.py](../../tests/sanity/dual_drone_differential_control.py))
- V2 ([tests/sanity/two_runners_alternating.py](../../tests/sanity/two_runners_alternating.py))
- V3 ([tests/sanity/asymmetric_critic_obs.py](../../tests/sanity/asymmetric_critic_obs.py))

→ Plan A (한 Scene 두 Drone) 입증. 골격 작성 단계 즉시 진입.

---

## 2. 범위

### 2.1 Phase 1 범위 (TODO.md §1.1 – §1.6)

- `BasePursuitEvasionEnv` ABC + `PursuitEvasion1v1Env` 구현체
- scripted **attacker** 1종 (`random_walk`) — Phase 1.0 (defender pretraining) 용
- scripted/loaded **defender** 3종 (`stationary` / `random_walk` / `pretrained`) — Phase 1.5 (attacker training) 용
- `agents/ppo_runner.py` thin wrapper (Phase 1.0 / 1.5 공통)
- `scripts/train_defender.py` — Phase 1.0, random‑walk attacker 상대로 defender_v0 PPO 학습
- `scripts/train_attacker.py` — Phase 1.5, `--defender {stationary,random_walk,pretrained}` 옵션
- `obs_groups` 파이프라인 사전 배선 (Phase 4의 `critic_priv` plug‑in 자리만 비워둠)
- shape contract / termination priority / reward zero‑sum / capture·mission·crash·timeout 종료 unit test

### 2.2 비범위 (다른 phase 책임)

- τ-step delay buffer → Phase 2
- forecaster $f_\psi$ → Phase 3
- privileged critic 입력 채우기 → Phase 4 (Phase 1은 wiring만)
- AMS‑DRL co‑training (alternating) → Phase 5
- CTBR action space, SimpleFlight Factor 1·2 → Phase 7 sim‑to‑real

CLAUDE.md §14 — 단일 PR에 phase 두 개 이상 섞지 않는다.

---

## 3. 영향 범위 (Files)

### 3.1 신규 파일 (CLAUDE.md §1)

| Path | 책임 |
|---|---|
| `envs/__init__.py` | 패키지 초기화 |
| `envs/base_pe_env.py` | `BasePursuitEvasionEnv` ABC. scene + 두 drone 소유, batched step, obs dict 출력, 추상 hook |
| `envs/pe_1v1/__init__.py` | 1v1 패키지 초기화 |
| `envs/pe_1v1/env.py` | `PursuitEvasion1v1Env(BasePursuitEvasionEnv)` |
| `envs/pe_1v1/cfgs.py` | env_cfg/obs_cfg/reward_cfg/command_cfg 빌더 (TODO.md §1.2 magnitude 표 그대로) |
| `envs/pe_1v1/scripted_attacker.py` | `RandomWalkAttacker` (Phase 1.0 — defender 학습 시 사용) |
| `envs/pe_1v1/scripted_defender.py` | `StationaryDefender`, `RandomWalkDefender`, `PretrainedDefender` (NN ckpt 로딩 + frozen forward) |
| `agents/__init__.py`, `agents/ppo_runner.py` | rsl‑rl `OnPolicyRunner` thin wrapper. Phase 1.0 / 1.5 공통 사용 (single runner). Phase 5 의 두‑runner orchestrator 는 별도 phase. |
| `configs/envs/pe_1v1_default.yaml` | spawn / g_mission 분포, dt, episode length, capture / mission 반경, reward magnitudes |
| `configs/train/ppo_attacker.yaml` | Attacker PPO hyperparams |
| `configs/train/ppo_defender.yaml` | Defender PPO hyperparams (Phase 1.0) |
| `scripts/train_defender.py` | Phase 1.0 — random‑walk attacker 상대로 defender_v0 학습 |
| `scripts/train_attacker.py` | Phase 1.5 — `--defender {stationary,random_walk,pretrained}` 옵션 |
| `scripts/eval.py` | cfgs.pkl 기반 deploy 평가, `critic_priv` 미build 강제 (CLAUDE.md §7 / V3 contract) |
| `tests/test_shapes.py` | obs/action shape, dtype, device contract |
| `tests/sanity/test_termination_priority.py` | mission_reached + captured 동시 trigger → attacker win 검증 |
| `tests/sanity/test_capture_termination.py` | 5종 종료 (mission_reached / captured / crash / oob / timeout) 트리거 |
| `tests/sanity/test_reward_zero_sum.py` | terminal reward zero‑sum 의무 검증 |
| `tests/test_scripted_policies.py` | random_walk attacker / 3종 defender output shape, finite, device |

### 3.2 수정 없음

[utils/drone_params.py](../../utils/drone_params.py), [controllers/pid_controller.py](../../controllers/pid_controller.py), [controllers/se3_controller.py](../../controllers/se3_controller.py), [configs/drones/nova.yaml](../../configs/drones/nova.yaml), 기존 sanity 4종.

PID controller는 Phase 1에서 **사용하지 않는다** (TODO.md §1.3 — defender 옵션이 stationary/random_walk/pretrained로 변경됨에 따라 PID chase defender가 제거됨). Phase 5 이후 baseline 비교 용도로 남겨둠.

### 3.3 격리 유지

[reference/legacy_envs/](../../reference/legacy_envs/) 절대 import 금지. `BaseDroneEnv` hook 명명 / `_init_rewards` 패턴은 *참조만*; 새 ABC를 처음부터 작성한다.

---

## 4. `BasePursuitEvasionEnv` 설계

### 4.1 책임 (CLAUDE.md §6, TODO.md §1.2)

- 하나의 `gs.Scene` + `attacker_drone` + `defender_drone` 두 entity (V1.5 검증 패턴 그대로).
- 두 드론 모두 env 책임 — env가 attacker policy / defender policy 양쪽을 호출하고 batched RPM dispatch.
- `_step_attacker()` 와 `_step_defender()` 가 cfg에 따라 다음 중 하나로 동작:
  - **Phase 1.0** 학습 모드: attacker = `random_walk` (frozen, NN 아님), defender = NN policy (학습 중)
  - **Phase 1.5** 학습 모드: attacker = NN policy (학습 중), defender = `stationary` / `random_walk` / `pretrained`
- batched step. for‑loop over (env, entity) 금지 (CLAUDE.md §14).
- obs는 `TensorDict` dict, flat concat은 actor / critic forward 직전 1회만:
  ```python
  obs = {
      # 학습 중인 ego 에이전트가 attacker일 때:
      "policy": torch.cat([s_a, s_d_kin, g_mission], dim=1),  # shape: (B, 18+6+3=27)
      # 학습 중인 ego 에이전트가 defender일 때:
      # "policy": torch.cat([s_d, s_a_kin], dim=1),           # shape: (B, 18+6=24)
      # g_mission은 defender에 노출하지 않음 (논문 §3.6.3 K6).
      #
      # Phase 1: critic_priv 키 자체를 만들지 않음 → obs_groups = {"actor":["policy"], "critic":["policy"]}
      # Phase 4에서 "critic_priv" 추가 → obs_groups = {"actor":["policy"], "critic":["policy","critic_priv"]}
  }
  ```
- τ‑delay buffer는 env 책임 (Phase 2). Phase 1은 wiring 자리만 둔다 — `_get_attacker_obs()` 가 `s_d_t` 즉시 반환.

### 4.2 추상 hook (TODO.md §1.2)

CLAUDE.md §3 — `target` 단어 금지. PE 도메인용으로 rename.

| Hook | 책임 |
|---|---|
| `_init_pe_buffers()` | PE 전용 텐서 (g_mission, dist_to_goal_prev, capture_buf, mission_reached_buf, crash_buf) 할당 |
| `_reset_pe(envs_idx)` | attacker / defender pose, g_mission 샘플 (batched) |
| `_step_attacker_policy()` | attacker action → RPM tensor (학습 모드 / scripted 모드 모두) |
| `_step_defender_policy()` | defender action → RPM tensor |
| `_update_pe_state()` | dist_to_goal, dist_to_attacker_defender, attacker / defender state buffer 갱신 |
| `_attacker_obs()` | attacker 입장 actor 입력 텐서 |
| `_defender_obs()` | defender 입장 actor 입력 텐서 (Phase 1.0 학습용) |
| `_compute_capture()` | $d(p^a_t, p^d_t) < d_{\text{capture}}$ |
| `_compute_mission_reached()` | $d(p^a_t, g_{\text{mission}}) < d_{\text{goal}}$ |
| `_compute_crash()` | (a) 지면 충돌 OR (b) `\|`roll`\|` or `\|`pitch`\|` > 60° OR (c) ‖v‖ > v_max |
| `_attacker_reward()` | TODO.md §1.2 attacker reward 표 |
| `_defender_reward()` | TODO.md §1.2 defender reward 표 (Phase 1.0 / 1.5 양쪽에서 계산되며 used by 학습 ego만) |

### 4.3 종료 조건 우선순위 (TODO.md §1.2 — 핵심 의무)

같은 step에서 여러 termination flag가 동시에 참일 수 있다. 처리 우선순위:

1. **`mission_reached` 우선**: `mission_reached AND captured` 동시 발생 → **attacker win** (mission 도달 = 폭발 = 목적 달성, capture 무력화).
2. `mission_reached` (단독) → attacker win (`R_mission_win`, `R_mission_failed`).
3. `captured` (단독) → defender win (`R_capture_win`, `R_captured`).
4. `crash` → 양쪽 모두 `R_crash` (zero‑sum 아님; 양쪽이 음수). attacker / defender 어느 쪽에서 crash 가 발생했는지 별도 buffer로 구분 가능 — Phase 1은 단순화: crash 발생 시 둘 다 `-1.0`.
5. `out_of_bounds` → attacker / defender 중 누가 이탈했는지에 따라 처리 (구체 규칙은 cfg).
6. `timeout` → defender win (`R_timeout_win = +0.5`, `R_timeout = -0.5`).

`tests/sanity/test_termination_priority.py` 가 1번 우선 규칙을 강제 검증.

### 4.4 Shape / 단위 의무 (CLAUDE.md §4 §5)

- 모든 buffer에 `# shape: (B, ...)` 주석.
- ENU 좌표계, 위치 m, 속도 m/s (world frame), 각도 rad (low‑level), deg (log only).
- quat order = (w,x,y,z) (V1 sanity 확정).
- `nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)` 모든 reward / obs 출력 직전.
- `x.norm(...)` 직후 `.clamp_min(1e-6)` 또는 `+ 1e-8`.
- 모든 텐서 `device=self.device`. CPU↔GPU 암묵 이동 금지.

---

## 5. `PursuitEvasion1v1Env` 구체

### 5.1 환경 파라미터 (TODO.md §1.2 — 확정값)

```yaml
sim:
  dt: 0.01
  substeps: 4
  episode_length_s: 5.0      # → max_steps = 500
termination:
  d_capture: 0.3             # m
  g_mission_radius: 0.3      # m
  pitch_max_deg: 60
  roll_max_deg: 60
  z_floor: 0.1
  v_max: 10.0                # m/s, crash 임계
  bound_xyz: 5.0             # workspace half‑extent
spawn:
  attacker_init_pos: { x: [-1.0, 1.0], y: [-1.0, 1.0], z: [1.0, 1.5] }
  defender_init_pos:
    distance_to_attacker: [1.0, 2.0]   # collapse 방지 (Phase 5.4 가이드 사전 적용)
    z: [1.0, 1.5]
g_mission:
  box: { x: [-2.5, 2.5], y: [-2.5, 2.5], z: [1.0, 1.5] }
  prefer_far_from_defender: true       # rejection sampling: |g - defender| > 1.0 m
```

### 5.2 Action space (TODO.md §1.2 확정)

**양 에이전트 공통**: scaled‑hover RPM, $a \in [-1,1]^4$, `rpms = (1 + 0.5·a) · hover_rpm`.

PWM 매핑 일관성 위해 CTBR은 Phase 7 sim‑to‑real 시 도입 (TODO 명시).

### 5.3 Defender obs 차원 (TODO.md §1.2 확정)

`defender_priv` = **kinematic 6‑dim** (pos+vel). Phase 2/3/4와 ckpt 호환을 위함.

### 5.4 Scripted / loaded 정책

`envs/pe_1v1/scripted_attacker.py`:
- **`RandomWalkAttacker`**: 매 step `action ~ U(-1, 1)^4` sampling → scaled‑hover RPM. Phase 1.0 defender 학습용. PID/PETN 미사용 — 단순 random.

`envs/pe_1v1/scripted_defender.py`:
- **`StationaryDefender`**: 매 step `hover_rpm` 그대로. RPM tensor shape `(B, 4)`.
- **`RandomWalkDefender`**: 매 step `action ~ U(-1, 1)^4` sampling → scaled‑hover RPM.
- **`PretrainedDefender`**: Phase 1.0 산출물 `defender_v0.ckpt` 로드, frozen forward (`requires_grad=False`, `eval()` 모드). attacker 학습 중 obs 받아 action 출력.

PID 기반 추적은 사용하지 않는다 (TODO.md §1.3 변경에 따라 제거). 따라서 `controllers/batched_pid.py` 파일도 만들지 않는다.

각 정책 인터페이스:
```python
class ScriptedPolicy(Protocol):
    def reset(self, envs_idx: torch.Tensor) -> None: ...
    def step(
        self,
        ego_state: dict[str, torch.Tensor],     # pos, vel, quat — 자기 상태
        opponent_state: dict[str, torch.Tensor],
        g_mission: torch.Tensor | None,         # attacker만
    ) -> torch.Tensor:                          # shape: (B, 4) action ∈ [-1,1]^4
        ...
```

env가 RPM 변환 (`scaled-hover`) 까지 책임지므로 정책은 action만 반환.

---

## 6. Reward 설계 (TODO.md §1.2 표 — asymmetric dense + zero‑sum terminal)

### 6.1 Attacker reward

| 항 | 형태 | 종류 | 권장 magnitude |
|---|---|---|---|
| `R_progress` | potential‑based: $\gamma \cdot \Phi(s_{t+1}) - \Phi(s_t)$, $\Phi = -\|p_a - p_{\text{mission}}\|$, $\gamma$ = PPO discount (0.99) | dense | ±0.002 / step max |
| `R_action_smooth` | $-\|a_t - a_{t-1}\|^2$ | dense | −0.0001 / step max |
| `R_existential` | $-w_1 / T_{\text{max}}$ (attacker only, $w_1=1.0$) | dense | −0.002 / step |
| `R_mission_win` | mission 도달 시 | terminal | **+1.0** |
| `R_captured` | capture 당함 | terminal | **−1.0** |
| `R_timeout` | timeout 시 | terminal | **−0.5** |
| `R_crash` | crash 시 | terminal | **−1.0** |

### 6.2 Defender reward

| 항 | 형태 | 종류 | 권장 magnitude |
|---|---|---|---|
| `R_block` | $= -(\text{attacker's }R_{\text{progress}})$ | dense | ±0.002 / step max |
| `R_action_smooth` | $-\|a_t - a_{t-1}\|^2$ | dense | −0.0001 / step max |
| `R_capture_win` | capture 성공 시 | terminal | **+1.0** |
| `R_mission_failed` | attacker가 mission 도달 시 | terminal | **−1.0** |
| `R_timeout_win` | timeout 시 (mission 저지 성공) | terminal | **+0.5** |
| `R_crash` | crash 시 | terminal | **−1.0** |

### 6.3 의무 사항

- **Existential penalty 는 attacker만**. Defender는 받지 않는다 (TODO 명시).
- **Terminal reward zero‑sum 보장**:
  - $R_{\text{mission\_win}} + R_{\text{mission\_failed}} = 0$
  - $R_{\text{capture\_win}} + R_{\text{captured}} = 0$
  - $R_{\text{timeout\_win}} + R_{\text{timeout}} = 0$
  - `R_crash`는 양쪽 모두 −1.0 — zero‑sum 아님 (안전 페널티). 학습 동역학상 attacker / defender 양쪽이 동일하게 회피.
- `R_block`은 attacker `R_progress`의 정확한 부호 반전. 한 step의 progress 가 양쪽 reward에 동시 반영되어 zero‑sum dense 흐름 보장.
- 합산 직전 `nan_to_num`. 각 항 `× scale` (TODO 표의 magnitude는 이미 step 단위라 `× dt` 추가 곱 안 함 — 이 점 BaseDroneEnv 패턴과 다르므로 cfgs.py 주석에 명시).

`tests/sanity/test_reward_zero_sum.py` 가 (i) terminal zero‑sum 3쌍, (ii) `R_block + R_progress = 0` (양쪽 dense 합) 을 매 step 단위로 강제.

---

## 7. 학습 파이프라인

### 7.1 `agents/ppo_runner.py`

rsl‑rl 5.2 `OnPolicyRunner` thin wrapper. Phase 1.0 (defender) / 1.5 (attacker) 양쪽에서 재사용 — single runner 한 인스턴스. Phase 5의 두‑runner orchestrator는 별도 phase.

책임:
- env, train_cfg, log_dir 받아 `OnPolicyRunner` 인스턴스 생성·반환.
- `cfgs.pkl` dump (CLAUDE.md §10): `(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, drone_config, ams_stage)` — `ams_stage = "phase1.0_defender"` or `"phase1.5_attacker_<defender_mode>"`.
- wandb run name = `phase1_<role>_<defender_mode>_<timestamp>_<git_sha>(-dirty)` (CLAUDE.md §10).
- `--logger {wandb,tensorboard}` 토글 (default wandb, TODO §1.5 확정).

### 7.2 `scripts/train_defender.py` — Phase 1.0

- random‑walk attacker (frozen, NN 아님) 상대로 defender PPO 학습.
- env: `PursuitEvasion1v1Env`, cfg는 `attacker_mode = "random_walk"`, `defender_mode = "nn"` 으로 설정 — 같은 env 재사용 (별도 capture_env 만들지 않음, TODO 명시).
- defender obs: `[s_d, s_a_kin]` (24‑dim), g_mission 미노출 (논문 §3.6.3 K6).
- 학습 sanity: random‑walk attacker 대해 defender capture rate > 0.5 (학습 ≤ 5k iter).
- 산출물: `logs/pe1v1_v0/defender_v0/<timestamp>/model_*.pt` 중 최고 capture rate ckpt → `defender_v0.ckpt`로 alias.
- 시드 4종 강제, `--max_iterations >= 100` 시 사용자 사전 승인 (CLAUDE.md §11 — 본 단계는 5k iter 명시이므로 PR‑B에서 사용자 OK 받음).

### 7.3 `scripts/train_attacker.py` — Phase 1.5

- CLI: `--defender {stationary,random_walk,pretrained}`, `--num_envs`, `--max_iterations`, `--seed`, `--logger`, `--wall_clock_budget_min`, `--show_viewer`, `--defender_ckpt` (pretrained 모드 시).
- 시드 4종 강제 (CLAUDE.md §10): `gs.init(seed=...)`, `torch.manual_seed`, `np.random.seed`, `random.seed`.
- attacker obs: `[s_a, s_d_kin, g_mission]` (27‑dim).
- 학습 sanity (학습 진행 여부 binary check, 절대 winrate 아님 — TODO 의주):
  - stationary defender 대해 attacker winrate > 0.9 (≤ 5k iter)
  - pretrained defender_v0 대해 winrate > 0.5 (≤ 10k iter)
- `obs_groups = {"actor":["policy"], "critic":["policy"]}` (Phase 1, critic_priv 미사용).

### 7.4 `scripts/eval.py`

- cfgs.pkl만 보고 env / runner 재구성 (CLAUDE.md §10).
- deploy mode: `obs["critic_priv"]` 키 build 자체 안 함 — V3 sanity test 4 contract 동일.
- 출력: episode 단위 winrate, mission_reached / captured / timeout / crash 비율.

---

## 8. 물리 / 이론 sanity check (CLAUDE.md §13 의무)

- **Plan A 회귀 가드**: env build 직후 `_assert_plan_a()` 1회 호출 — quat order = (w,x,y,z), `hover_rpm ∈ [1000, 50000]`, drone 한쪽 `set_pos` 가 다른 쪽 / 다른 env 슬롯에 누수 없음.
- **Hover sanity**: stationary defender + random_walk attacker (정지에 가까운 mean) 1초 → drone 양쪽 `pos[:, 2] - 1.0 < 0.1 m` (단순 sanity).
- **Termination priority** (`tests/sanity/test_termination_priority.py`): mission_reached + captured 동시 trigger → `extras["episode"]` 가 `attacker_win`으로 분류.
- **5종 종료** (`tests/sanity/test_capture_termination.py`): mission_reached / captured / crash / oob / timeout 각각이 `reset_buf` 트리거 + 정확한 reward 부여.
- **Reward zero‑sum** (`tests/sanity/test_reward_zero_sum.py`): (i) terminal 3쌍 zero‑sum, (ii) dense `R_block + R_progress = 0` 매 step.
- **Reward sanity**: 모든 항 finite, attacker가 g_mission으로 직선 이동 시 `R_progress > 0`, defender가 attacker progress 차단 시 `R_block > 0` (signed test).
- **Shape contract** (`tests/test_shapes.py`): obs / action key가 `(B, ...)` shape, `dtype == gs.tc_float`, `device == self.device`.

---

## 9. Rollback plan

- 모든 신규 파일이 새 디렉토리에 격리 (`envs/base_pe_env.py`, `envs/pe_1v1/`, `agents/ppo_runner.py`, `scripts/train_*.py`, `scripts/eval.py`) → `git revert` / PR revert 한 번으로 복원 가능.
- 체크포인트 호환성: Phase 1은 새 obs space (legacy hover와 다름) → 로그 디렉토리 분리: `logs/pe1v1_v0/`. legacy hover ckpt와 충돌 없음.
- Phase 1.0 의 `defender_v0.ckpt` 산출물 자체는 별도 ckpt → Phase 1.5 실패 시 `--defender stationary` 로 fallback 학습 가능 (gate 완화).
- 의존성 추가 없음 (rsl-rl, torch, genesis, tensordict 모두 [pyproject.toml](../../pyproject.toml) 에 포함).
- 단계적 디버깅: `--defender stationary` → `random_walk` → `pretrained` 순으로 attacker 학습 난이도 점진 증가.

---

## 10. 테스트 / Acceptance gate

### 10.1 Unit (CPU 가능)

- `tests/test_shapes.py` — obs / action shape, dtype, device.
- `tests/sanity/test_termination_priority.py` — 동시 trigger 처리.
- `tests/sanity/test_capture_termination.py` — 5종 종료.
- `tests/sanity/test_reward_zero_sum.py` — terminal + dense zero‑sum.
- `tests/test_scripted_policies.py` — random_walk attacker / 3종 defender output shape, finite.

### 10.2 Integration (GPU 필요)

기존 sanity 4종 회귀 (no regression):
```
python tests/sanity/single_drone_hover.py
python tests/sanity/two_drones_in_one_env.py
python tests/sanity/dual_drone_differential_control.py
python tests/sanity/asymmetric_critic_obs.py
python tests/sanity/two_runners_alternating.py
```

PR‑A smoke (5 iter, CLAUDE.md §11):
```
python scripts/train_defender.py --max_iterations 5 --num_envs 64 --seed 0 --logger tensorboard
python scripts/train_attacker.py --defender stationary --max_iterations 5 --num_envs 64 --seed 0 --logger tensorboard
```

### 10.3 Phase 1 → Phase 2 진입 acceptance gate (TODO.md §1.4 §1.5)

Phase 1.0 (defender pretraining):
- random‑walk attacker 대해 defender capture rate **> 0.5** (5k iter 내)
- `defender_v0.ckpt` artifact 첨부

Phase 1.5 (attacker training):
- stationary defender attacker winrate **> 0.9** (5 seed 평균, ≤ 5k iter)
- pretrained defender_v0 attacker winrate **> 0.5** (5 seed 평균, ≤ 10k iter)
- random_walk defender attacker winrate 보고 (gate 아님; 비교 reference)
- wandb URL + winrate / episode length / reward decomposition 표 PR 본문에 첨부
- TODO.md §1.5 reward shaping ablation (potential‑based progress vs sparse‑only) 1회 보고

---

## 11. PR 분할 (CLAUDE.md §14)

Phase 1 안에서도 review 폭 관리:

- **PR‑A (env 골격)**: `envs/base_pe_env.py`, `envs/pe_1v1/{env,cfgs,scripted_attacker,scripted_defender}.py`. 5종 unit test green. NN 학습 없이 scripted vs scripted (e.g., random_walk attacker × random_walk defender) rollout으로 reward 흐름 / 종료 처리 / zero‑sum 검증.
- **PR‑B (Phase 1.0 defender pretraining)**: `agents/ppo_runner.py`, `scripts/train_defender.py`, `configs/train/ppo_defender.yaml`. 5k iter 학습 → `defender_v0.ckpt` 산출 + capture rate > 0.5 보고.
- **PR‑C (Phase 1.5 attacker training + acceptance gate)**: `scripts/train_attacker.py`, `scripts/eval.py`, `configs/train/ppo_attacker.yaml`, `envs/pe_1v1/scripted_defender.py::PretrainedDefender` 활성화. 5 seed × 3 defender 학습 결과, winrate 표, ckpt, Phase 2 진입 승인 요청.

---

## 12. 결정 사항 (TODO.md 2026-04-26 개정에서 모두 반영됨, 본 plan에서 추가 결정 필요한 항목 명시)

TODO.md에서 확정된 항목 — 본 plan은 그대로 따름:

| 항목 | 확정값 | 출처 |
|---|---|---|
| Action space | scaled‑hover RPM ∈ [-1,1]^4 (양 에이전트 공통) | TODO §1.2 |
| `defender_priv` 차원 | kinematic 6‑dim | TODO §1.2 |
| Defender 옵션 | stationary / random_walk / pretrained | TODO §1.3 |
| Defender 철학 | mission 저지 (capture는 수단) | TODO §Phase 1 |
| Reward 구조 | potential‑based progress + zero‑sum dense + zero‑sum terminal | TODO §1.2 |
| Existential penalty | attacker만 | TODO §1.2 |
| timeout 처리 | defender win | TODO §1.2 |
| 동시 trigger | mission_reached + captured → attacker win | TODO §1.2 |
| episode_length_s / dt | 5.0 / 0.01 (500 step) | TODO §1.2 |
| d_capture / g_mission_radius | 0.3 m / 0.3 m | TODO §1.2 |
| logger default | wandb | TODO §1.5 |

본 plan에서 추가 결정 필요한 항목 (PR‑A 시작 전 사용자 confirm):

| # | 항목 | 권장 | 이유 |
|---|---|---|---|
| P1 | Defender obs 구성 | `[s_d, s_a_kin]` 24‑dim, g_mission 미노출 | 논문 §3.6.3 K6 |
| P2 | crash 시 reward 처리 | 양쪽 모두 −1.0 (zero‑sum 아님) | TODO §1.2 표 그대로; 양쪽 회피 동역학 의도 |
| P3 | spawn collapse 방지 거리 | `\|p_a - p_d\| ∈ [1.0, 2.0] m` | Phase 5.4 가이드 사전 적용 |
| P4 | random_walk RPM 분포 | $a \sim U(-1, 1)^4$ uniform | TODO "무작위 action" 명시; PID 미사용 |
| P5 | `R_progress` magnitude clipping | ±0.002 / step max (TODO 표) → 구현 시 `clamp(±0.002)` | 학습 안정성, dense reward 폭주 방지 |
| P6 | `defender_v0.ckpt` 보관 위치 | `logs/pe1v1_v0/defender_v0/best.ckpt` + symlink → `defender_v0.ckpt` | 재현성 + Phase 1.5에서 명시 경로 |

---

## 13. Critical files (참조용)

### 13.1 읽기 전용

- [TODO.md](../../TODO.md) §Phase 1 (line 119–205)
- [CLAUDE.md](../../CLAUDE.md) §1 §3 §4 §5 §6 §7 §10 §11 §13 §14
- [reference/legacy_envs/base_drone_env.py](../../reference/legacy_envs/base_drone_env.py) — hook 명명 / `_init_rewards` 패턴 *참조만*
- [reference/legacy_envs/hover/env.py](../../reference/legacy_envs/hover/env.py) — task hook 채움 패턴 *참조만*
- [tests/sanity/two_drones_in_one_env.py](../../tests/sanity/two_drones_in_one_env.py) — Plan A scene 빌드 / spawn / batched RPM
- [tests/sanity/asymmetric_critic_obs.py](../../tests/sanity/asymmetric_critic_obs.py) — `obs_groups` cfg layout
- [docs/section3_formulation.md](../section3_formulation.md) §3.1.2 §3.3 §3.6.3 — Goal‑Delay POMDP, defender V‑critic
- [docs/section4_method.md](../section4_method.md) §4.2 §4.5 — obs / critic 차원, reward design

### 13.2 재사용 (import / wrap)

- [utils/drone_params.py](../../utils/drone_params.py) — `DroneParams`
- [configs/drones/nova.yaml](../../configs/drones/nova.yaml) — drone 물리 상수
- `genesis as gs` — `gs.Scene`, `gs.morphs.Drone`, `gs.options.*`, `gs.tc_float`
- `rsl_rl.runners.OnPolicyRunner` — V2 / V3 sanity와 동일 cfg layout

### 13.3 사용하지 않음 (Phase 1 한정)

- [controllers/pid_controller.py](../../controllers/pid_controller.py), [controllers/se3_controller.py](../../controllers/se3_controller.py) — defender pid_chase 옵션 제거됨에 따라 Phase 1에서 미사용. Phase 5 이후 baseline 비교 시 재논의.
- `controllers/batched_pid.py` — 본 plan에서 신규 작성 안 함 (TODO 변경에 따라 불필요).

---

## 14. Open questions (구현 중 결정)

- random_walk attacker / defender 의 action smoothing 여부. uniform sampling만으로 충분히 동역학적 stress test가 되는지, 또는 OU process로 부드럽게 할지 — Phase 1.0 학습 안정성 보고 결정.
- pretrained defender ckpt 의 BatchNorm / Dropout — rsl‑rl PPO actor가 이를 사용하는지 V3 sanity 빌드 시 actor 구조에서 확인. 사용 시 `eval()` 모드 강제.
- domain randomization (DR) 도입 여부 — Phase 1에서는 미도입 (CLAUDE.md §1 변경 범위 최소화). Phase 7 sim‑to‑real 에서 재논의.
- crash 발생 envs 의 reset 후 g_mission 재샘플 정책 (즉시 vs 다음 episode 시작 시) — 일관성 위해 즉시 재샘플 권장.

---

*본 plan 승인 시 PR‑A부터 코드 작성 시작.*
*P1–P6 결정사항은 PR‑A 시작 전 사용자 confirm 의무 (CLAUDE.md §1 — obs / reward schema 변경).*
