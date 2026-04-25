# CLAUDE.md — Genesis Pursuit-Evasion (논문 작업용)

> **이 문서는 변하지 않는 규칙만 담는다.** 구현해야 할 것·진행 상황은 [TODO.md](./TODO.md).
> 새로운 phase가 추가돼도 이 규칙은 그대로다. 규칙을 바꾸려면 PR + 사용자 승인.

---

## 0. 의사소통

* 사용자는 한국어/영어 자유롭게, **Claude는 항상 한국어로 답변**한다.
* 사용자가 작성한 논문 노트(`paper/`, `docs/papers/`)는 **읽기만** 한다. 수정 요청 없이 자동 갱신 금지.
* Claude는 항상 **Devil's Advocate** 모드로 행동한다 (사용자 `userPreferences` 기반):
  1. 가장 취약한 논리적 연결고리 먼저 비판
  2. 최악의 엣지 케이스 시뮬레이션
  3. 비판 후에만 "당장 검증할 3가지"를 우선순위 순으로 제시

---

## 1. 변경 정책 (Change Policy) — **가장 강한 규칙**

* **기존 동작하는 코드는 손대지 않는다.** 새 기능은 **새 파일**로 추가가 원칙.
  obs shape, action space, env 인터페이스, config schema를 변경하는 PR은 **사용자 명시 승인** 필수.
* **체크포인트 호환성 깨지는 변경 금지** (obs 차원, normalization, action 정의 등). 깨야 한다면 별도 폴더(`logs_v2/`)로 분리.
* **silent migration 금지**: config 키 rename, default 값 변경, 좌표계 전환은 deprecation warning 1회 추가 후 다음 PR에서 적용.
* **삭제는 영구**: 코드 삭제 PR은 그 이유와 영향받는 호출자를 본문에 명시. `git revert` 대비.

---

## 2. 기술 스택 고정 (Stack Lock)

| Layer | 선택 | 비고 |
|---|---|---|
| Physics | **Genesis AI** | Omnidrones / IsaacGym으로 갈아타지 않는다. |
| RL Framework | **rsl-rl-lib >= 5.0.0** | `OnPolicyRunner` 사용. 두 정책은 **runner 두 개를 띄워서 alternating**. multi-policy fork 금지. |
| Tensors | **PyTorch float32** | `gs.tc_float`, `gs.tc_int` 통일. mixed precision 도입은 별도 phase. |
| Numpy | float32 강제 | `np.float64`는 controller 내부 수치 안정성 외에는 사용 금지. |
| Config | **YAML 단일 진입점** | 모든 물리 상수는 `configs/<entity>/*.yaml` → `DroneParams`. **하드코딩 금지**. |
| Logging | **WandB 기본**, TB는 fallback | `--logger` CLI 인자로 토글. |

* **Python 버전, 패키지 버전**은 `pyproject.toml`이 단일 출처. README 또는 docstring에 적힌 버전이 다르면 `pyproject.toml`이 정답.

---

## 3. 도메인 표기 규약 (Symbol Contract) — **논문 §3과 1:1 동기화**

* `s^a` = **attacker** state (학습 대상, 우리 논문 PoV)
* `s^d` = **defender** state (지연 관측 대상)
* `g_mission` = attacker가 도달해야 할 navigation goal (관측 가능, conditioning variable)
* `τ` = communication delay step 수 (정수 단위, env dt 배수)
* `f_ψ` = forecaster (transformer)
* `Q_θ` = asymmetric critic
* `π_φ^a`, `π_φ^d` = attacker / defender 정책

**규칙**:
1. 코드 변수명은 위 기호와 1:1 매핑. `attacker_state`, `defender_state`, `mission_goal`, `delay_tau`, `forecaster`, `critic`, `attacker_policy`, `defender_policy`.
2. 짧은 이름(`sa`, `sd`)은 텐서 연산 hot path에서만 허용, public API/저장 키엔 풀네임.
3. `target` 단어 금지 — `g_mission`(목적지)인지 `defender`(추적 대상)인지 모호함. 둘 다 풀네임으로.

---

## 4. 텐서 / Shape 규약

* **모든 텐서 변수에 shape 주석 의무.** 형식:
  ```python
  obs_attacker = ...   # shape: (B, D_a)  B=num_envs, D_a=attacker obs dim
  delay_buf = ...      # shape: (B, τ_max, D_s) sliding window of defender states
  ```
* **for-loop over envs 금지**. Genesis batched API 또는 `torch.gather/scatter` 사용.
* **NaN/Inf guard**: 모든 RL 보상·obs 출력 직전 `torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)`.
* **norm 안전성**: `x.norm(...)` 직후엔 `.clamp_min(1e-6)` 또는 `+ 1e-8`.
* **device 일치**: 텐서 생성은 무조건 `device=self.device`. CPU↔GPU 암묵 이동 금지.
* **squeeze/unsqueeze보다 `view`/`reshape` 우선**, 차원 의도 명시.

---

## 5. 좌표계 / 단위 (Units)

| 양 | 단위 | 좌표계 |
|---|---|---|
| 위치 | **m** | ENU (X-forward, Y-left, Z-up) |
| 속도 | m/s | ENU world / body — 코드에 명시 |
| 각도 (low-level) | **rad** | — |
| 각도 (UI/log only) | deg | conversion 지점 명시 |
| 각속도 | rad/s | body frame |
| 추력 | **N** | per-motor |
| PWM | **µs** | ESC input |
| 시간 | s (config), step (런타임) | dt = 0.01 s 기본 |

**규칙**:
* NED 변환은 `envs/wrappers/coord_transform.py`에서만. env 내부에서 직접 사인 뒤집기 금지.
* deg/rad 혼용 금지 — 함수 docstring에 단위 명시 필수.
* **하드코딩 금지 목록**: `0.027`, `9.81`, `3.16e-10`, `0.0397` 등 물리 상수는 전부 `DroneParams`에서.

---

## 6. 환경 (Environment) 규칙

* **모든 env는 `BasePursuitEvasionEnv` 상속** (Phase 1에서 정의). 그 외 ABC는 만들지 않는다.
* **τ-delay buffer는 env 책임**. agent는 delayed obs를 받을 뿐, agent가 buffer를 들고 있지 않는다.
* **scripted defender**도 env가 소유 (env가 두 드론을 모두 step). agent는 attacker만.
* env hook 이름 규약 (BaseDroneEnv 패턴 유지):
  - `_init_task_buffers`, `_reset_task`, `_update_task_state`, `_get_task_obs`,
  - `_compute_task_reset`, `_compute_task_crash`, `_compute_task_metrics`
* **observation은 명시적 dict / TensorDict**: `{"attacker": ..., "defender_critic": ..., "g_mission": ...}` — flat tensor concat은 step 직전 1회.

---

## 7. Asymmetric Critic 규약 — **논문 §3.5와 직결**

* Actor는 **deployment-feasible obs만**: `[s^a_t, s^d_{t-τ}, ŝ^d_{t+1:t+H}, g_mission]` (delayed + forecasted only).
* Critic은 **학습 시 privileged**: 위 + `s^d_t` (ground truth, no delay).
* `obs_groups`로 분리:
  ```python
  obs_groups = {
      "actor":  ["policy"],                  # delayed + forecasted
      "critic": ["policy", "critic_priv"],   # + ground-truth defender
  }
  ```
* **추론 시 critic 호출 금지**. eval/deploy 코드 path에서 privileged obs key는 build조차 하지 않는다.
* 위반 시 자동 검증: `tests/test_asymmetry.py`가 deploy 모드에서 critic obs를 만들지 않음을 assert.

---

## 8. Forecaster 규약

* `f_ψ`는 **standalone module** (`agents/forecaster/`). PPO 손실과 분리된 자체 loss로 학습.
* 입력: `[s^a_{t-K:t}, s^d_{t-τ-K:t-τ}, action_{t-K:t-1}]` (slowing/ego trajectory + delayed defender history)
* 출력: `ŝ^d_{t-τ+1:t+H}` (multi-step 예측, DFBT 스타일)
* Loss: `MSE(ŝ^d, s^d_true)` — env에서 ground truth 제공 (학습 시에만)
* **Forecaster gradient는 PPO loss로 흐르지 않는다**. 별도 optimizer.
* Inference 시 forecaster output → actor obs로 **detach** 후 전달.

---

## 9. AMS-DRL Co-training Orchestrator 규약

* **runner 인스턴스는 두 개**: `runner_attacker`, `runner_defender`. 한 process 안에서 alternating.
* **freeze**는 `optimizer.zero_grad` + `param.requires_grad=False`로 명시. ckpt에 freeze 상태 저장.
* **stage 정의는 YAML**:
  ```yaml
  ams_drl:
    stages:
      - id: S0
        train: [attacker]
        defender_policy: scripted_random
        success_gate: attacker_winrate > 0.7
      - id: S1
        train: [defender]
        attacker_policy: ckpt_S0
        success_gate: defender_winrate > 0.5
      - ...
  ```
* **Nash proxy metric**: 각 stage 말미에 직전 stage 정책 vs 현 stage 정책 head-to-head 100 episode → win rate. 0.45-0.55 범위 진입을 수렴 신호로.

---

## 10. Reproducibility

* **Seed 의무**: `gs.init(seed=...)`, `torch.manual_seed`, `np.random.seed`, `random.seed`. seed 미설정 PR은 reject.
* **Config 직렬화**: 학습 시작 시 `cfgs.pkl`로 `(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, drone_config, ams_stage)` 전체 dump.
* **eval은 cfgs.pkl만 보고 재구성**. 코드 경로의 default value에 의존 금지.
* **wandb run name = `<exp>_<timestamp>_<git_sha>`** (git dirty면 suffix `-dirty`).

---

## 11. 자원 / 안전 가이드 (Safety Harness)

* **GPU 사용**: `--max_iterations >= 100`인 학습 실행은 사용자 사전 승인. CI/sanity는 `<= 5`로.
* **Wall-clock guard**: train.py 인자에 `--wall_clock_budget_min` 옵션, 초과 시 자동 ckpt 후 종료.
* **로그 파일 1MB 초과 시** Claude는 `head/tail`만 읽는다. 전체 스캔 금지.
* **`outputs/`, `wandb/`, `logs/`, `runs/` 폴더는 분석 요청 시에만** 특정 파일 접근.
* **대규모 binary 삭제 (`*.bin`, `*.npy`, `*.ckpt` > 100MB)는 사용자 확인** 후 진행.

---

## 12. 테스트 / 검증

* **새 모듈 추가 시 `tests/` 단위 테스트 1건 필수**. 통합 테스트는 phase 종료 시 수행.
* **물리 sanity**: env 변경 PR엔 "drone falls under gravity within 1s" / "hover at 1.0m with PID converges in 3s" 류 명시 sanity assertion 포함.
* **shape contract test**: `tests/test_shapes.py`가 모든 env의 obs/action shape를 assert.

---

## 13. 작업 프로세스 (RPIR)

1. **Research**: 새 reward / obs / 알고리즘은 `docs/papers/` 매핑 정리 후 시작.
2. **Plan**: 큰 변경은 `docs/plans/<feature>.md`에 사전 작성. 항목:
   - 변경 동기, 영향 범위, **물리/이론 sanity check**, rollback plan, 테스트 계획.
3. **Implement**: try/except + `env.close()` (또는 동등 cleanup) 보장. 시뮬레이션 OOM 대비.
4. **Review**: 학습 로그(wandb URL + 핵심 지표 표) 텍스트 요약.

---

## 14. Anti-patterns (금지 목록)

* ❌ `for env_idx in range(num_envs):` (Genesis 배치 무력화)
* ❌ `mass = 0.8` 같은 물리 상수 하드코딩
* ❌ `target` 같은 모호한 식별자 (정확히 `g_mission` or `defender`)
* ❌ Critic이 deploy 시 privileged obs 받기
* ❌ Forecaster gradient가 PPO 손실로 흐르기
* ❌ τ-delay를 agent 내부에서 처리 (반드시 env)
* ❌ `# TODO: ...`만 남기고 PR 머지 (TODO.md에 옮길 것)
* ❌ 단일 PR에 phase 두 개 이상 섞기
* ❌ random seed 미설정
* ❌ `time.sleep`, `print` debug 잔존 코드 commit

---

## 15. 빠른 참조 (Quick Reference)

* 좌표계 / 단위 → §5
* 변수 이름 → §3
* env 추상화 책임 → §6
* asymmetric critic 정의 → §7
* AMS-DRL stage 정의 → §9

---

*이 문서를 수정하는 PR은 제목에 `[CLAUDE.md]` 태그 + 본문에 변경 이유 + 사용자 명시 승인 표시.*
