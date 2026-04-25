# TODO.md — Genesis Pursuit-Evasion (논문 구현 로드맵)

> **이 문서는 진척 관리만.** 코딩 규칙은 [CLAUDE.md](./CLAUDE.md).
> 완료 항목은 `[x]` 체크. 우선순위: 🔴 높음 / 🟡 중간 / 🟢 낮음.
> Phase 간 dependency 위반 PR은 reject.

---

## 📌 논문과의 매핑 (v2 결정사항)

* PoV: **attacker** drone 학습 (defender = AMS-DRL co-trained)
* 핵심 3축: (i) attacker PoV + goal-conditioned PETN, (ii) asymmetric τ-step comm delay, (iii) direct forecaster + asymmetric privileged critic
* 주요 참고: AMS-DRL [13] (§3.6 framework), Sánchez Roncero [1] (가장 가까운 경쟁작), DFBT [3] (forecaster baseline), SimpleFlight [25] (sim-to-real best practices), Lambrechts/Ebi (asymmetric critic 이론)
* 자세한 인용 목록 → `paper/section3_4_v8.md`, `paper/section4_v3.md`

---

## ⚠️ 검증 의무 (Phase 0 전 결론 필요)

> 이 셋이 NO이면 phase plan 자체를 갈아엎어야 한다.

* [ ] 🔴 **V1**: Genesis가 한 `Scene` 안에서 두 `Drone` entity의 batched parallel propeller control을 지원하는가?
  * 검증 방식: `tests/sanity/two_drones_in_one_env.py` — 8192 envs, 각 env에 attacker+defender 두 entity, 둘 다 hover RPM 줘서 모두 z=1.0 안정 호버 확인.
  * 실패 시 plan B: env 두 개를 따로 돌리고 state synchronizer로 연결 (Phase 1 재설계).
* [ ] 🔴 **V2**: 같은 process에서 `OnPolicyRunner` 두 개 (attacker, defender)를 alternating으로 돌릴 때 GPU 메모리/optimizer state가 충돌하지 않는가?
  * 검증: 작은 toy env (8 envs)에서 두 runner 각각 5 iter씩 alternating 5회.
* [ ] 🔴 **V3**: `obs_groups`로 actor/critic obs 분리 시 critic만 privileged tensor 받게 할 수 있는가?
  * 검증: HoverEnv 위에서 critic_priv 키 추가, `runner.actor_critic.evaluate(...)` 호출 시 privileged 값이 흘러가는지 print 확인.

---

## Phase 0 — 새 레포 부트스트랩 + 자산 이전 🔴

> **목표**: 기존 single-agent quadrotor 레포의 핵심 자산만 살려 깨끗한 출발점 확보.

### 0.1 레포 인프라
* [ ] 🔴 새 GitHub repo 생성 (이름 후보: `genesis-pursuit-evasion`, `gpe-1v1`, 사용자 결정)  ← 사용자 액션
* [x] 🔴 `pyproject.toml` 작성 (Python 3.10, genesis-world, rsl-rl-lib>=5.0, torch, tensordict, wandb, scipy, pyyaml)
* [x] 🔴 `.gitignore` (logs/, wandb/, outputs/, runs/, *.bin, *.ckpt, __pycache__/, .venv/)
* [x] 🟡 `.pre-commit-config.yaml` (ruff, black, isort, mypy strict on `agents/`, `envs/`)
* [x] 🟡 GitHub Actions: lint + smoke test (현재는 lint + pytest collect-only, GPU 필요한 PPO smoke는 Phase 1에서 추가)
* [x] 🟢 `LICENSE` (MIT)
* [x] 🔴 `CLAUDE.md` 커밋 (이 문서의 짝)  ← 파일 존재. git init은 사용자 결정.
* [x] 🔴 `TODO.md` 커밋 (현재 파일)  ← 파일 존재. git init은 사용자 결정.
* [x] 🟢 `README.md` 초안 (한 줄 설명 + 인용 예정 BibTeX placeholder)

### 0.2 기존 레포에서 이전할 자산
* [x] 🔴 `utils/drone_params.py` 그대로 이전
* [x] 🔴 `configs/cf2x/drone_params.yaml`, `configs/nova/drone_params.yaml` 이전 (→ `configs/drones/cf2x.yaml`, `configs/drones/nova.yaml`)
* [x] 🔴 `assets/urdf/nova/`, `assets/meshes/nova/` 이전
* [x] 🟡 `controllers/pid_controller.py`, `controllers/se3_controller.py` 이전 (defender scripted policy 재료)
* [x] 🟡 `tests/test_pid_controller.py`, `test_se3_controller.py` 이전  *(동작 재확인은 Genesis GPU 환경 필요 — V1 sanity와 묶어서 실행)*
* [x] 🟡 `utils/realtime_plotter.py` 이전 (debug 도구)
* [x] 🟢 기존 `envs/`(hover/trajectory/racing/rates) 전부는 **`reference/legacy_envs/`로 격리**. 이번 phase에선 import 안 함. (`tracking`, `wrappers`, `base_drone_env.py`, `__init__.py`까지 포함)

### 0.3 레포 구조 확정 (목표 트리)

```
genesis-pursuit-evasion/
├── CLAUDE.md
├── TODO.md
├── README.md
├── pyproject.toml
├── .pre-commit-config.yaml
├── assets/
│   └── urdf/, meshes/
├── configs/
│   ├── drones/             # nova, cf2x physical params
│   ├── envs/               # 1v1 PE env configs (attacker/defender pose, mission goal range)
│   ├── ams_drl/            # stage definitions YAML
│   └── train/              # PPO hyperparams
├── envs/
│   ├── base_pe_env.py      # BasePursuitEvasionEnv (Phase 1)
│   ├── pe_1v1/
│   │   ├── env.py
│   │   └── cfgs.py
│   ├── delays/
│   │   └── delay_buffer.py # τ-step buffer (Phase 2)
│   └── wrappers/coord_transform.py
├── agents/
│   ├── ppo_runner.py       # rsl-rl wrapper
│   ├── forecaster/         # Phase 3
│   │   ├── transformer.py
│   │   └── trainer.py
│   ├── critic/             # Phase 4 — asymmetric critic의 obs 빌더
│   └── orchestrator/       # Phase 5 — AMS-DRL stage manager
├── controllers/            # PID/SE3 (defender scripted)
├── utils/
├── docs/
│   ├── plans/              # 큰 변경 사전 plan
│   └── papers/             # 외부 paper별 구현 노트
├── paper/                  # 사용자 작성 노트 (read-only)
├── scripts/
│   ├── train_attacker.py
│   ├── train_defender.py
│   ├── train_ams_drl.py
│   ├── eval.py
│   └── sanity/             # one-off 검증 스크립트
├── tests/
│   ├── test_shapes.py
│   ├── test_asymmetry.py
│   └── sanity/
└── reference/
    └── legacy_envs/        # 기존 single-agent envs (보존만, import 금지)
```

* [x] 🔴 위 구조 commit. 빈 폴더는 `.gitkeep`. *(파일시스템 트리 생성 + .gitkeep 완료. `git init` 및 첫 commit은 사용자 결정.)*

---

## Phase 1 — 1v1 환경 골격 🔴

> **목표**: 한 env 안에서 attacker + scripted defender 동시 simulate. delay/forecaster 없는 perfect-info baseline 작성.

### 1.1 사전 검증 (Phase 0의 V1, V2, V3 통과 후 시작)
* [ ] 🔴 V1/V2/V3 모두 ✅ 확인되었는지 PR 본문에 명시

### 1.2 BasePursuitEvasionEnv
* [ ] 🔴 `envs/base_pe_env.py` 작성
  * 두 `Drone` entity 소유: `self.attacker_drone`, `self.defender_drone`
  * 두 드론 공통 step (Genesis batched)
  * obs dict 출력: `{"attacker": ..., "defender_priv": ..., "g_mission": ...}`
  * 추상 hook: `_attacker_obs`, `_defender_obs`, `_attacker_reward`, `_defender_reward`, `_compute_capture`, `_compute_mission_reached`
* [ ] 🔴 종료 조건 정의:
  * `mission_reached`: attacker가 g_mission 반경 내 도달 → attacker win
  * `captured`: attacker-defender 거리 < d_capture → defender win
  * `out_of_bounds`, `crash`, `timeout`
* [ ] 🟡 reward 기본 형태 (논문 §3.6 따라):
  * Attacker: `+r_progress(g_mission) − r_threat(defender) + r_terminal_win`
  * Defender: `−r_distance(attacker) + r_terminal_capture`

### 1.3 1v1 PE Env 구현
* [ ] 🔴 `envs/pe_1v1/env.py` (BasePursuitEvasionEnv 상속)
* [ ] 🔴 `envs/pe_1v1/cfgs.py` — 초기 spawn 분포, g_mission 분포, dt, episode_length
* [ ] 🔴 **Scripted defender**: 처음엔 stationary, 다음 random walk, 다음 PID-greedy chase (controllers/pid_controller.py 활용). 모드는 cfg로 토글.

### 1.4 Attacker 학습 (perfect info)
* [ ] 🔴 `agents/ppo_runner.py` — rsl_rl `OnPolicyRunner` thin wrapper. 단일 정책 학습용.
* [ ] 🔴 `scripts/train_attacker.py` — `--defender scripted_{stationary,random,pid_chase}` 옵션
* [ ] 🔴 학습 sanity:
  * stationary defender 대해 attacker가 g_mission 도달 winrate > 0.9
  * pid_chase에 대해서도 winrate > 0.5
* [ ] 🟡 reward shaping ablation 1회 (progress vs sparse)

### 1.5 Tests
* [ ] 🔴 `tests/test_shapes.py` — 1v1 env의 obs/action shape assert
* [ ] 🟡 `tests/sanity/test_capture_termination.py`

---

## Phase 2 — Asymmetric τ-step Communication Delay 🔴

> **목표**: env 내부에 delay buffer 도입. attacker는 `s^d_{t-τ}`만 관측. forecaster 없이 naive delayed-obs baseline.

### 2.1 Delay Buffer
* [ ] 🔴 `envs/delays/delay_buffer.py` — `DelayBuffer(B, τ_max, D)` ring buffer
  * `push(s_d_t)`, `query(τ)` → `s_d_{t-τ}`
  * τ는 환경 reset 시 cfg에서 sampling (constant or [τ_min, τ_max])
* [ ] 🔴 BasePursuitEvasionEnv에 buffer 통합 — **agent에는 절대 buffer 노출 금지**
* [ ] 🔴 reset 시 buffer를 `s^d_{t=0}` 복제로 채우기 (warmup 없는 첫 τ step의 garbage 방지)

### 2.2 Attacker obs 변경
* [ ] 🔴 `obs["attacker"]`에 `s^d_{t-τ}` 포함, `obs["defender_priv"]`에 `s^d_t` (ground truth)
* [ ] 🔴 critic은 `obs_groups`로 priv 접근, actor는 delayed만

### 2.3 Naive Baseline
* [ ] 🔴 τ ∈ {0, 5, 10, 20} step ablation 학습 (각 5 seed)
* [ ] 🔴 winrate vs τ 곡선 wandb로 시각화 — 예상: τ↑일수록 winrate↓ (degradation 확인)

### 2.4 Tests
* [ ] 🔴 `tests/test_delay_buffer.py` — push/query 정합성, reset 동작
* [ ] 🔴 `tests/test_asymmetry.py` — deploy 모드에서 critic priv obs가 build되지 않음 assert (CLAUDE.md §7 강제)

---

## Phase 3 — Direct Forecaster (DFBT 스타일) 🔴

> **목표**: transformer forecaster `f_ψ`로 `ŝ^d_{t-τ+1:t+H}` 예측. PPO와 분리된 자체 loss.

### 3.1 Forecaster 모델
* [ ] 🔴 `agents/forecaster/transformer.py`
  * 입력: `[s^a_{t-K:t}, s^d_{t-τ-K:t-τ}, a_{t-K:t-1}]`
  * 출력: `ŝ^d_{t-τ+1:t+H}` (multi-step)
  * 작은 transformer (4 layer, d_model=128) 시작
* [ ] 🔴 `agents/forecaster/trainer.py`
  * 자체 optimizer (PPO와 분리)
  * Loss: MSE on rolling window. env에서 ground truth `s^d_t`를 제공받음.
  * **Forecaster gradient는 PPO 손실로 흐르지 않음** — `torch.no_grad()` 또는 `detach()` 명시

### 3.2 Env 통합
* [ ] 🔴 BasePursuitEvasionEnv가 forecaster를 hold (선택적). attacker obs에 `ŝ^d` 포함.
* [ ] 🔴 학습 schedule: 처음 N step은 forecaster freeze (cold start) → MSE 일정 임계 도달 시 PPO에 노출

### 3.3 Baseline 비교
* [ ] 🔴 DFBT [Wu 2025] 재현 baseline (uniform delay, full obs) 동일 env에서 측정
* [ ] 🔴 Naive (Phase 2) vs DFBT vs Ours forecaster — winrate / forecast MSE / sample efficiency 비교

### 3.4 Tests
* [ ] 🟡 `tests/test_forecaster.py` — 입출력 shape, gradient isolation 검증

---

## Phase 4 — Asymmetric Privileged Critic 🔴

> **목표**: critic이 학습 시 `s^d_t` (no delay, no forecast) 접근. actor는 forecast/delayed만.

### 4.1 Critic obs builder
* [ ] 🔴 `agents/critic/obs_builder.py`
  * actor obs와 critic obs를 분리해서 build
  * `obs_groups = {"actor": ["policy"], "critic": ["policy", "critic_priv"]}` 적용
* [ ] 🔴 deploy/eval 코드 path에서 `critic_priv` key가 build되지 않음을 검증하는 테스트

### 4.2 학습
* [ ] 🔴 Phase 3 forecaster + Phase 4 critic 결합 학습
* [ ] 🔴 Ablation: symmetric critic (priv 없음) vs asymmetric. value loss 분산 / sample efficiency 비교
* [ ] 🟡 Lambrechts 2025 / Ebi 2025의 이론적 정당화 항목을 paper §3.5에 매핑한 `docs/papers/asymmetric_critic_theory.md` 작성

### 4.3 Tests
* [ ] 🔴 `tests/test_asymmetry.py` 확장: training mode에서는 priv 흐름, eval mode에서는 차단

---

## Phase 5 — AMS-DRL Co-training Orchestrator 🔴

> **목표**: defender도 학습. bipartite alternating 단계 구성. Nash 수렴 proxy.
> **주의**: NTU-ICG 레포에는 deployment 코드만 있음. 알고리즘은 논문 §III에서 직접 구현.

### 5.1 정책 두 벌 동시 운용
* [ ] 🔴 `scripts/train_ams_drl.py`
  * `runner_attacker`, `runner_defender` 두 인스턴스
  * 한 env step에서 둘 다 행동, 보상 분리
  * stage별 freeze toggle

### 5.2 Stage 정의
* [ ] 🔴 `configs/ams_drl/stage_default.yaml`
  ```yaml
  stages:
    - id: S0
      train: [attacker]
      defender_policy: scripted_random
      success_gate: attacker_winrate > 0.7
      max_iters: 500
    - id: S1
      train: [defender]
      attacker_policy: ckpt_S0
      success_gate: defender_winrate > 0.5
      max_iters: 500
    - id: S2_alternating
      train: [attacker, defender]   # alternating per checkpoint
      switch_every: 50
      max_cycles: 20
      nash_gate: |winrate - 0.5| < 0.05 over last 5 cycles
  ```
* [ ] 🔴 stage 매니저 구현 (`agents/orchestrator/stage_manager.py`)

### 5.3 Nash proxy metric
* [ ] 🔴 stage 종료 시 head-to-head 100 episode (직전 stage 정책 vs 현 stage 정책) → win rate
* [ ] 🔴 wandb 로그에 Nash gap, win rate, episode length distribution

### 5.4 Tests / 안정성
* [ ] 🟡 collapse 케이스 검증 — defender가 attacker spawn 위치 점유 → attacker 학습 정체. **defender spawn 제약** 추가.
* [ ] 🟡 win rate 50/50 plateau 외에 "둘 다 진다" 케이스 (timeout draw) 비율 모니터링

---

## Phase 6 — 평가 / 비교 실험 🟡

> **목표**: 논문 §5 실험 데이터 확보.

### 6.1 Baseline 재현
* [ ] 🟡 Sánchez Roncero 2025 [1] 재현 — body-rate RL, AMSPB. 단 PoV 반대(pursuer 학습) → 우리 env에서 evader 자리에 학습. AMSPB은 그들의 코드/논문 따라 직접 구현 (코드 비공개).
* [ ] 🟡 DFBT [3] uniform-delay full-obs baseline (Phase 3에서 일부 진행)
* [ ] 🟡 AMS-DRL [13] no-delay (perfect info) baseline (Phase 5의 S0~S2를 delay=0으로)

### 6.2 Ablation
* [ ] 🟡 ours w/o forecaster, ours w/o asymmetric critic, ours w/o AMS-DRL (defender scripted)
* [ ] 🟡 τ sweep: {0, 5, 10, 15, 20, 30}
* [ ] 🟡 H (forecast horizon) sweep: {1, 3, 5, 10}

### 6.3 결과 정리
* [ ] 🟡 wandb → CSV → matplotlib 일괄 plot 스크립트 (`scripts/plots/`)
* [ ] 🟡 paper §5 표/그래프 업데이트

---

## Phase 7 — Sim-to-Real (선택, future) 🟢

> **목표**: Crazyflie deployment. SimpleFlight [25] 5대 요소 적용.

* [ ] 🟢 SimpleFlight Factor 1: actor input에 velocity + rotation matrix
* [ ] 🟢 SimpleFlight Factor 2: critic input에 time vector
* [ ] 🟢 SimpleFlight Factor 3: action difference regularization
* [ ] 🟢 SimpleFlight Factor 4: selective SysID + DR
* [ ] 🟢 SimpleFlight Factor 5: large batch
* [ ] 🟢 ROS bridge (Crazyflie firmware ↔ trained policy)
* [ ] 🟢 motion capture (Vicon/OptiTrack) 통합
* [ ] 🟢 단순화된 1v1 실기 비행 (defender = manual or pre-recorded trajectory)

---

## Phase 8 — 문서 / 재현성 / 공개 🟢

* [ ] 🟢 README full version (논문 abstract, citation, 재현 명령)
* [ ] 🟢 `docs/reproduction.md` (전체 phase 재현 명령)
* [ ] 🟢 학습된 모델 weights public release (HuggingFace / GitHub Releases)
* [ ] 🟢 paper preprint arXiv 업로드 후 BibTeX 추가

---

## ✅ 완료 (Done)

> 작업 완료 후 위에서 옮겨오기. 처음엔 비어있음.

* [x] 비판적 검토 + 새 레포 결정 (2026-04-25)
* [x] 기존 레포 자산 분류 (보존 vs 폐기) 정책 합의
* [x] CLAUDE.md, TODO.md 초안 작성
* [x] **Phase 0 부트스트랩 완료** (2026-04-25)
  - 디렉토리 트리 + `.gitkeep` (agents/{forecaster,critic,orchestrator}, envs/{pe_1v1,delays,wrappers}, configs/{drones,envs,ams_drl,train}, scripts/sanity, tests/sanity, docs/{plans,papers}, paper/, reference/legacy_envs/)
  - 인프라: `pyproject.toml`, `.gitignore`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `LICENSE`, `README.md`
  - 자산 이전: `utils/{drone_params, realtime_plotter}.py`, `controllers/{pid,se3}_controller.py`, `tests/test_{pid,se3}_controller.py`, `configs/drones/{cf2x,nova}.yaml`, `assets/{urdf,meshes}/nova/`
  - `reference/legacy_envs/`: `hover, trajectory, racing, rates, tracking, wrappers, base_drone_env.py` 격리 보존
  - 검증: TOML/YAML 파싱 OK, 마이그레이션된 6개 .py 파일 AST parse OK
  - **남은 Phase 0 외부 액션 (사용자만 가능)**: GitHub repo 생성, `git init` + 첫 commit, `pip install -e ".[dev]"`로 dev 환경 구축
  - **Phase 1 진입 전 의무**: V1, V2, V3 sanity 통과 (TODO ⚠️ 섹션)

---

## 📎 외부 참고

* AMS-DRL 논문: Xiao & Feroskhan, IEEE TAI 2024 ([13] in `paper/section3_4_v8.md`) — **레포는 deployment-only, 학습 코드 직접 구현 필요**
* Sánchez Roncero 2025 ([1]): 가장 가까운 경쟁작, KTH
* DFBT [3], Wu 2025 ICML: forecaster 원조
* SimpleFlight [25], Chen 2025 RA-L: sim-to-real best practices
* Lambrechts 2025 [4], Ebi 2025 [5]: asymmetric critic 이론
* 전체 28편 → `paper/section3_4_v8.md`

---

*작업 시작 전 Phase의 V (Validation) 항목 먼저 ✅. Phase 간 dependency 위반 PR은 reject.*
