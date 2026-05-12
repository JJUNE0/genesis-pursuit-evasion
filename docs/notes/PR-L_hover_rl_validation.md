# PR-L — Hover RL validation (rate-mode action 진짜 검증)

작업일: 2026-04-29
선행: PR-I/J/K (controller 게인 튜닝, hover stability, trajectory tracking)

## 변경 동기

PR-K trajectory tracking까지 완료했지만 사용자 viewer 피드백:
- "viewer에서 보니까 모터가 회전하는게 아니라 까딱까딱하는데 뭔가 이상하네ㅋㅋ"
- "여기서 제어기 검증하는건 의미 없을 것 같고, action을 rate로 바꿨을 때 잘되는
  코드로 검증하는 걸로 바꿔야겠다 우리 목적을"

즉 PR-I/J/K의 standalone controller validation은 **의미 부족**. 진짜 검증은
"BodyRateController 위에서 RL 정책이 실제로 학습되는가"이다. 이를 위해 단순
hover task RL 학습 인프라를 구축.

참고: `reference/legacy_envs/hover/env.py` (legacy single-drone hover RL),
`scripts/test/{train,eval}.py` (legacy task-pluggable launcher).

## 작업 범위

### Phase 1 — Standalone HoverEnv (단일 드론, rate-mode action)

신규 파일:
- `envs/hover/__init__.py`
- `envs/hover/env.py` — `HoverEnv` 클래스 (~340 LOC)
- `envs/hover/cfgs.py` — `get_cfgs()` (env / obs / reward / command 4 dict)

`HoverEnv` 핵심:
- single drone (`gs.Scene` + `gs.morphs.Drone`).
- `action_mode = "rate"` (default) → BodyRateController 통합. `"rpm"` fallback.
- obs (22 dim): `rel_pos(3) + rot_mat(9) + lin_vel_body(3) + ang_vel_body(3)
  + last_action(4)`.
- reward: target / hover / lin_vel / smooth / yaw / angular / crash (legacy
  hover env과 동일 패턴 + dt 스케일링).
- termination: roll/pitch/floor/xy/z OOB + crash.
- random spawn + random target (yaml 설정).
- rsl-rl 5.x interface: `cfg`, `obs_groups`, `num_actions`, `num_envs`,
  `device`, `reset()`, `step(actions)`, `get_observations()`.

cfg defaults:
- dt=0.005, episode_length_s=5.0 → max_steps 1000.
- spawn x∈[-0.5,0.5] y∈[-0.5,0.5] z∈[0.8,1.2].
- target x∈[-1,1] y∈[-1,1] z∈[0.8,1.5].
- BodyRateController: PR-I 게인 그대로 (kp=[0.05,0.10,0.10], kd=[1e-4,5e-5,1e-4]).

### Phase 2 — Train / eval 스크립트

- `scripts/train_hover.py` — rsl-rl `OnPolicyRunner`, PPO. defaults:
  num_envs=4096, max_iterations=500, logger=tensorboard.
- `scripts/eval_hover.py` — viewer 모드, ckpt 로드 후 1 episode 시각화.

기존 `agents/ppo_runner.py`의 `make_runner / make_run_name / dump_cfgs_pkl`
재사용 → cfgs.pkl 자동 dump (CLAUDE.md §10 reproducibility).

### Phase 3 — Sanity (코드 동작 확인 — 학습은 사용자 GPU에서)

- `pytest tests/` **40/40** 통과 (회귀 0).
- HoverEnv build + 50 step (action=0) sanity:
  - obs shape `(4, 22)` ✓
  - rew finite, done=0 ✓
  - 50 step 후 drone이 hover thrust로 거의 제자리 (action=0 이라 BodyRate
    controller가 hover_rpm 출력) ✓

## 변경 안 한 것 (의도적)

- `utils/rate_controller.py` — controller 그대로.
- `configs/envs/pe_1v1_*.yaml` — PE 1v1 학습 영향 없음.
- 기존 `envs/pe_1v1/`, `envs/base_pe_env.py` — pursuit-evasion 학습 무영향.

## 사용자 GPU 액션 (학습 시작)

```bash
# Tensorboard logger (기본)
python scripts/train_hover.py --num_envs 4096 --max_iterations 500

# Wandb
python scripts/train_hover.py --num_envs 4096 --max_iterations 500 \
    --logger wandb --exp_name hover_rate_v0

# Baseline 비교 (action_mode = rpm, legacy fallback)
python scripts/train_hover.py --num_envs 4096 --max_iterations 500 \
    --action_mode rpm --exp_name hover_rpm_v0

# 학습 후 viewer 평가
python scripts/eval_hover.py
```

## 학습 성공 기준 (사용자 검토 필요)

| 지표 | 목표 (≥ 500 iter 학습 후) |
|---|---|
| 평균 episode reward | 양수 안정 |
| dist_to_target_mean | < 0.2m |
| crash_rate | < 5% |
| 평균 episode length | 1000 (max, 안 죽음) |
| 액션 평균 abs | < 0.5 (smooth, saturation 안 함) |

목표 미달 시:
- reward scale 재튜닝 (`hover` ↑ `crash` ↑)
- spawn box 좁히기 (학습 초반 task 단순화)
- `action_mode: "rpm"` baseline과 비교해서 rate-mode 자체 issue인지 확인

## 미해결 / 향후

1. **Asymmetric critic**: 현재 `obs_groups = {"actor":["policy"], "critic":["policy"]}`로
   동일. PE 1v1 진입 시 (Phase 4) critic_priv 추가.
2. **τ-delay buffer**: 단일 drone task라 적용 안 함. PE 1v1 (Phase 2) 진입 시.
3. **학습 안정성 fallback** (학습 실패 시):
   - max_body_rate ±π → ±π/2 축소 (saturation 회피)
   - motor LPF 추가 (PR-J에서 우려한 sim-to-real 갭)
   - reward scale 재조정

## 비고

이번 PR은 **인프라**만 작성. 실제 학습 + 정책 검증은 사용자 GPU 액션 후 결과를
보고 단계적 진행. PR-I/J/K가 controller 자체의 정량적 properties를 입증했다면,
PR-L은 그 controller 위에서 학습이 작동하는지 **종합 검증**의 시작점.

PE 1v1 학습으로 진입하기 전 hover task로 BodyRateController + RL 결합이
healthy한지 확인 → 결과 좋으면 PE env에 `action_mode: "rate"` 적용 자신있게
진행 가능.
