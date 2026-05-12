# Phase 1 — Terminal reward minimax 복원 (per-side crash/oob split)

## 변경 동기

기존 `R_crash = -1.0` 구조는 `crash_buf = a_crash | d_crash`로 양쪽 추락을 한
buffer에 합쳐서 처리. 결과:

- defender가 chase 중 추락해도 **attacker가 R_crash=-1 페널티**를 받음.
  → minimax 위반: 상대 추락이 자기 페널티가 됨.
- attacker가 추락해도 defender가 동일 페널티를 받음.
  → defender 학습 신호에 노이즈 (특히 PR-D 좁은 spawn에서 빈번).

§3 게임 이론적으로 자기 추락(self crash)과 상대 추락(opponent crash)은 다른
사건이며, 보상 부호도 달라야 한다. PR-D3 reward 비대칭(mission ±3, capture ±1)
까지 결합되면 부등식 가드가 없을 때 자살 익스플로잇이 가능해진다:

- attacker가 "잡힐 것 같으면 자살" — `R_self_crash > R_captured`이면 발생.
- defender가 "mission 막을 자신 없으면 자살" — `R_self_crash > R_mission_failed`이면 발생.

따라서 (a) crash/oob를 per-side로 분리하고, (b) `R_self_crash <= min(R_captured,
R_mission_failed)`, `R_opp_crash <= min(R_capture_win, R_mission_win)`을
init-time assert로 강제한다.

## 영향 범위

- `envs/base_pe_env.py`
  - `crash_buf, oob_buf` → per-side 4개 buf로 분리
  - `_compute_crash`, `_compute_oob` 추상 메서드 시그니처: `Tensor` →
    `tuple[Tensor, Tensor]` (attacker, defender)
  - `step()` priority 해결: per-side flag와 합산 flag 동시 산출
  - `last_term`: 합산 `crash`/`oob` 키 + per-side `att_crash`/`def_crash`/
    `att_oob`/`def_oob` 키 (eval.py backward-compat)
  - `_update_episode_extras`: 합산 + per-side 키 동시 노출
- `envs/pe_1v1/env.py`
  - `R_crash` → `R_self_crash`, `R_opp_crash` 분리, init assert 4개
  - `_compute_crash`, `_compute_oob` tuple 리턴
  - `_terminal_reward_attacker/_defender`: self_term / opp_term 두 갈래로
  - `_defender_reward`: r_goal_protect 보류 + Phase 1.5 TODO 주석 추가
- `configs/envs/pe_1v1_default.yaml`, `configs/envs/pe_1v1_phase15.yaml`
  - `R_crash: -1.0` 삭제
  - `R_self_crash: -3.0`, `R_opp_crash: 0.5` 추가
- `tests/sanity/test_reward_zero_sum.py`
  - mission/capture/timeout zero-sum 검증 유지
  - crash bucket 합산 zero-sum 검증 → per-side minimax 부호 매트릭스로 교체
  - `test_constants_invariants` 추가 (yaml 부등식 노출 검증)

## 물리/이론 sanity check

- 부등식 ① `R_self_crash(-3) <= R_captured(-1)` ✓ (-3 < -1)
- 부등식 ② `R_self_crash(-3) <= R_mission_failed(-3)` ✓ (등호)
- 부등식 ③ `R_opp_crash(0.5) <= R_capture_win(1)` ✓
- 부등식 ④ `R_opp_crash(0.5) <= R_mission_win(3)` ✓

`R_self_crash = -3` 이 학습 초기 보상 분포에 -3 spike를 만든다는 우려는
의식적으로 수용. PPO advantage normalize가 흡수하며, defender pretrain
수렴 후엔 crash 빈도가 자연 감소해 spike도 소멸.

## Rollback plan

- 코드 변경은 `git revert <commit>`로 단일 커밋 되돌림.
- yaml은 commit과 함께 묶어 같이 revert.
- 보상 분포가 바뀌어 기존 ckpt와 호환 깨짐 → **별도 `logs_v2/` 폴더로 학습 시작**.
  legacy `logs/` 폴더는 보존 (PR-D3까지의 정책 백업).

## 테스트 계획

1. `pytest tests/sanity/` — termination priority, capture termination,
   reward zero-sum 모두 grean.
2. `python scripts/eval.py` 빠른 sanity (counts['crash']/['oob'] 정상 동작).
3. 50k step 짧은 run (logs_v2/) — 사용자 별도 검증:
   - `att_crash_rate`, `def_crash_rate` 분리 로깅 확인
   - defender ep_return mean 양수 전환 (chase coef 그대로) 여부
   - `dist_to_attacker_defender` 평균 감소 추세

## 비고

`r_goal_protect` (defender-side dense, attacker r_goal_dense의 zero-sum
mirror)는 **이번 PR에서 제외**. 이유: defender obs에 `g_mission`이 없어
actor가 reward 인과를 추론하지 못함 (free energy). zero-sum critic 회복은
이론적 이점이지만 actor 학습 효과 미입증. Phase 1.5 attacker 학습 진입 시
defender obs schema 확장과 함께 재검토 (CLAUDE.md §1 — obs schema 변경은
사용자 명시 승인 필수). `_defender_reward` 본문에 `TODO(phase15)` 주석으로
회수 위치 표시.
