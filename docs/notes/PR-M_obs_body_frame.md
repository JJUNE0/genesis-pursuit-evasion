# PR-M — Obs schema body-frame 변환 (yaw + translation invariance)

작업일: 2026-04-29
선행: PR-D~G (world frame absolute obs로 학습), PR-L (rate-mode RL 인프라)

## 변경 동기 (사용자 명시 승인 받음)

기존 obs는 자기/상대 drone, g_mission 모두 **world frame 절대 좌표**.
사용자 지적: 상대 위치/속도가 절대값이라 정책이 internally subtract + yaw 회전
변환을 학습해야 함 → 학습 burden + sim-to-real 불리.

해결: 상대 drone과 g_mission을 **ego body frame**으로 변환.
- attacker obs: defender 상대 위치/속도 + g_mission이 attacker body frame
- defender obs: attacker 상대 위치/속도가 defender body frame

자기 위치/속도(`pos_a`, `vel_a`)는 **world frame 그대로 유지**:
- 두 drone + goal 공유 reference 보존
- spatial / position-wise reward 계산 (e.g., progress, dist_to_goal)이 직접 사용

## 변경 사항

### 1. Helper 추가 — `envs/base_pe_env.py`

`quat_to_rotmat_flat` 아래에 `rotate_to_body_frame()` 추가:

```python
def rotate_to_body_frame(vec_world, quat) -> torch.Tensor:
    """World vector → ego body frame: R^T · vec. shape (B, 3)."""
    R = quat_to_rotmat_flat(quat).view(-1, 3, 3)   # (B, 3, 3) body→world
    return torch.bmm(R.transpose(1, 2), vec_world.unsqueeze(-1)).squeeze(-1)
```

### 2. `envs/pe_1v1/env.py::_attacker_obs` 변경

- **self vel** (`vel_a`): world → **body** (`R^T · attacker_vel`)
- `pos_d_kin = defender_pos * scale` → `rel_pos_d = R^T·(defender_pos - attacker_pos) * scale`
- `vel_d_kin = defender_vel * scale` → `rel_vel_d = R^T·(defender_vel - attacker_vel) * scale`
- `g = g_mission * scale` → `rel_g = R^T·(g_mission - attacker_pos) * scale`

obs 차원 동일 (31). 의미만 변경 (world abs → body relative).

### 3. `envs/pe_1v1/env.py::_defender_obs` 변경

- **self vel** (`vel_d`): world → **body**
- `pos_a_kin → rel_pos_a` body frame
- `vel_a_kin → rel_vel_a` body frame

obs 차원 동일 (28). 의미만 변경.

### Self vel을 body frame으로 한 추가 이유

- action이 body frame (`[T_norm, wx_ref, wy_ref, wz_ref]`)이라 obs도 body
  representation 일관성. 정책이 "내가 forward로 얼마나 가고 있나" 직접 obs.
- yaw rotation invariance 강화 (자기 + 상대 모두 body).
- SimpleFlight Factor 1 (Chen 2025 RA-L) 권장 패턴.
- HoverEnv (`envs/hover/env.py`)에서 사용한 frame과 일관.

### 최종 obs frame 정리

| | Attacker | Defender |
|---|:---:|:---:|
| pos (자기) | world | world |
| **vel (자기)** | **body** | **body** |
| rot_mat | body→world | body→world |
| ang_vel (자기) | body | body |
| rel_pos (상대) | body | body |
| rel_vel (상대) | body | body |
| rel_g | body | — |
| last_action | (4) | (4) |
| 합 | 31 | 28 |

## Sanity 검증

### Helper unit test

```
identity quat: [1, 2, 3] → [1, 2, 3]   ✓
yaw 90° (q = [cos45, 0, 0, sin45]):
  world +x → body [0, -1, 0]   ✓ (수학 일치)
```

### 전체 회귀

`pytest tests/` 40/40 통과. shape contract test (`tests/test_shapes.py`)는 차원
검사라 body frame 변환에도 통과 — obs 의미만 바뀌고 31/28 그대로.

## 영향 범위

### Code

| 파일 | 변경 | LOC |
|---|---|---:|
| `envs/base_pe_env.py` | `rotate_to_body_frame` 신규 helper | +9 |
| `envs/pe_1v1/env.py` | `_attacker_obs`, `_defender_obs` 본문 교체 | ~+20 |

### Ckpt 호환성 — **BROKEN**

obs 의미 (값) 바뀌었으므로 PR-D~G까지 학습된 정책은 더 이상 valid 입력 못 받음.
같은 obs vector 위치에 다른 의미가 들어가서 정책이 entirely 다른 행동 출력.

**필수 조치**: 단계 3 (rate mode) 학습은 **fresh init**.
- `logs_v2_rate/` 신규 폴더 사용
- 기존 `logs/pe1v1_v0/{attacker,defender}_v0/` 그대로 보존 (rpm + world abs obs 정책)

CLAUDE.md §1 "체크포인트 호환성 깨지는 변경 금지. 깨야 한다면 별도 폴더로 분리"
원칙에 따라 logs/ 분리.

## 학습 영향 예상

### 개선 예상
1. **Yaw rotation invariance**: 정책이 drone yaw에 무관 (같은 maneuver를 어느
   yaw 방향이든 같은 행동으로 학습). 학습 sample efficiency ↑.
2. **Translation invariance**: g_mission이 ego 기준 상대 위치 → spatial
   absolute에 의존 안 함.
3. **Sim-to-real 친화** (Phase 7): 실기에서 절대 좌표는 GPS / mocap 의존.
   body frame은 onboard IMU + relative sensor만으로 가능.

### 잠재 우려
- `pos_a` (자기 world) 와 `rot_a` (body→world)는 그대로라 정책이 절대 좌표
  부분도 학습 — 개선 일부 상쇄. 하지만 spatial reward (mission distance,
  capture distance)가 world frame이라 쓸모 있음.
- 학습 중 발산 시 obs scaling 재조정 필요할 수 있음 (rel_pos는 max 5m이지만
  rel_vel은 v_rel = v_d - v_a라 절대 v_max 10 m/s의 2배까지 가능 → scale 재검토).

## 사용자 GPU 액션 (다음 단계)

```bash
# Defender pretraining (Phase 1.0)
python scripts/train_defender.py --num_envs 1024 --max_iterations 5000 \
    --seed 0 --logger wandb --exp_name defender_v1_rate_body

# Attacker (Phase 1.5)
python scripts/train_attacker.py --defender pretrained \
    --defender_ckpt logs/pe1v1_v1_rate_body/defender_v1_rate_body/.../actor_v1.pt \
    --num_envs 1024 --max_iterations 5000 --seed 0 --logger wandb \
    --exp_name attacker_v1_rate_body
```

기대 효과:
- 이전 PR-D~G 학습보다 sample efficiency ↑
- mean_std 폭주 완화 (정책이 deterministic해질 incentive ↑)
- 실제 학습 결과로 검증 필요

## 변경 안 한 것

- obs 차원 (31 / 28 그대로) — shape contract 안전
- PE env termination / reward / spawn — 무수정
- `utils/rate_controller.py` — 무수정
- `tests/` — pass (shape only)
- HoverEnv (`envs/hover/`) — 단독 hover라 무관

## 비고

PR-M은 PR-L 검증 (rate RL 학습 가능)을 발판 삼은 obs schema 개선. 학습 한 번
더 돌려서 결과 비교 가능.
