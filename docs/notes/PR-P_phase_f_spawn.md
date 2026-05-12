# PR-P — Phase F: spawn 확대 + dense reward alpha 완화

작업일: 2026-04-30
선행: PR-O squashed Gaussian + Phase A/B/C/D/E reward symmetrization 사이클.

## 동기 — ams_s2_v6 attacker 압도 collapse

`logs/pe1v1_v0/ams_s2_v6` Phase E 학습 13 cycles:

| cyc | att | def | \|Δ\| |
|---:|---:|---:|---:|
| 1 | 0.807 | 0.072 | 0.735 |
| **5** | **0.951** | 0.030 | **0.921** |
| 13 | 0.800 | 0.047 | 0.753 |

attacker 시작부터 압도 (이전 ams_s2_v[2-4]는 정반대 — defender 압도). Phase E
대칭 reward + squashed actor + Phase 1.0/1.5 baseline (mission winrate 0.55)
조합으로 attacker가 진짜 capability 발휘 → 짧은 spawn에선 물리적 자연 우세.

### 거리/시간 분석

현재 (Phase E) spawn:
- attacker x=[0.5, 1.5]   → spawn center 1.0
- defender x=[-1.5, -0.5] → spawn center -1.0
- g_mission x=[-3.0, -2.0] → goal center -2.5
- attacker → goal 거리 ≈ 3.5m
- v_max=10m/s → 직진 시 mission 도달 시간 ≈ **0.35s**
- defender → attacker 거리 ≈ 2m, chase 시 ≈ 0.2s + maneuver overhead

즉 attacker가 직진해버리면 defender가 chase 자체 못 따라잡음. **물리적 race
condition** — reward shaping 무관.

## 처방 — 거리 확대 (TODO §5.6)

### Spawn 확대

| 항목 | Phase E | Phase F |
|---|---|---|
| `attacker_init_pos.x` | [0.5, 1.5] | **[2.0, 4.0]** |
| `defender_init_pos.x` | [-1.5, -0.5] | **[-4.0, -2.0]** |
| `g_mission.box.x` | [-3.0, -2.0] | **[-6.0, -4.0]** |
| `termination.bound_xyz` | 5.0 | **10.0** |

attacker → goal 거리 ≈ **3.5m → 8m (2.3×)**. 직진 mission 시간 ≈ **0.35s → 0.8s**.
defender chase 시간 충분 + maneuver 여유.

### Dense alpha 완화

거리 확대 시 chase / goal_dense 신호가 멀리서 거의 0이 됨. coef는 그대로
유지하고 **alpha만 완화**해서 신호 거리 유지:

| coef | Phase E | Phase F | dist=8m 시 신호 비교 |
|---|---|---|---|
| `chase_alpha_attacker` | 1.0 | **0.3** | exp(-8) → exp(-2.4): **0.0003 → 0.091 (25×)** |
| `chase_alpha_defender` | 0.5 | **0.3** | exp(-4) → exp(-2.4): 0.018 → 0.091 (5×) |
| `goal_dense_alpha` | 0.5 | **0.2** | exp(-4) → exp(-1.6): 0.018 → 0.20 (11×) |

dist=2m (가까울 때):
- chase_alpha 0.3, dist=2: exp(-0.6) = 0.55
- goal_dense 0.2, dist=2: exp(-0.4) = 0.67
- 가까울 때도 신호 유지 (decay 매우 완만)

### 변경 안 한 것 — 변수 격리

Phase E 그대로:
- `chase_coef_attacker_*`, `chase_coef_defender_*` (per-side coef)
- `goal_dense_coef_*`
- `R_*` terminal (-3/+3 대칭)
- `omega_coef_*`, `action_smooth_*`, `omega_clamp_max_sq`
- `enable_wrench_saturation`
- train yaml (squashed Gaussian)

즉 **변경 항목 4개만** (spawn 3 + alpha 3개 — 6 변수). 학습 결과 비교 시
spawn/alpha 효과 분리 측정 가능.

## 검증

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from envs.pe_1v1.cfgs import load_pe_1v1_cfg, make_env_cfg, make_command_cfg, make_reward_cfg
raw = load_pe_1v1_cfg('configs/envs/pe_1v1_phase_f.yaml')
env_cfg = make_env_cfg(raw, ego='attacker')
cmd_cfg = make_command_cfg(raw)
rew_cfg = make_reward_cfg(raw)
print(f'bound_xyz: {env_cfg[\"bound_xyz\"]}')
print(f'spawn att x: {cmd_cfg[\"spawn\"][\"attacker_init_pos\"][\"x\"]}')
print(f'spawn def x: {cmd_cfg[\"spawn\"][\"defender_init_pos\"][\"x\"]}')
print(f'goal x:      {cmd_cfg[\"g_mission\"][\"box\"][\"x\"]}')
print(f'chase_alpha_attacker: {rew_cfg[\"chase_alpha_attacker\"]}')
print(f'chase_alpha_defender: {rew_cfg[\"chase_alpha_defender\"]}')
print(f'goal_dense_alpha:     {rew_cfg[\"goal_dense_alpha\"]}')
"
```

기대 출력:
```
bound_xyz: 10.0
spawn att x: [2.0, 4.0]
spawn def x: [-4.0, -2.0]
goal x:      [-6.0, -4.0]
chase_alpha_attacker: 0.3
chase_alpha_defender: 0.3
goal_dense_alpha:     0.2
```

## 학습 명령 (사용자 GPU)

### Phase 1.0 — defender (squashed, stationary attacker 상대)

```bash
.venv/bin/python scripts/train_defender.py \
    --train_yaml configs/train/ppo_defender_squash.yaml \
    --env_yaml configs/envs/pe_1v1_phase_f.yaml \
    --attacker_mode stationary \
    --num_envs 1024 --max_iterations 5000 --seed 0 --logger wandb \
    --logs_root logs/pe1v1_v5_phase_f/defender_v0
```

### Phase 1.5 — attacker (squashed, **stationary defender 상대** — Phase 1 명명)

> 2026-04-30 정정: Phase 1.5는 stationary defender만 사용. pretrained
> defender 상대 학습은 Phase 2 (AMS-DRL) cycle 1에서 직접 수행.

```bash
.venv/bin/python scripts/train_attacker.py \
    --train_yaml configs/train/ppo_attacker_squash.yaml \
    --env_yaml configs/envs/pe_1v1_phase_f.yaml \
    --defender stationary \
    --num_envs 1024 --max_iterations 10000 --seed 0 --logger wandb \
    --logs_root logs/pe1v1_v5_phase_f/attacker_v0
```

### Phase 2 (AMS-DRL, ams_s2_v7) — Phase F env + co-training

Phase 1.0/1.5 종료 후 model_*.pt 경로 채워서:

```bash
.venv/bin/python scripts/train_ams_drl.py \
    --attacker_ckpt logs/pe1v1_v5_phase_f/attacker_v0/<run>/attacker_v0.pt \
    --defender_ckpt logs/pe1v1_v5_phase_f/defender_v0/<run>/defender_v0.pt \
    --attacker_resume logs/pe1v1_v5_phase_f/attacker_v0/<run>/model_<n>.pt \
    --defender_resume logs/pe1v1_v5_phase_f/defender_v0/<run>/model_<n>.pt \
    --env_yaml configs/envs/pe_1v1_phase_f.yaml \
    --attacker_train_yaml configs/train/ppo_attacker_squash.yaml \
    --defender_train_yaml configs/train/ppo_defender_squash.yaml \
    --num_envs 1024 --switch_every 50 --max_cycles 30 \
    --n_h2h_episodes 400 --convergence_window 3 --convergence_eps 0.10 \
    --draw_max 0.2 --seed 0 --logger wandb \
    --logs_root logs/pe1v1_v0/ams_s2_v7
```

## 통과 기준

### Phase 1.0 / 1.5 (baseline 재학습)
- Phase 1.0 stationary 상대 captured_rate iter 5000에서 > 0.7 (이전 phase_e 동등 또는 약함 — 거리 멀어 chase 더 어려움)
- Phase 1.5 pretrained defender 상대 mission winrate iter 10000에서 > 0.4 (이전 phase_e 0.55에서 약간 약함 예상)

### ams_s2_v7 (5 cycle smoke)
- c1 |att_wr − def_wr| **< 0.50** (ams_s2_v6는 0.735, 즉 균형 회복 신호)
- 5 cycle 후 어느 한쪽 ≥ 0.85 압도 없음 (집중적 collapse 차단)
- def_crash, att_crash 둘 다 < 10% (controller 처방 효과 유지)

| 시나리오 | 다음 step |
|---|---|
| A — \|Δ\| 안정적 < 0.30 | 본격 학습 (max_cycles 60) |
| B — 한쪽 0.6+ 우세지만 collapse 안 함 | spawn 추가 미세 조정 (예: 5m/12m) 또는 chase_alpha 재조정 |
| C — 다시 collapse | 재진단 — controller 또는 reward 잔존 issue |

## Devil's Advocate

- **거리 8m + bound 10m**: drone v_max=10m/s + 5초 episode = 50m 이동 가능 → 거리 8m 충분. 그러나 oob 위험 ↑ — bound 10이 spawn 4 + chase 마진 6m이라 OK.
- **alpha 0.3 / 0.2**: 너무 완만하면 hover trap (chase 영역에 머무름). 그러나 coef 그대로라 hover incentive 작음. 학습 monitoring으로 ep_length 패턴 보면 hover trap 식별 가능.
- **R_self_crash = R_captured = -3.0 등호 잔존**: borderline 자살 incentive. ams_s2_v6에서 att_crash 거의 0이었는데 거리 확대 후 attacker가 evade 어려워지면 자살 학습 가능. 첫 5 cycle att_crash trend 점검 필요.
- **base ckpt — Phase 1.0/1.5 fresh restart 비용**: ~2시간 GPU. ams_s2_v6 ckpt를 그대로 쓰면 spawn 변경에 적응 cycle 1-2 들어 더 빠를 수도. 그러나 baseline 깨끗한 비교 위해 fresh.
