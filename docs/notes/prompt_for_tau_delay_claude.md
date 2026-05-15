# Phase F baseline 학습이 너의 코드 변경 때문에 두 번 망함

너가 어제 2026-05-12 저녁에 `envs/base_pe_env.py`와 `envs/pe_1v1/env.py`에 τ-delay 인프라 (`defender_state_history` ring buffer)를 추가했어. 두 가지 문제 발생:

## 1. 학습 도중 TypeError로 cycle 18에서 abort

`base_pe_env.py` `_push_defender_history()`:
```python
self.defender_state_history = torch.roll(
    self.defender_state_history, shifts=-1, dim=0,   # ← 당시 코드
)
```

PyTorch `torch.roll` 의 keyword는 **`dims` (plural)**. `dim=0` 호출 → `TypeError: roll() got an unexpected keyword argument 'dim'`.

학습 (`logs/phasef/ams_v6_highspawn`) cycle 17 끝나는 시점 (~20:48)에 코드 추가 → cycle 18 A.check r4 child process가 새 코드 import → 첫 step()에서 fail → 전체 학습 abort. cycle 18-20 결과 못 만듬.

## 2. Backward compatibility 깨짐 — 재학습도 결과 다름

`env.py` `_attacker_obs()` line 408에서 `delayed_kin = self.defender_state_history[0]` **무조건 사용**. `tau_delay=0` (default)도 history path 거치게 됨.

같은 yaml (`configs/envs/pe_1v1_ams_v6.yaml`) + 같은 seed (`--seed 0`)로 다시 학습 (`logs/phasef/ams_v6_overnight`) → 결과 매우 다름:

| cycle | 기존 (코드 변경 전) | 새 (변경 후) |
|---|---|---|
| 5 | a_win 0.283 | 0.113 |
| 6 | **0.600** | **0.035** |
| 7 | **0.680** | 0.050 |
| 8 | **0.721** | 0.025 |

cycle 6-8 attacker mission rate 60-72% → 3-5%로 폭락. **학습 실패**.

`tau_delay=0`이라 값은 동일해야 하지만 `torch.roll` + tensor indexing 추가 ops가 GPU에서 cuDNN nondeterministic ops 발동 → seed reproducibility 깨짐 → 학습 trajectory 변경.

## 요구사항

1. `env.py` `_attacker_obs()` 에 backward compat 분기:
   ```python
   if self.tau_delay > 0:
       delayed_kin = self.defender_state_history[0]
       delayed_def_pos = delayed_kin[:, :3]
       delayed_def_vel = delayed_kin[:, 3:]
   else:
       delayed_def_pos = self.defender_pos
       delayed_def_vel = self.defender_vel
   ```
2. `base_pe_env.py` `_push_defender_history()` 도 `tau_delay==0`일 때 skip (overhead 제거 + ops 안 발생)
3. smoke test로 `tau_delay=0` 학습이 기존 코드와 **100% 동일 결과** 확인 (seed=0, 짧은 학습 비교)
4. **baseline 학습 진행 중에는 절대 핵심 env 코드 변경 X**

피해: 사용자 어제 학습 2시간 + overnight 학습 2시간 = **4시간 wall 낭비**. baseline (`ams_v6_highspawn cycle 16, a_win 0.836`) 도 재현 불가능 — 같은 seed라도 코드 다르면 결과 다름.
