# Phase C — Curriculum scratch 학습 (OOD collapse 처방, 수정판)

> **상태**: 사용자 승인 대기 (§11). **수정 이력**: v1에서 새 orchestrator 신설 제안 → v2에서 기존 `scripts/train_ams_drl.py` (scratch + per-stage adaptive 이미 구현됨) 재사용으로 변경. CLAUDE.md §1 "기존 동작 코드 손대지 않음" 준수.
> **선행 진단**: `logs/pe1v1_v0/ams_s2_v30_per_stage/` C1~C4 — solo 학습은 둘 다 잘 됐으나 AMS 진입 직후 collapse. **reward bug 아님, OOD distribution shift**. 자세한 데이터는 v1 §1 참조 (이 파일은 v2이므로 진단 요약만 유지).

---

## 1. 동기 (간략)

- `attacker_v0` (vs stationary def): mission **0.94**, oob 0.026, crash 0
- `defender_v0` (vs ?): captured **0.99**, def_crash **0**, def_oob **0**
- 같은 ckpt가 AMS C1에서 def_crash_z **76%**, def_oob **90%** — 학습 분포 밖 attacker 행동에 OOD 출력.
- → 환경 + 상대 분포를 **점진 확장**해서 OOD 영역 자체를 사전학습으로 흡수.

---

## 2. 기존 자산 활용 (수정 v2 핵심)

### 2.1 `train_ams_drl.py`에 이미 있는 것

| 기능 | 구현 위치 | curriculum에서 활용 |
|---|---|---|
| `--train_mode scratch` | line 79~83, 268~277 | stage 0 진입 시 ckpt 없이 시작 |
| `--train_mode pretrain` + `--*_resume` | line 84~93, 280~299 | stage K → K+1 ckpt 인계 |
| cycle 1 scratch attacker = vs stationary | line 392~408 | stage 0 자연 구현 |
| cycle 2+ alternating = vs prev learned | line 410~422 | **opponent curriculum 자동** |
| `--attacker_min_winrate` / `--per_stage_max_rounds` (A.adaptive) | line 137~145, 449~514 | per-stage 진급 gate |
| `--defender_min_winrate` (B.adaptive) | line 140~143, 550~616 | 동일 |
| `--env_yaml` per-call | line 113~118 | **stage별 yaml 갈아끼우기 가능** |
| H2H eval per cycle | line 618~641 | 졸업 metric 자동 산출 |

→ **새 orchestrator 불필요**. 각 curriculum stage를 `train_ams_drl.py --max_cycles=1 --env_yaml stage_K.yaml`로 한 번 호출하면 됨. opponent curriculum은 stage K 산출 ckpt가 stage K+1의 `--attacker_ckpt`/`--defender_ckpt`로 인계되며 자연 진행.

### 2.2 따라서 신규 자산은 둘뿐

1. **Stage env yaml 6개** (`configs/envs/pe_1v1_curriculum/stage_{0..5}.yaml`)
2. **Thin shell wrapper** (`scripts/train_curriculum.sh` 또는 `train_curriculum.py`) — stage 시퀀스를 자동 호출 + ckpt path 인계만 담당. 진급 게이트는 `train_ams_drl.py --attacker_min_winrate / --defender_min_winrate`에 위임.

---

## 3. Stage 설계

### 3.1 4축 동기 진급

| 축 | Stage 0 | Stage 5 (= AMS) | 비고 |
|---|---|---|---|
| spawn box (attacker) | x∈[2.0,3.0], y∈[-0.3,0.3], z∈[1.2,1.4] | x∈[0.5,4.0], y∈[-1,1], z∈[1.0,1.5] | 점진 확장 |
| spawn box (defender) | x∈[-3.0,-2.0], y∈[-0.3,0.3], z∈[1.2,1.4] | x∈[-4.0,-0.5], y∈[-1,1], z∈[1.0,1.5] | 동일 |
| goal box | x∈[-6,-5], y∈[-0.5,0.5] | x∈[-10,-4], y∈[-1.5,1.5] | mission 난이도 |
| termination | bound=8, z_floor=0.3, **attitude crash on (60°)** | bound_x=12/y=6/z=6, z_floor=0.05, **crash off** | 안전망 풀림 |
| **opponent (자동)** | **stationary** (cycle 1 scratch 자동) | **prev-stage learned ckpt** (cycle 2+ 자동) | `train_ams_drl.py`가 처리 |

### 3.2 Stage 정의 (제안)

```
Stage 0: easy — 좁은 spawn, near goal, 엄격 termination
Stage 1: spawn-medium — spawn 60% 확장
Stage 2: goal-medium — goal box 60% 확장
Stage 3: relax-term — attitude crash off, z_floor 0.1
Stage 4: full-env — spawn/goal/term = AMS env, opponent만 prev stage learned
Stage 5: full-AMS — pe_1v1_phase_m_ams.yaml과 1:1 동일 (졸업 게이트)
```

각 stage는 별도 yaml 파일. **Stage 5 yaml = `pe_1v1_phase_m_ams.yaml` 그대로 복사** (transition 0).

### 3.3 Stage별 호출 명령 (예)

```bash
# Stage 0 (scratch, opponent=stationary, env=stage_0)
python scripts/train_ams_drl.py \
  --train_mode scratch \
  --env_yaml configs/envs/pe_1v1_curriculum/stage_0.yaml \
  --num_envs 1024 --switch_every 200 --max_cycles 1 \
  --attacker_min_winrate 0.7 --defender_min_winrate 0.5 \
  --per_stage_max_rounds 3 \
  --logs_root logs/pe1v1_v0/curriculum_v1/stage_0 \
  --seed 0 --logger wandb

# Stage 1 (pretrain from stage 0, env=stage_1)
python scripts/train_ams_drl.py \
  --train_mode pretrain \
  --attacker_ckpt logs/.../stage_0/attacker_v1.pt \
  --defender_ckpt logs/.../stage_0/defender_v1.pt \
  --attacker_resume logs/.../stage_0/model_attacker_v1.ckpt \
  --defender_resume logs/.../stage_0/model_defender_v1.ckpt \
  --env_yaml configs/envs/pe_1v1_curriculum/stage_1.yaml \
  --num_envs 1024 --switch_every 200 --max_cycles 1 \
  --attacker_min_winrate 0.7 --defender_min_winrate 0.5 \
  --per_stage_max_rounds 3 \
  --logs_root logs/pe1v1_v0/curriculum_v1/stage_1 \
  --seed 0 --logger wandb

# ... stage 2~4 동일 패턴

# Stage 5 (= AMS S2 본 학습)
python scripts/train_ams_drl.py \
  --train_mode pretrain \
  --attacker_ckpt logs/.../stage_4/attacker_v1.pt \
  --defender_ckpt logs/.../stage_4/defender_v1.pt \
  --*_resume ... \
  --env_yaml configs/envs/pe_1v1_phase_m_ams.yaml \
  --num_envs 1024 --switch_every 100 --max_cycles 20 \
  --attacker_min_winrate 0.4 --defender_min_winrate 0.4 \
  --per_stage_max_rounds 5 \
  --convergence_window 5 --convergence_eps 0.05 --draw_max 0.2 \
  --logs_root logs/pe1v1_v0/ams_s2_v31_curriculum \
  --seed 0 --logger wandb
```

---

## 4. 신규/수정 파일 (수정 v2)

### 신규 파일만

- `configs/envs/pe_1v1_curriculum/stage_0.yaml` ~ `stage_5.yaml` (6개)
  - schema는 `pe_1v1_phase_m_ams.yaml`과 동일, 4축 값만 차등.
  - stage 5 = ams yaml과 1:1 동일 (확인 게이트).
- `scripts/train_curriculum.sh` 또는 `scripts/train_curriculum.py`
  - 위 §3.3 6개 명령을 시퀀스로 실행. 이전 stage ckpt path를 다음 stage 인자로 자동 전달.
  - 각 stage 종료 후 cycle_001/h2h.json 읽어서 진급 winrate 확인 → 졸업 게이트 fail 시 abort + 사용자 알림.
  - shell이면 50줄 미만, python이면 100줄 미만으로 충분.
- `tests/test_curriculum_stages.py` (가벼움)
  - stage_{0..5}.yaml 6개가 모두 valid yaml + 필수 키 (sim, termination, spawn, g_mission, reward, action) 존재 + stage_5와 ams yaml diff 0인지 확인.

### 수정 파일

- **없음.** `train_ams_drl.py` / `train_attacker.py` / `train_defender.py` / `envs/pe_1v1/env.py` 모두 무수정.

---

## 5. 영향 범위

| 영역 | 영향 |
|---|---|
| obs / action shape | 무변경 |
| reward keys / 값 | **무변경** (사용자 명시 — solo 검증 신호 보존) |
| 기존 ams_s2_v30 ckpt | 폐기 안 함, 옆에 보관 |
| 학습 시간 | stage당 switch_every=200 × per_stage_max_rounds=3 = 600 iter (한쪽), 양쪽 = 1200 iter, 6 stage = **7200 iter (≈ 4시간 GPU @ 1024 envs)**. §11 사전 승인. |
| 코드 변경량 | yaml 6개 + shell ~50줄 + 테스트 1개. **새 모듈 없음**. |

---

## 6. 물리/이론 sanity check

### 6.1 Stage 0 학습 가능성
- spawn 좁음 + goal 가까움 + stationary 상대. 사실상 phase1.5 attacker solo 학습과 동등.
- 기존 attacker solo는 600 iter에 mission 0.4, 800 iter에 0.87 → stage 0은 200~400 iter면 0.7 도달 가능. `--switch_every 200 --per_stage_max_rounds 3`으로 600 iter cap.

### 6.2 Stage 0 → 1 진급 시 distribution shift 폭
- spawn 60% 확장 → 입력 분포 1.6× 표준편차. 기존 ckpt가 좁은 분포에서 0.7 winrate면, 1.6× 분포에서 winrate 0.4~0.5 dip 예상. per-stage adaptive (max_rounds=3, switch_every=200)로 600 iter 추가 학습 → 0.7 회복 시도.
- 회복 실패 시 `train_ams_drl.py`가 0.7 미만으로 stage 종료 → wrapper가 abort.

### 6.3 Stage 4 → 5 transition
- stage 5 yaml = ams yaml 1:1 동일. transition 0 → distribution shift 0. 졸업 보장.
- 단, stage 4 yaml과 ams yaml의 차이가 작아야 stage 4 → 5 dip도 작음. stage 4를 ams yaml의 90% 수준으로 설계.

### 6.4 Opponent curriculum 자동 작동 확인
- Stage 0 (scratch, max_cycles=1): cycle 1 attacker = vs stationary, cycle 1 defender = vs stage 0 attacker (active learned). → 졸업 시 양쪽 모두 active opponent 분포 일부 학습.
- Stage 1+ (pretrain, max_cycles=1): attacker = vs stage 0 defender (active), defender = vs stage 1 attacker (active). → opponent 분포가 매 stage 자연 확장.
- **stage 4 직전에 별도 opponent 전환 단계 불필요** — alternating 구조가 매 stage 그 일을 함.

---

## 7. Rollback plan

- stage K (K∈{1..5})에서 졸업 게이트 fail (attacker_min_winrate or defender_min_winrate 미달):
  - 1차: `--per_stage_max_rounds`를 3 → 5로 늘려 K stage 재학습.
  - 2차: stage K-1과 K 사이에 **K' (보간 stage)** 추가 — 4축 진급 폭 30%로 축소.
  - 3차: scratch 포기, **resume + 분포 확장 사전학습** (옵션 A — 기존 attacker_v0/defender_v0에 stage 4~5 yaml로 추가 학습)으로 전환.
- AMS S2 (stage 5) 졸업 후 alternating 본 학습에서 collapse 재발 시:
  - `--switch_every` 100 → 200으로 늘려 적응 시간 ↑.
  - `--per_stage_max_rounds` 5 → 3으로 축소 (over-training 진동 방지).
  - 그래도 실패면 reward 재검토 (마지막 옵션, solo 검증 신호 깨짐 위험).

---

## 8. 검증 / 테스트 계획

### 8.1 단위
- `tests/test_curriculum_stages.py`
  - 6개 yaml 파싱 OK + 필수 키 존재
  - stage_5.yaml ≡ pe_1v1_phase_m_ams.yaml (deep diff)
  - stage K → K+1 spawn / goal box monotonic 확장 (회귀 없음)
  - stage 0 termination이 stage 5보다 엄격 (bound 작고 z_floor 큼)

### 8.2 통합 (sanity)
- Stage 0 smoke 1 cycle (`--smoke` flag로 cpu, num_envs=16, switch_every=2):
  - `train_ams_drl.py` 정상 종료 + cycle_001/h2h.json 생성
  - attacker_v1.pt, defender_v1.pt 생성

### 8.3 학습 검증 (per stage)
- 매 stage 종료 시 cycle_001/h2h.json 자동 확인 (wrapper):
  - att_winrate ≥ attacker_min_winrate
  - def_winrate ≥ defender_min_winrate
  - 둘 다 fail 시 wrapper abort + 사용자 알림
- 추가: stage 시작/끝 wandb tag (`stage=0..5`)로 시계열 비교 가능하게.

---

## 9. 사용자 원안과의 차이 (수정 v2 — 더 단순해짐)

| 사용자 원안 | v1 제안 | **v2 제안** |
|---|---|---|
| termination 점진 완화 | ✅ | ✅ |
| spawn 점진 확장 | ✅ + goal box 동기 | ✅ |
| reward 안 건드림 | ✅ | ✅ |
| (없음) | opponent curriculum stage 4 (별도 정의) | **자동 — train_ams_drl.py의 alternating이 처리** |
| (없음) | per-side sequential 학습 | **불필요 — alternating 그대로 사용** |
| (없음) | 진급+강등 메커니즘 (새 로직) | **`--*_min_winrate` + `--per_stage_max_rounds`로 위임** |
| (없음) | 새 orchestrator (`scripts/train_curriculum.py`) | **shell wrapper만 (50줄)** |
| (없음) | curriculum_manager.py 모듈 | **삭제** |

→ 신규 파일 6+1+1 = 8개, 수정 파일 0개.

---

## 10. TODO 항목 (사용자 승인 후 순서)

1. **(승인 게이트)** 사용자가 본 plan v2 승인 + GPU wall budget 4시간 사전 승인 (§11).
2. `configs/envs/pe_1v1_curriculum/stage_5.yaml` 작성 (= ams yaml 복사) → diff 0 확인.
3. `configs/envs/pe_1v1_curriculum/stage_{0..4}.yaml` 작성 (4축 값만 차등).
4. `tests/test_curriculum_stages.py` 작성 + pytest 통과.
5. `scripts/train_curriculum.sh` 작성 (50줄). 또는 python wrapper.
6. **Stage 0 smoke test** — `--smoke` 모드로 1분 미만 검증.
7. **Stage 0 본 학습** (실제 GPU). 수렴 확인 후 stage 1로.
8. Stage 1~5 순차 실행. 매 stage 종료 시 wrapper가 진급 게이트 자동 확인.
9. Stage 5 졸업 → AMS S2 본 학습 (`ams_s2_v31_curriculum/`) 시작.

---

## 11. 열린 질문 (사용자 결정 필요)

1. **`switch_every`**: stage당 200 iter (≈300초)면 충분한가? solo 600 iter에 mission 0.4 → 200 iter면 stage 0 부족할 수 있음. **stage 0만 switch_every=400, stage 1~4는 200**으로 비대칭 권장.
2. **진급 게이트 임계값**: stage 0~3 `attacker_min_winrate=0.7`, stage 4~5 `0.4` (AMS 균형 환경)로 점진 완화. defender도 동일 패턴. 동의?
3. **Stage 진급 폭**: 4축 60% 단계가 너무 큰가/작은가? 너무 크면 stage 6단계 → 8단계로 세분 필요.
4. **Wrapper 언어**: shell vs python? shell이 단순하지만 게이트 json 파싱 시 jq 의존. python이면 의존 0 + 테스트 가능.
5. **Logs 보존 정책**: stage 0~4 ckpt도 영구 보관 vs 졸업 ckpt만 유지? 디버깅 위해 전부 보관 권장 (디스크 6 stage × ~2GB ≈ 12GB).
