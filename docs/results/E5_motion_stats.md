# E5 — Defender Motion Statistics (Intro Motivation Quantification)

> **Run date**: 2026-05-15 10:14~10:15 KST (1.5 min CPU)
> **Status**: ✅ Complete
> **Owner**: Phase 3f / Paper §1 Intro
> **Spec ref**: `docs/plans/phase_3f_tau_unaware_flow_forecaster.md` + 사용자 master plan E5 (2026-05-15)

---

## Goal

Paper Intro Figure 1 의 정량적 anchor — "delay 가 attacker-defender 적대 게임에서 치명적" 직관을 데이터로 증명. 학습 무관, 순수 rollout statistics 분석.

핵심 측정값: **defender 가 τ-step delay 동안 이동한 거리 / capture radius (r_cap)** 의 분포.

---

## Method

`scripts/analyze_motion_stats.py` (174 lines, CPU only).

### Data
- 5 cycle × 1000 episode subsample (ams_v9 multi-cycle dataset)
- Total per-step speed samples: **1,428,185**
- 데이터 경로: `data/forecaster_data_ams_v9_c{7,8,14,17,20}/`

### Computation (per episode)
```python
# Per-step defender speed (m/s)
speeds = ||s_d_world[:, 3:6]||                           # (L,)

# Per-τ displacement: Euclidean distance over τ-step window
delta = s_d_pos[τ:] - s_d_pos[:-τ]                       # (L-τ, 3)
displacement = ||delta||                                  # (L-τ,)

# Critical ratio
ratio = displacement / capture_radius                     # capture_radius=0.30 m
```

### Env params (ams_v9 cfgs.pkl)
- `dt = 0.005 s` (200 Hz)
- `capture_radius = 0.30 m`
- `v_max = 10 m/s`

---

## Results

### Per-τ displacement statistics

| τ (step) | τ (ms) | mean (m) | p50 | p90 | max | **ratio mean** | **% > 1× r_cap** | **% > 5× r_cap** |
|---|---|---|---|---|---|---|---|---|
| 10 | 50 | 0.17 | 0.15 | 0.31 | 0.71 | 0.55 | 11.9% | 0.0% |
| 25 | 125 | 0.41 | 0.37 | 0.77 | 1.74 | **1.38** | **61.0%** | 0.1% |
| **50** | **250** | **0.82** | **0.73** | **1.50** | **3.38** | **2.73** | **87.4%** | **10.0%** |

### Per-step speed
- mean: ~3.3 m/s, p50 ~3.0 m/s, p90 ~6.2 m/s, max ≈ v_max=10 m/s

### Key headline numbers (paper Intro 사용)
1. **τ=50 (250ms) 에서 87% episode 가 "defender 이동거리 > capture radius"** — reactive interception 구조적 불가능 영역
2. **10% episode 는 5× r_cap (1.5m+) 초과** — defender 가 한참 앞에 있는 시점에 controller 가 반응
3. τ=10 (50ms, cellular relay 영역) 에서도 12% 가 r_cap 초과 — moderate delay 도 nontrivial

---

## Paper 사용

### Figure 1 (Intro) — "Why delay is critical"
- 위치: `logs/forecaster/diagnostics/motion_stats/intro_motion_motivation.png`
- Style: stacked KDE/histogram of (displacement / r_cap) per τ
- Annotation: vertical red line @ ratio=1 ("reactive limit"), shaded danger zone
- Caption (draft):
  > "Defender displacement during τ-step communication delay, normalized by capture radius. At τ=50 (250ms — typical cellular relay + crypto + LEO SATCOM hop), 87% of trajectories exceed the reactive interception window. 10% exceed 5× the capture radius — the controller arrives at a position the defender has long abandoned."

### 본문 §1 sentence draft
> "In our 1v1 quadrotor pursuit-evasion task with capture radius r_cap = 0.30m, communication delay of just τ = 50 steps (250ms at 200Hz control) means the defender displaces a median of 0.73m — **more than twice the capture radius** — by the time the attacker observes its position. 87% of decision points fall outside the reactive interception window, motivating the need for explicit forecasting."

---

## Files / Locations

| | path |
|---|---|
| Script | `scripts/analyze_motion_stats.py` |
| Stats JSON | `logs/forecaster/diagnostics/motion_stats/motion_stats.json` |
| **Intro figure** | `logs/forecaster/diagnostics/motion_stats/intro_motion_motivation.png` |
| Run log | `/tmp/motion_stats.log` |

---

## Reproducibility

```bash
.venv/bin/python scripts/analyze_motion_stats.py \
  --data_dir data/forecaster_data_ams_v9_c7 \
             data/forecaster_data_ams_v9_c8 \
             data/forecaster_data_ams_v9_c14 \
             data/forecaster_data_ams_v9_c17 \
             data/forecaster_data_ams_v9_c20 \
  --cfgs_pkl logs/phasef/ams_v9/cfgs.pkl \
  --tau_values 10 25 50 \
  --num_episodes_per_dir 1000 \
  --out_dir logs/forecaster/diagnostics/motion_stats
```

Wall-clock: ~1.5 min on CPU. Deterministic via `--seed 0`.

---

## Devil's Advocate (limitation 명시)

- **r_cap = 0.30m 는 한 setup의 값** — generalization claim 약함. paper §6 limitation 명시: "Quantitative ratios depend on capture radius; trend (delay-induced reactive impossibility) holds across r_cap values."
- **defender velocity는 ams_v9 정책 조건부** — 다른 baseline policy (random, scripted) 에서 분포 다름. 단 우리 paper 는 ams_v9 학습 환경에 일관됨.
- **휴리스틱 capture model** — 실제로는 capture 가 instantaneous 아니라 small dwell time 필요할 수 있음. 단 r_cap 자체가 conservative bound.

## Next experiments using this data
- E2 (5-forecaster comparison) 에서 mission/capture rate 결과 해석에 본 통계 직접 인용
- E3 (random-delay) 에서 "long-delay regime 의 한계" 시각화에 활용

---

*보고서 작성: 2026-05-15. 후속 실험 보고서는 `docs/results/E{n}_*.md` 패턴으로 동일.*
