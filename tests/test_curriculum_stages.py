"""Phase C — Curriculum stage yaml schema + monotonicity contracts.

Plan: docs/plans/phase_c_curriculum_scratch.md.

What this test pins (so refactor / accidental edits fail loud):
  1. 6 stage yamls (0..5) parse & contain required top-level keys.
  2. ``stage_5.yaml`` is byte-identical to ``pe_1v1_phase_m_ams.yaml``
     (the curriculum graduation gate — transition to AMS must be a no-op).
  3. ``reward`` / ``action`` / ``sim`` / ``obs_scales`` blocks are identical
     across all 6 stages (curriculum varies env, NOT reward signal).
  4. 4-axis difficulty is monotone non-decreasing across 0→5:
       - spawn xy span widens
       - goal box xy span widens
       - bound_xyz expands
       - z_floor lowers (more floor margin given up)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CURR_DIR = ROOT / "configs" / "envs" / "pe_1v1_curriculum"
AMS_YAML = ROOT / "configs" / "envs" / "pe_1v1_phase_m_ams.yaml"

STAGE_IDS = list(range(5))    # v4 spec (사용자 2026-05-05): stage 4 = AMS 본 학습. stage 5 폐기.
TOP_KEYS = ["sim", "termination", "spawn", "g_mission", "obs_scales", "reward", "action"]
# reward 는 stage 0 한정 변경 허용 (stage 0 goal_dense 5× 강화).
# stage 1~4 간 reward 동일성은 별도 테스트.
SHARED_BLOCKS = ["sim", "obs_scales", "action"]
REWARD_SHARED_STAGES = [1, 2, 3, 4]


def _load(p: Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def stages() -> dict[int, dict]:
    return {k: _load(CURR_DIR / f"stage_{k}.yaml") for k in STAGE_IDS}


@pytest.fixture(scope="module")
def ams_cfg() -> dict:
    return _load(AMS_YAML)


def test_all_stage_yamls_exist_and_parse(stages):
    for k in STAGE_IDS:
        assert isinstance(stages[k], dict), f"stage_{k}.yaml not a dict"


def test_required_top_keys(stages):
    for k, cfg in stages.items():
        missing = [key for key in TOP_KEYS if key not in cfg]
        assert not missing, f"stage_{k} missing top keys: {missing}"


def test_stage4_matches_ams_except_bound_yz_and_reward(stages, ams_cfg):
    """Stage 4 = AMS 본 학습. v4: bound_y/z만 다름 (6 → 3). v11 (2026-05-05):
    reward 도 다름 (Option B + timeout off — Nash 균형 강화).

    여전히 일치해야 하는 것: spawn / g_mission / sim / obs_scales / action +
    termination 의 bound_y/z 외 모든 키. reward 일치는 별도 (stage 1~4 동일성).
    """
    s4 = stages[4]
    for block in ("spawn", "g_mission", "sim", "obs_scales", "action"):
        assert s4[block] == ams_cfg[block], f"stage_4.{block} differs from AMS.{block}"
    s4_term = dict(s4["termination"])
    ams_term = dict(ams_cfg["termination"])
    for k in ("bound_y", "bound_z"):
        s4_term.pop(k, None)
        ams_term.pop(k, None)
    assert s4_term == ams_term, (
        "stage_4.termination differs from AMS in keys other than bound_y/z"
    )
    assert s4["termination"]["bound_y"] == 3.0
    assert s4["termination"]["bound_z"] == 3.0


def test_v11_terminal_rewards_consistent(stages):
    """v11 (2026-05-05) Option B + timeout off — 모든 stage에 적용 검증."""
    for k in STAGE_IDS:
        r = stages[k]["reward"]
        assert r["R_mission_win"] == 20.0, f"stage_{k} R_mission_win != 20 (v17)"
        assert r["R_captured"] == -7.0, f"stage_{k} R_captured != -7"
        assert r["R_timeout"] == -1.0, f"stage_{k} R_timeout != -1"
        assert r["R_capture_win"] == 20.0, f"stage_{k} R_capture_win != 20 (v17)"
        assert r["R_mission_failed"] == -7.0, f"stage_{k} R_mission_failed != -7"
        assert r["R_timeout_win"] == 1.0, f"stage_{k} R_timeout_win != +1"
        assert r["R_self_crash"] == -10.0, f"stage_{k} R_self_crash != -10"
        assert r["R_opp_crash"] == 1.0, f"stage_{k} R_opp_crash != +1"
        assert r["existential"] == 0.0002, f"stage_{k} existential != 0.0002 (v16)"


def test_shared_blocks_identical_across_stages(stages):
    """sim / obs_scales / action 는 stage 간 변경 금지."""
    base = stages[0]
    for k in STAGE_IDS[1:]:
        for block in SHARED_BLOCKS:
            assert stages[k][block] == base[block], (
                f"stage_{k}.{block} differs from stage_0.{block} — curriculum may "
                f"only vary spawn / g_mission / termination / (stage 0 reward)."
            )


def test_reward_identical_across_stages_1_to_5(stages):
    """stage 1..5 의 reward 는 동일해야 한다 (stage 5 = AMS, stage 1~4도 같은 reward)."""
    base = stages[REWARD_SHARED_STAGES[0]]["reward"]
    for k in REWARD_SHARED_STAGES[1:]:
        assert stages[k]["reward"] == base, (
            f"stage_{k}.reward differs from stage_{REWARD_SHARED_STAGES[0]}.reward."
        )


def _xy_span(cfg: dict, group: str, key_x: str = "x", key_y: str = "y") -> tuple[float, float]:
    sub = cfg["spawn"][group] if group in cfg.get("spawn", {}) else cfg["g_mission"]["box"]
    sx = float(sub[key_x][1]) - float(sub[key_x][0])
    sy = float(sub[key_y][1]) - float(sub[key_y][0])
    return sx, sy


def test_spawn_xy_monotone(stages):
    """attacker / defender spawn box 너비는 0→5에서 단조 비감소."""
    for group in ("attacker_init_pos", "defender_init_pos"):
        prev_x, prev_y = _xy_span(stages[0], group)
        for k in STAGE_IDS[1:]:
            cx, cy = _xy_span(stages[k], group)
            assert cx >= prev_x - 1e-9, f"stage_{k} {group} x span shrunk: {cx} < {prev_x}"
            assert cy >= prev_y - 1e-9, f"stage_{k} {group} y span shrunk: {cy} < {prev_y}"
            prev_x, prev_y = cx, cy


def test_goal_xy_monotone(stages):
    prev_x, prev_y = _xy_span(stages[0], group="__goal__")
    for k in STAGE_IDS[1:]:
        cx, cy = _xy_span(stages[k], group="__goal__")
        assert cx >= prev_x - 1e-9, f"stage_{k} goal x span shrunk: {cx} < {prev_x}"
        assert cy >= prev_y - 1e-9, f"stage_{k} goal y span shrunk: {cy} < {prev_y}"
        prev_x, prev_y = cx, cy


def test_bound_monotone(stages):
    """bound_x / bound_y / bound_z 는 단조 비감소."""
    for axis in ("bound_x", "bound_y", "bound_z"):
        prev = float(stages[0]["termination"][axis])
        for k in STAGE_IDS[1:]:
            cur = float(stages[k]["termination"][axis])
            assert cur >= prev - 1e-9, f"stage_{k} {axis} shrunk: {cur} < {prev}"
            prev = cur


def test_z_floor_monotone_decrease(stages):
    """z_floor 는 단조 비증가 (난이도 증가 = 안전망 풀림)."""
    prev = float(stages[0]["termination"]["z_floor"])
    for k in STAGE_IDS[1:]:
        cur = float(stages[k]["termination"]["z_floor"])
        assert cur <= prev + 1e-9, f"stage_{k} z_floor grew: {cur} > {prev}"
        prev = cur
