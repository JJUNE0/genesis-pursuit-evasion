"""S1 diagnostic: AMS-S2 cycle-wise termination metric trajectory.

Reads tfevents from cycle_001..cycle_020 attacker_train/defender_train phases,
concatenates them on a global PPO-iter axis with cycle/role boundaries marked,
and dumps a 7-panel PNG + per-phase summary table.

CLAUDE.md compliance: read-only diagnostic, no env / config changes.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

ROOT = Path(
    "/home/cocel/Desktop/Research/genesis-pursuit-evasion/logs/pe1v1_v0/ams_s2"
)
N_CYCLES = 20
METRICS = [
    "Episode/captured_rate",
    "Episode/att_crash_rate",
    "Episode/def_crash_rate",
    "Episode/timeout_rate",
    "Episode/att_oob_rate",
    "Episode/def_oob_rate",
    "Train/mean_episode_length",
]


def find_events(cycle_dir: Path, role: str) -> Path | None:
    role_dir = cycle_dir / f"{role}_train"
    if not role_dir.exists():
        return None
    runs = sorted(d for d in role_dir.iterdir() if d.is_dir())
    if not runs:
        return None
    files = sorted(runs[-1].glob("events.out.tfevents.*"))
    return files[0] if files else None


def load_tfevents(ef: Path) -> tuple[dict, float]:
    ea = event_accumulator.EventAccumulator(
        str(ef), size_guidance={"scalars": 0}
    )
    ea.Reload()
    tags = ea.Tags()["scalars"]
    out = {}
    first_wall = None
    for k in METRICS:
        if k in tags:
            evs = ea.Scalars(k)
            steps = np.array([e.step for e in evs])
            vals = np.array([e.value for e in evs])
            out[k] = (steps, vals)
            if first_wall is None and len(evs):
                first_wall = evs[0].wall_time
    return out, (first_wall if first_wall else 0.0)


# ---------------- Collect ----------------
data: dict[str, dict[int, dict]] = {"attacker": {}, "defender": {}}
first_wall: dict[str, dict[int, float]] = {"attacker": {}, "defender": {}}
for c in range(1, N_CYCLES + 1):
    cdir = ROOT / f"cycle_{c:03d}"
    for role in ("attacker", "defender"):
        ef = find_events(cdir, role)
        if ef is None:
            continue
        d, t0 = load_tfevents(ef)
        data[role][c] = d
        first_wall[role][c] = t0

# Determine phase order per cycle by wall time
phases: list[tuple[str, int, float, float]] = []
xpos = 0.0
for c in range(1, N_CYCLES + 1):
    pairs = [
        (role, first_wall[role].get(c, float("inf")))
        for role in ("attacker", "defender")
        if c in data[role]
    ]
    pairs.sort(key=lambda x: x[1])
    for role, _ in pairs:
        # Use any metric to derive step span
        steps_max = 0
        for m in METRICS:
            if m in data[role][c] and len(data[role][c][m][0]):
                steps_max = max(steps_max, int(data[role][c][m][0].max()))
        x_start = xpos
        x_end = xpos + steps_max
        phases.append((role, c, x_start, x_end))
        xpos = x_end

print(f"phases concat: total x_extent = {xpos:.0f} PPO iters")

# ---------------- Plot ----------------
fig, axes = plt.subplots(
    len(METRICS), 1, figsize=(18, 2.4 * len(METRICS)), sharex=True
)
for i, m in enumerate(METRICS):
    ax = axes[i]
    for role, c, xs, xe in phases:
        if c not in data[role] or m not in data[role][c]:
            continue
        steps, vals = data[role][c][m]
        color = "tab:blue" if role == "attacker" else "tab:red"
        ax.plot(xs + steps, vals, color=color, alpha=0.75, lw=1.0)
    # cycle vertical lines (only at attacker phase start typically)
    last_c = None
    for role, c, xs, xe in phases:
        if c != last_c:
            ax.axvline(xs, color="gray", alpha=0.25, lw=0.5)
            last_c = c
    label = m.replace("Episode/", "").replace("Train/", "")
    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, alpha=0.3)

# cycle labels on top axis
ax_top = axes[0]
last_c = None
for role, c, xs, xe in phases:
    if c != last_c:
        ax_top.text(
            xs, ax_top.get_ylim()[1] * 1.02, f"c{c}",
            fontsize=7, alpha=0.6, ha="left", va="bottom",
        )
        last_c = c

axes[-1].set_xlabel(
    "global PPO iter (cycles concatenated; blue=attacker phase, red=defender phase)"
)
fig.suptitle(
    "AMS-S2 termination & ep_length across cycles 1-20  "
    "(diagnostic S1 — env-internal training distribution)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.985))
out_png = ROOT / "diagnostics_s1_termination_trajectory.png"
fig.savefig(out_png, dpi=120)
print(f"saved {out_png}")

# ---------------- Per-phase summary table ----------------
def tail_mean(arr: np.ndarray, n: int = 5) -> float:
    if len(arr) == 0:
        return float("nan")
    return float(arr[-min(n, len(arr)):].mean())


print()
print("=== Per-cycle / per-phase last-5-points mean ===")
hdr_fields = ["cyc", "role", "cap", "att_cr", "def_cr", "to", "a_oob", "d_oob", "ep_len"]
print("  ".join(f"{h:>7s}" for h in hdr_fields))
print("-" * 72)
for role, c, _, _ in phases:
    if c not in data[role]:
        continue
    row = [f"{c:>7d}", f"{role:>7s}"]
    for m in METRICS:
        v = tail_mean(data[role][c].get(m, (None, np.array([])))[1])
        if "rate" in m:
            row.append(f"{v:>7.3f}")
        else:
            row.append(f"{v:>7.1f}")
    print("  ".join(row))
