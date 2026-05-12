"""Pretty-printed training summary for PE 1v1 scripts.

Used by ``scripts/train_attacker.py`` and ``scripts/train_defender.py`` to
echo the resolved configuration before learning starts. Output style
matches the ICROS ``build_env`` summary box (ANSI-coloured, fixed-width).
"""

from __future__ import annotations

import re
from typing import Any


# ANSI palette --------------------------------------------------------------
_H   = "\033[1;38;5;75m"    # box border (bold cyan-blue)
_K   = "\033[38;5;245m"     # key (gray)
_V   = "\033[1;97m"         # value (bold white)
_ON  = "\033[1;38;5;114m"   # ON (bold green)
_OFF = "\033[38;5;203m"     # OFF (red)
_WR  = "\033[38;5;214m"     # warn / section header (orange)
_DM  = "\033[2;37m"         # dim separator
_R   = "\033[0m"            # reset

_COLOR = {"v": _V, "on": _ON, "off": _OFF, "w": _WR, "k": _K}
_W = 78  # box inner width


def _vlen(s: str) -> int:
    return len(re.sub(r"\033\[[^m]*m", "", s))


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _vlen(text))


def _box_line(left: str, fill: str, right: str) -> None:
    print(f"{_H}{left}{fill * _W}{right}{_R}")


def _box_top(title: str = "") -> None:
    _box_line("┌", "─", "┐")
    if title:
        print(f"{_H}│{_R}{_pad(f'  {_H}{title}{_R}', _W)}{_H}│{_R}")
        _box_line("├", "─", "┤")


def _box_bottom() -> None:
    _box_line("└", "─", "┘")


def _row(key: str, val: Any, color: str = "v", indent: int = 2) -> None:
    c = _COLOR.get(color, _V)
    val_str = str(val)
    key_str = f"{' ' * indent}{_K}{key}{_R}"
    key_visible = indent + len(key)
    gap = max(2, 38 - key_visible)
    max_val_len = _W - key_visible - gap - 1
    if len(val_str) > max_val_len:
        val_str = val_str[: max_val_len - 3] + "..."
    line = key_str + " " * gap + f"{c}{val_str}{_R}"
    print(f"{_H}│{_R}{_pad(line, _W)}{_H}│{_R}")


def _separator() -> None:
    inner = f"  {'─' * (_W - 4)}  "
    print(f"{_H}│{_DM}{inner}{_R}{_H}│{_R}")


def _section(title: str) -> None:
    content = f"  {_WR}▸ {title}{_R}"
    print(f"{_H}│{_R}{_pad(content, _W)}{_H}│{_R}")


def _bool_tag(b: bool) -> tuple[str, str]:
    return ("ON", "on") if b else ("OFF", "off")


def _dict_rows(cfg: Any, indent: int = 4) -> None:
    """Recursively print attributes of a dict / object config."""
    if isinstance(cfg, dict):
        items = dict(cfg)
    elif hasattr(cfg, "__dict__"):
        items = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    else:
        return

    for k, v in items.items():
        if isinstance(v, dict):
            _row(k, "", "k", indent)
            for sk, sv in v.items():
                if isinstance(sv, dict):
                    _row(sk, "", "k", indent + 4)
                    for ssk, ssv in sv.items():
                        _row(ssk, ssv, "v", indent + 8)
                else:
                    _row(sk, sv, "v", indent + 4)
        elif isinstance(v, (list, tuple)):
            _row(k, v, "v", indent)
        else:
            _row(k, v, "v", indent)


def print_pe_train_summary(
    *,
    title: str,
    ego: str,
    opponent_label: str,
    opponent_mode: str,
    opponent_ckpt: str | None,
    num_envs: int,
    max_iterations: int,
    seed: int,
    backend: str,
    device: str,
    logger: str,
    show_viewer: bool,
    wall_clock_budget_min: float,
    iters_per_chunk: int | None,
    resume: str | None,
    run_name: str,
    log_dir: Any,
    drone_yaml: str,
    env_yaml: str,
    train_yaml: str,
    env_cfg: Any,
    obs_cfg: Any,
    reward_cfg: Any,
    command_cfg: Any,
) -> None:
    """Echo the resolved phase-1.x training configuration."""

    print()
    _box_top(title)

    _section("TRAINING")
    _row("ego", ego, "w")
    _row("num_envs", num_envs)
    _row("max_iterations", max_iterations)
    _row("seed", seed)
    _row("backend", backend)
    _row("device", device)
    _row("logger", logger)
    bv, bc = _bool_tag(show_viewer)
    _row("show_viewer", bv, bc)
    _row(
        "wall_clock_budget_min",
        wall_clock_budget_min if wall_clock_budget_min > 0 else "off",
    )
    _row(
        "iters_per_chunk",
        iters_per_chunk if iters_per_chunk is not None else "= max_iterations",
    )
    if resume:
        _row("resume", resume, "w")

    _separator()
    _section("OPPONENT POLICY")
    _row(opponent_label, opponent_mode, "w")
    if opponent_mode == "pretrained":
        _row(
            "opponent_ckpt",
            opponent_ckpt or "<missing>",
            "v" if opponent_ckpt else "off",
        )

    _separator()
    _section("RUN PATHS")
    _row("run_name", run_name)
    _row("log_dir", str(log_dir))
    _row("env_yaml", env_yaml)
    _row("train_yaml", train_yaml)
    _row("drone_yaml", drone_yaml)

    for name, cfg in [
        ("env_cfg", env_cfg),
        ("obs_cfg", obs_cfg),
        ("reward_cfg", reward_cfg),
        ("command_cfg", command_cfg),
    ]:
        _separator()
        _section(name.upper())
        _dict_rows(cfg)

    _box_bottom()
    print()


def print_pe_train_complete(
    *,
    title: str,
    final_model_pt: Any | None = None,
    final_model_alias: Any | None = None,
    actor_pt: Any | None = None,
    actor_alias: Any | None = None,
    skipped_reason: str | None = None,
) -> None:
    """Pretty-print end-of-training summary box.

    Use after train_attacker.py / train_defender.py finish — replaces the
    flat ``[ckpt]`` / ``[actor]`` prints with a single bordered block.
    """
    print()
    _box_top(title)

    _section("FROZEN ACTOR (downstream consumer)")
    if actor_pt is not None:
        _row("actor_pt", str(actor_pt))
        if actor_alias is not None:
            _row("alias", str(actor_alias))
    else:
        reason = skipped_reason or "missing"
        _row("actor_pt", f"SKIPPED ({reason})", "off")

    _separator()
    _section("PPO STATE (resumable)")
    if final_model_pt is not None:
        _row("model_pt", str(final_model_pt))
        if final_model_alias is not None:
            _row("alias", str(final_model_alias))
    else:
        _row("model_pt", "MISSING", "off")

    _box_bottom()
    print()


def print_resume_loaded(*, role: str, path: Any) -> None:
    """Compact one-line resume confirmation (kept short by design — a full
    box for a single fact is overkill).
    """
    print(f"\n{_H}▸{_R} {_WR}[resume]{_R} {role}: {_V}{path}{_R}\n")
