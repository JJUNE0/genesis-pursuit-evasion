"""
실시간 드론 상태 플로터 (matplotlib ion 모드)

Usage:
    plotter = RealtimePlotter("Nova PID")
    # 루프 내:
    plotter.update(step, pos, target)
    # 종료 시:
    plotter.close()
"""
from __future__ import annotations

import numpy as np
from collections import deque

try:
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False


class RealtimePlotter:
    """
    매 step 호출하면 X/Y/Z 위치 및 고도 오차를 실시간으로 그립니다.
    matplotlib 미설치 시 자동으로 no-op으로 동작합니다.
    """

    def __init__(
        self,
        title: str = "Drone State",
        window: int = 500,
        update_interval: int = 10,
    ):
        """
        Parameters
        ----------
        title           : 그래프 창 제목
        window          : 표시할 최근 스텝 수 (rolling window)
        update_interval : N 스텝마다 그래프 갱신 (성능 조절)
        """
        self._ok = _MPL_OK
        if not self._ok:
            print("[RealtimePlotter] matplotlib 없음 — 플롯 비활성화.")
            return

        self._interval = update_interval
        n = window

        self._steps: deque[int]   = deque(maxlen=n)
        self._px:    deque[float] = deque(maxlen=n)
        self._py:    deque[float] = deque(maxlen=n)
        self._pz:    deque[float] = deque(maxlen=n)
        self._tx:    deque[float] = deque(maxlen=n)
        self._ty:    deque[float] = deque(maxlen=n)
        self._tz:    deque[float] = deque(maxlen=n)

        plt.ion()
        self._fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
        self._fig.suptitle(title, fontsize=11)
        self._ax_x, self._ax_y, self._ax_z, self._ax_e = axs

        for ax, lbl in zip(axs, ["X [m]", "Y [m]", "Z [m]", "Alt Err [m]"]):
            ax.set_ylabel(lbl, fontsize=8)
            ax.grid(True, alpha=0.3)
        axs[-1].set_xlabel("Step", fontsize=8)
        axs[-1].axhline(0, color="k", lw=0.5)

        self._fig.tight_layout()
        self._fig.canvas.draw()
        plt.show(block=False)

    def update(self, step: int, pos: np.ndarray, target: np.ndarray) -> None:
        """
        Parameters
        ----------
        step   : 현재 시뮬레이션 스텝
        pos    : np.ndarray (3,) — 현재 드론 위치 [x, y, z]
        target : np.ndarray (3,) — 목표 위치 [x, y, z]
        """
        if not self._ok:
            return

        self._steps.append(step)
        self._px.append(float(pos[0]));    self._tx.append(float(target[0]))
        self._py.append(float(pos[1]));    self._ty.append(float(target[1]))
        self._pz.append(float(pos[2]));    self._tz.append(float(target[2]))

        if step % self._interval != 0:
            return

        s = list(self._steps)
        for ax, actual, tgt, lbl in [
            (self._ax_x, list(self._px), list(self._tx), "X [m]"),
            (self._ax_y, list(self._py), list(self._ty), "Y [m]"),
            (self._ax_z, list(self._pz), list(self._tz), "Z [m]"),
        ]:
            ax.clear()
            ax.set_ylabel(lbl, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.plot(s, actual, "b-",  lw=1, label="actual")
            ax.plot(s, tgt,    "r--", lw=1, label="target")
            ax.legend(loc="upper right", fontsize=7)

        alt_err = [tz - pz for tz, pz in zip(self._tz, self._pz)]
        self._ax_e.clear()
        self._ax_e.set_ylabel("Alt Err [m]", fontsize=8)
        self._ax_e.set_xlabel("Step", fontsize=8)
        self._ax_e.grid(True, alpha=0.3)
        self._ax_e.axhline(0, color="k", lw=0.5)
        self._ax_e.plot(s, alt_err, "g-", lw=1)
        self._ax_e.fill_between(s, alt_err, 0, alpha=0.15, color="g")

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        if not self._ok:
            return
        plt.ioff()
        plt.close(self._fig)
