"""
PID Controller 인터랙티브 테스트 (Genesis AI)

키보드로 목표 위치를 이동시키면 PID 제어기가 추종합니다.

조작:
    ↑ / ↓       : 전진 / 후진 (Y축)
    ← / →       : 좌 / 우 (X축)
    E           : 상승 (Z+)
    Q           : 하강 (Z-)
    ESC         : 종료

사용법:
    python scripts/test_pid_controller.py
    python scripts/test_pid_controller.py --drone nova
    python scripts/test_pid_controller.py --headless
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind
from utils.drone_params import DroneParams
from controllers.pid_controller import DronePIDController
from utils.realtime_plotter import RealtimePlotter


DRONE_CONFIGS = {
    "cf2x": "configs/cf2x/drone_params.yaml",
    "nova":  "configs/nova/drone_params.yaml",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--drone", default="cf2x", choices=list(DRONE_CONFIGS))
    parser.add_argument("--no-plot", action="store_true", help="실시간 그래프 비활성화")
    args = parser.parse_args()

    show_viewer = not args.headless

    # ------------------------------------------------------------------
    # 1. 파라미터 로드
    # ------------------------------------------------------------------
    params = DroneParams(DRONE_CONFIGS[args.drone])

    # ------------------------------------------------------------------
    # 2. Genesis 씬 구성
    # ------------------------------------------------------------------
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 2.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=45,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(show_world_frame=True),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        gs.morphs.Drone(
            file=params.urdf_path,
            pos=(0.0, 0.0, 0.5),
            propellers_link_name=params.prop_link_names,
            propellers_spin=[int(s) for s in params.spin_directions],
        )
    )
    target_marker = scene.add_entity(
        morph=gs.morphs.Sphere(radius=0.05, collision=False),
        surface=gs.surfaces.Rough(color=(1.0, 0.0, 0.0, 0.5)),
    )

    scene.build()

    # ------------------------------------------------------------------
    # 3. 실시간 플로터
    # ------------------------------------------------------------------
    plotter = RealtimePlotter(
        title=f"PID — {params.name}",
        window=500,
        update_interval=10,
    ) if not args.no_plot else None

    # ------------------------------------------------------------------
    # 4. 목표 위치 및 키바인딩
    # ------------------------------------------------------------------
    target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    step_xy = 0.05   # [m] 수평 이동 단위
    step_z  = 0.05   # [m] 수직 이동 단위
    is_running = True

    def move_target(delta: np.ndarray):
        target[:] = np.clip(target + delta, [-3, -3, 0.2], [3, 3, 3])

    def stop():
        nonlocal is_running
        is_running = False

    if show_viewer:
        scene.viewer.register_keybinds(
            Keybind(name="forward",  key=Key.UP,     key_action=KeyAction.HOLD,    callback=move_target, args=(np.array([ 0,  step_xy, 0]),)),
            Keybind(name="backward", key=Key.DOWN,   key_action=KeyAction.HOLD,    callback=move_target, args=(np.array([ 0, -step_xy, 0]),)),
            Keybind(name="left",     key=Key.LEFT,   key_action=KeyAction.HOLD,    callback=move_target, args=(np.array([-step_xy, 0, 0]),)),
            Keybind(name="right",    key=Key.RIGHT,  key_action=KeyAction.HOLD,    callback=move_target, args=(np.array([ step_xy, 0, 0]),)),
            Keybind(name="up",       key=Key.E,      key_action=KeyAction.HOLD,    callback=move_target, args=(np.array([ 0, 0,  step_z]),)),
            Keybind(name="down",     key=Key.Q,      key_action=KeyAction.HOLD,    callback=move_target, args=(np.array([ 0, 0, -step_z]),)),
            Keybind(name="quit",     key=Key.ESCAPE, key_action=KeyAction.RELEASE, callback=stop),
        )

    # ------------------------------------------------------------------
    # 5. 제어기 초기화
    # ------------------------------------------------------------------
    controller = DronePIDController(drone=drone, dt=0.01, params=params,
                                    pid_params=params.pid_gains)  # None이면 DEFAULT 사용
    KF = params.kf  # 추력 계수 (YAML 기반)

    # ------------------------------------------------------------------
    # 6. 시뮬레이션 루프
    # ------------------------------------------------------------------
    print(f"\n[PID Test] 드론: {params.name}  |  hover_pwm={params.hover_pwm:.0f}µs")
    print("  ↑↓←→: 목표 이동 (XY)  |  E: 상승  |  Q: 하강  |  ESC: 종료\n")
    print(f"{'Step':>6}  {'Target':>18}  {'Pos':>24}  {'Alt Err':>8}")

    try:
        step = 0
        while is_running:
            # 목표 마커 위치 업데이트
            target_marker.set_pos(target, zero_velocity=True)

            # 제어 계산: PWM → Thrust → RPM
            pwm    = controller.update(target)
            thrusts = params.pwm_to_thrust(pwm)                         # [N]
            rpms   = np.sqrt(np.clip(thrusts, 0, None) / KF)           # [RPM]
            drone.set_propellers_rpm(rpms)

            scene.step()

            pos = drone.get_pos().cpu().numpy().flatten()
            if plotter is not None:
                plotter.update(step, pos, target)

            if step % 50 == 0:
                alt_err = target[2] - pos[2]
                print(f"{step:>6}  "
                      f"[{target[0]:+.2f},{target[1]:+.2f},{target[2]:+.2f}]  "
                      f"[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}]  "
                      f"{alt_err:>+8.3f}m")
            step += 1

    except KeyboardInterrupt:
        gs.logger.info("종료됨.")
    finally:
        if plotter is not None:
            plotter.close()
        gs.logger.info("PID 테스트 종료.")


if __name__ == "__main__":
    main()
