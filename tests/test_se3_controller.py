"""
SE3 Geometric Controller 인터랙티브 테스트 (Genesis AI)

키보드로 목표 위치를 이동시키면 SE3 제어기가 추종합니다.

조작:
    ↑ / ↓       : 전진 / 후진 (Y축)
    ← / →       : 좌 / 우 (X축)
    E           : 상승 (Z+)
    Q           : 하강 (Z-)
    ESC         : 종료

사용법:
    python scripts/test_se3_controller.py
    python scripts/test_se3_controller.py --drone nova
    python scripts/test_se3_controller.py --headless
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind
from utils.drone_params import DroneParams
from controllers.se3_controller import SE3Controller
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
        title=f"SE3 — {params.name}",
        window=500,
        update_interval=10,
    ) if not args.no_plot else None

    # ------------------------------------------------------------------
    # 4. 목표 위치 및 키바인딩
    # ------------------------------------------------------------------
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    step_xy = 0.05
    step_z  = 0.05
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
    # 5. SE3 제어기 초기화
    # ------------------------------------------------------------------
    controller = SE3Controller(params=params, dt=0.01)
    KF = params.kf  # 추력 계수 (YAML 기반)

    # ------------------------------------------------------------------
    # 6. 시뮬레이션 루프
    # ------------------------------------------------------------------
    print(f"\n[SE3 Test] 드론: {params.name}  |  hover_pwm={params.hover_pwm:.0f}µs")
    print("  ↑↓←→: 목표 이동 (XY)  |  E: 상승  |  Q: 하강  |  ESC: 종료\n")
    print(f"{'Step':>6}  {'Target':>18}  {'Pos':>24}  {'Alt Err':>8}  {'Thrusts [N]':>26}")

    try:
        step = 0
        while is_running:
            # 목표 마커 위치 업데이트
            target_marker.set_pos(target, zero_velocity=True)

            # 현재 상태 읽기
            pos  = drone.get_pos().cpu().numpy().flatten().astype(np.float64)
            quat = drone.get_quat().cpu().numpy().flatten().astype(np.float64)
            vel  = drone.get_vel().cpu().numpy().flatten().astype(np.float64)
            ang  = drone.get_ang().cpu().numpy().flatten().astype(np.float64)

            if plotter is not None:
                plotter.update(step, pos, target)

            # SE3 제어 계산: Thrust [N] → RPM
            controller.set_pose(pos, quat)
            controller.set_twist(vel, ang)
            controller.set_target(pos_des=target, yaw_des=0.0)
            thrusts = controller.compute()                               # [N]
            rpms    = np.sqrt(np.clip(thrusts, 0, None) / KF)          # [RPM]
            drone.set_propellers_rpm(rpms)

            scene.step()

            if step % 50 == 0:
                alt_err = target[2] - pos[2]
                print(f"{step:>6}  "
                      f"[{target[0]:+.2f},{target[1]:+.2f},{target[2]:+.2f}]  "
                      f"[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}]  "
                      f"{alt_err:>+8.3f}m  "
                      f"[{thrusts[0]:.3f},{thrusts[1]:.3f},{thrusts[2]:.3f},{thrusts[3]:.3f}]")
            step += 1

    except KeyboardInterrupt:
        gs.logger.info("종료됨.")
    finally:
        if plotter is not None:
            plotter.close()
        gs.logger.info("SE3 테스트 종료.")


if __name__ == "__main__":
    main()
