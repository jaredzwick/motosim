"""
Headless debug: load the scene, place the bike at various x positions,
print what perception sees and where it projects the cones to.

Run with: python3 debug_perception.py
No viewer. No mjpython needed.
"""

import numpy as np
import mujoco
from perception import detect_cones_hsv, project_detections_to_world


FPV_WIDTH, FPV_HEIGHT = 640, 480
TARGET_SPEED = 5.0
WHEEL_RADIUS = 0.312


def debug_at_bike_x(model, data, renderer, fpv_cam_id, fpv_fovy_rad, bike_x: float):
    """Place the bike at a given x and print everything perception sees."""
    # Teleport bike to bike_x
    data.qpos[0] = bike_x
    data.qpos[1] = 0.0
    # keep upright quaternion
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    # cruise speed
    data.qvel[0] = TARGET_SPEED
    omega = TARGET_SPEED / WHEEL_RADIUS
    data.joint("rear_spin").qvel[0] = omega
    data.joint("front_spin").qvel[0] = omega

    mujoco.mj_forward(model, data)

    cam_pos = data.cam_xpos[fpv_cam_id].copy()
    cam_xmat = data.cam_xmat[fpv_cam_id].reshape(3, 3).copy()

    renderer.update_scene(data, camera="fpv")
    frame = renderer.render()

    dets = detect_cones_hsv(frame)
    project_detections_to_world(
        dets,
        img_width=FPV_WIDTH,
        img_height=FPV_HEIGHT,
        fovy_rad=fpv_fovy_rad,
        cam_pos=cam_pos,
        cam_xmat=cam_xmat,
        cone_top_z=0.65,
    )

    print(f"\n=== bike_x={bike_x:+.1f} ===")
    print(f"  camera world pos: ({cam_pos[0]:+.3f}, {cam_pos[1]:+.3f}, {cam_pos[2]:+.3f})")
    print(f"  camera z-axis (look dir = -z): points at ({-cam_xmat[0,2]:+.3f}, {-cam_xmat[1,2]:+.3f}, {-cam_xmat[2,2]:+.3f})")
    print(f"  {len(dets)} HSV detections:")
    for i, d in enumerate(dets):
        x, y, w, h = d.bbox
        wxy = d.world_xy if d.world_xy is not None else ("None", "None")
        wxs = f"{wxy[0]:+6.2f}" if d.world_xy else "  None"
        wys = f"{wxy[1]:+6.2f}" if d.world_xy else "  None"
        print(f"    det {i}: bbox=({x:3d},{y:3d}, w={w:3d},h={h:3d}, area={d.area:5d}, AR={d.aspect_ratio:.2f}) "
              f"-> world=({wxs}, {wys})")

    # Expected cone positions from scene.xml
    expected = [(5, 0, "cone-0 was-5"), (15, 0, "cone-0"), (10, 2, "cone-1"), (17, -2, "cone-2"),
                (24, 2, "cone-3"), (31, -2, "cone-4")]
    ahead = [(ex, ey, name) for ex, ey, name in expected if ex > bike_x]
    print(f"  cones ahead per scene.xml (truth):")
    for ex, ey, name in sorted(ahead, key=lambda t: t[0])[:4]:
        print(f"    truth {name}: world=({ex:+6.2f}, {ey:+6.2f}) dist={ex-bike_x:.1f}")


def main():
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    fpv_cam_id = model.camera("fpv").id
    fpv_fovy_rad = np.deg2rad(float(model.cam_fovy[fpv_cam_id]))

    renderer = mujoco.Renderer(model, width=FPV_WIDTH, height=FPV_HEIGHT)

    print("=" * 70)
    print(f"FPV camera: fovy={np.rad2deg(fpv_fovy_rad):.1f} deg, resolution {FPV_WIDTH}x{FPV_HEIGHT}")
    print(f"IPM projects bbox-TOP onto z=0.7 plane (assumed cone top height)")
    print("=" * 70)

    for bike_x in [0.0, 3.0, 6.0, 9.0, 12.0]:
        debug_at_bike_x(model, data, renderer, fpv_cam_id, fpv_fovy_rad, bike_x)


if __name__ == "__main__":
    main()
