"""Verify the scene XML loads, steps cleanly, and renders a frame."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(BASE_DIR, "xmls/vega_right_arm_scene.xml")
OUT_IMG = os.path.join(BASE_DIR, "scene_verify.png")


def main():
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    print(f"Loaded scene: {m.njnt} joints, {m.nbody} bodies, {m.nu} actuators")

    # Print actuator names
    print("\nActuators:")
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = m.actuator_ctrlrange[i]
        print(f"  {i:2d}  {name:20s}  [{lo:.3f}, {hi:.3f}]")

    # Print site names
    print("\nSites:")
    for i in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
        print(f"  {i:2d}  {name}")

    # Step simulation and check for instability
    mujoco.mj_resetDataKeyframe(m, d, 1)  # grasp_ready keyframe
    print(f"\nqpos after keyframe reset (length={len(d.qpos)}):")
    print(f"  Lift={d.qpos[0]:.3f}  L_arm_j2={d.qpos[6]:.3f}  R_arm_j2={d.qpos[24]:.3f}")

    print("Stepping 100 steps...")
    for i in range(100):
        mujoco.mj_step(m, d)
        if np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)):
            print(f"  NaN detected at step {i}!")
            break
    else:
        print(f"  Stable. Lift={d.qpos[0]:.3f}  L_arm_j2={d.qpos[6]:.3f}  R_arm_j2={d.qpos[24]:.3f}")

    # Render video with adjusted free camera
    renderer = mujoco.Renderer(m, height=480, width=640)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.5
    cam.elevation = -20.0
    cam.azimuth = 160.0       # shifted left
    cam.lookat[:] = [0.4, 0.0, 0.9]

    import imageio
    out_video = OUT_IMG.replace(".png", ".mp4")
    frames = []
    mujoco.mj_resetDataKeyframe(m, d, 1)  # grasp_ready keyframe
    for _ in range(300):  # 10 seconds at 30fps
        mujoco.mj_step(m, d)
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render())

    imageio.mimwrite(out_video, frames, fps=30)
    print(f"Saved video: {out_video}")

    # Close-up from palmar side — camera looks at palm face-on
    palm_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm_site")
    mujoco.mj_resetDataKeyframe(m, d, 1)
    mujoco.mj_forward(m, d)
    palm_pos = d.site_xpos[palm_site_id].copy()

    cam_hand = mujoco.MjvCamera()
    cam_hand.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_hand.distance = 0.3
    cam_hand.elevation = 0.0
    cam_hand.azimuth = 250.0   # from +X side, slightly left
    cam_hand.lookat[:] = palm_pos

    hand_frames = []
    mujoco.mj_resetDataKeyframe(m, d, 1)
    for _ in range(90):
        mujoco.mj_step(m, d)
        renderer.update_scene(d, camera=cam_hand)
        hand_frames.append(renderer.render())

    hand_video = out_video.replace(".mp4", "_hand_closeup.mp4")
    imageio.mimwrite(hand_video, hand_frames, fps=30)
    print(f"Saved hand close-up: {hand_video}")


if __name__ == "__main__":
    main()
