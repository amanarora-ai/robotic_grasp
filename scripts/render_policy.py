"""Render SAC policy episodes from a saved checkpoint.

Run in a fresh process (separate from any JAX training process) to avoid os.fork deadlock.
Usage:
  python scripts/render_policy.py --ckpt checkpoints/<run_dir> [--out videos/out.mp4] [--n 4]
"""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, argparse, json
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "mujoco_playground"))

import numpy as np
import jax
import jax.numpy as jnp
import flax
import mujoco
import imageio

from brax.training.agents.sac import networks as sac_networks
from brax.training.acme import running_statistics, specs

ACTION_DIM = 18

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Checkpoint directory (default: latest HER run)")
    parser.add_argument("--out",  default=None,  help="Output mp4 path")
    parser.add_argument("--n",    type=int, default=4, help="Number of episodes")
    parser.add_argument("--fps",  type=int, default=20)
    parser.add_argument("--tag",  default="latest", help="Params tag: latest or best")
    parser.add_argument("--ycb-object", default=None,
                        help="Override YCB object: potted_meat_can | banana | mug | foam_brick. "
                             "Use to run a cube-trained policy on a real YCB object.")
    args = parser.parse_args()

    if args.ckpt is None:
        import glob
        ckpt_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "checkpoints", "*HER*")), key=os.path.getmtime)
        if not ckpt_dirs:
            ckpt_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "checkpoints", "*")), key=os.path.getmtime)
        ckpt_dir = ckpt_dirs[-1]
        print(f"Auto-selected: {ckpt_dir}")
    else:
        ckpt_dir = args.ckpt
    if not os.path.isabs(ckpt_dir):
        # Accept either bare run name or full relative path
        candidate = os.path.join(BASE_DIR, ckpt_dir)
        if not os.path.isdir(candidate):
            candidate = os.path.join(BASE_DIR, "checkpoints", ckpt_dir)
        ckpt_dir = candidate

    if args.out:
        out_path = args.out
    else:
        run_name = os.path.basename(ckpt_dir)
        out_path = os.path.join(BASE_DIR, "videos", f"{run_name}_{args.tag}.mp4")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load config
    cfg_path = os.path.join(ckpt_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    print(f"Config: {cfg}")

    # Build env — use goal-conditioned env for HER runs, plain env otherwise
    is_her = "HER" in os.path.basename(ckpt_dir)
    if is_her:
        from envs.vega_pick_ycb_goal import VegaPickYCBGoal, default_config, OBS_DIM
        env_cfg = default_config()
        env_cfg.fixed_arm_init     = cfg.get("fixed_arm_init", True)
        env_cfg.lift_target_height = cfg.get("lift_target_height", 0.01)
        env_cfg.reward_config.scales.reach   = cfg.get("reach_scale", 0.05)
        env_cfg.reward_config.scales.lift    = cfg.get("lift_scale", 4.0)
        env_cfg.reward_config.scales.hold    = 1.5
        env_cfg.reward_config.scales.success = cfg.get("success_scale", 20.0)
        env = VegaPickYCBGoal(env_cfg)
        obs_dim = OBS_DIM
    else:
        from envs.vega_pick_ycb import VegaPickYCB, default_config
        env_cfg = default_config()
        env_cfg.action_mode = '18d'
        env_cfg.reward_config.scales.obj_touch = cfg.get("obj_touch_scale", 0.5)
        env_cfg.reward_config.scales.reach     = cfg.get("reach_scale", 0.4)
        env_cfg.reward_config.scales.lift      = cfg.get("lift_scale", 1.2)
        env_cfg.reward_config.scales.success   = cfg.get("success_scale", 5.0)
        env_cfg.fixed_arm_init     = cfg.get("fixed_arm_init", False)
        env_cfg.ycb_object         = args.ycb_object or cfg.get("ycb_object", None)
        env_cfg.lift_target_height = cfg.get("lift_target_height", 0.05)
        env = VegaPickYCB(env_cfg)
        obs_dim = env.observation_size

    # Build networks (same architecture as training)
    sac_nets = sac_networks.make_sac_networks(
        observation_size   = obs_dim,
        action_size        = ACTION_DIM,
        hidden_layer_sizes = (256, 256, 256),
    )
    inference_fn = sac_networks.make_inference_fn(sac_nets)

    # Init dummy state to get shapes
    rng = jax.random.PRNGKey(0)
    dummy_policy_params = sac_nets.policy_network.init(rng)
    dummy_norm_params   = running_statistics.init_state(specs.Array((obs_dim,), jnp.float32))
    ckpt_template = {"policy_params": dummy_policy_params, "normalizer_params": dummy_norm_params}

    # Load checkpoint
    params_path = os.path.join(ckpt_dir, f"{args.tag}_params.msgpack")
    print(f"Loading: {params_path}")
    with open(params_path, "rb") as f:
        ckpt = flax.serialization.from_bytes(ckpt_template, f.read())
    params = (ckpt["normalizer_params"], ckpt["policy_params"])
    policy_fn = inference_fn(params)

    jit_reset = jax.jit(env.reset)
    jit_step  = jax.jit(env.step)
    jit_infer = jax.jit(policy_fn)

    # MuJoCo renderer
    mj_model = env.mj_model
    mj_data  = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = 1.8
    cam.elevation = -25.0
    cam.azimuth   = 155.0
    cam.lookat[:] = [0.55, 0.0, 0.75]

    all_frames = []
    ep_length  = env._config.episode_length

    for ep in range(args.n):
        rng, rng_reset = jax.random.split(rng)
        state  = jit_reset(rng_reset)
        frames = []
        total_r = 0.0
        for t in range(ep_length):
            rng, rng_act = jax.random.split(rng)
            action, _ = jit_infer(state.obs, rng_act)
            state = jit_step(state, action)
            total_r += float(state.reward)

            # Sync mujoco state for rendering
            mj_data.qpos[:] = np.array(state.data.qpos)
            mj_data.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_kinematics(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            frames.append(renderer.render().copy())

            if float(state.done) > 0.5:
                break

        obj_z  = float(state.data.xpos[env._obj_body][2])
        height = obj_z - float(env._table_surface_z) - float(env._obj_rest_offset)
        print(f"  ep={ep+1}/{args.n}  steps={len(frames)}  "
              f"total_r={total_r:.2f}  obj_height={height*100:.1f}cm")
        all_frames.extend(frames)

    renderer.close()

    print(f"\nSaving {len(all_frames)} frames → {out_path}")
    imageio.mimwrite(out_path, all_frames, fps=args.fps, quality=7)
    print("Done.")


if __name__ == "__main__":
    main()
