"""PPO training for VegaPickYCB using the softgrasp reward specification.

Uses envs.vega_pick_ycb_softgrasp.VegaPickYCB, which implements:
  - reach      = 1 - tanh(2 * palm_dist)
  - obj_touch  = mean(1 - tanh(10 * tip_dist))
  - grasp      = palm_gate * (clip(n_near/3, 0, 0.5) + clip(n_contact/2, 0, 0.5))
  - lift       = grasp_ok * height_factor * time_factor  (all in [0, 1])
  - hold       = grasp_active * clip(obj_height / 0.01, 0, 1)  (disabled by default)
  - success    = (held_steps >= 5)

Default scales match the spec's "suggested" values; override via CLI if sweeping.
"""
import os
import sys
import time
import functools
import datetime
import json
import argparse

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True --xla_gpu_autotune_level=0"
os.environ["XLA_FLAGS"] = xla_flags
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      "/home/csgrad/amanaror/WorkSpace/robotic_grasp/.jax_cache")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "mujoco_playground"))

import jax
import numpy as np
from ml_collections import config_dict

if not hasattr(jax, 'device_put_replicated'):
    def _device_put_replicated(x, devices):
        n = len(devices)
        x_rep = jax.tree_util.tree_map(
            lambda a: np.stack([np.asarray(a)] * n), x
        )
        return jax.device_put(x_rep, devices[0])
    jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from envs.vega_pick_ycb_softgrasp import VegaPickYCB, default_config as env_default_config

LOGDIR   = os.path.join(BASE_DIR, "logs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

_REWARD_COMPONENTS = [
    "reach", "default_pos", "progress", "obj_touch",
    "grasp", "lift", "hold", "success",
    "action_smooth", "jerk_smooth",
]


# ---------------------------------------------------------------------------
# Spec-default reward scales (softgrasp)
# ---------------------------------------------------------------------------

SOFTGRASP_DEFAULT_SCALES = dict(
    reach       = 0.5,   # spec range 0.3–1.0; 0.5 = middle
    default_pos = 0.0,
    progress    = 0.0,
    obj_touch   = 0.5,
    grasp       = 1.0,
    lift        = 3.0,
    hold        = 0.0,   # disabled per spec
    success     = 10.0,
    action_smooth = 0.0,
    jerk_smooth   = 0.0,
    fall_penalty  = 0.0,
)


def ppo_config():
    return config_dict.create(
        num_timesteps          = 50_000_000,
        num_evals              = 1000,
        episode_length         = 250,
        action_repeat          = 1,
        num_envs               = 1024,
        num_eval_envs          = 16,
        learning_rate          = 3e-4,
        entropy_cost           = 0.01,
        discounting            = 0.99,
        unroll_length          = 20,
        batch_size             = 512,
        num_minibatches        = 4,
        num_updates_per_batch  = 5,
        normalize_observations = True,
        reward_scaling         = 1.0,
        max_grad_norm          = 1.0,
        clipping_epsilon       = 0.3,
        network_factory = config_dict.create(
            policy_hidden_layer_sizes = (256, 256, 256),
            value_hidden_layer_sizes  = (256, 256, 256),
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="softgrasp_v1")
    ap.add_argument("--smoke", action="store_true",
                    help="Short run (5M steps) for pipeline check.")
    # Env / reward
    ap.add_argument("--ycb-object", default=None,
                    help="YCB name (e.g. potted_meat_can, banana, mug, foam_brick). "
                         "None = default placeholder cube.")
    ap.add_argument("--fixed-arm-init", action="store_true")
    for k, v in SOFTGRASP_DEFAULT_SCALES.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v,
                        help=f"Reward scale for {k} (default {v})")
    # PPO
    ap.add_argument("--num-timesteps", type=int, default=0,
                    help="Override PPO num_timesteps (0 = keep default).")
    ap.add_argument("--num-evals", type=int, default=0)
    ap.add_argument("--num-envs",  type=int, default=0)
    ap.add_argument("--entropy",   type=float, default=0.0)
    ap.add_argument("--clipping-eps", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--reward-scaling", type=float, default=0.0)
    ap.add_argument("--restore", default="")
    args = ap.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name  = f"VegaPickYCB_PPO_softgrasp-{timestamp}-{args.name}"
    if args.smoke:
        exp_name += "-smoke"
    ckpt_path = os.path.join(CKPT_DIR, exp_name)
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Experiment : {exp_name}")
    print(f"Checkpoint : {ckpt_path}")

    # ---- Env config ----
    env_cfg = env_default_config()
    # Override default scales with the spec's suggested values (or CLI overrides).
    for k in SOFTGRASP_DEFAULT_SCALES.keys():
        setattr(env_cfg.reward_config.scales, k, float(getattr(args, k)))
    if args.fixed_arm_init:
        env_cfg.fixed_arm_init = True
    if args.ycb_object:
        env_cfg.ycb_object = args.ycb_object

    print("\nReward scales (softgrasp spec):")
    for k in SOFTGRASP_DEFAULT_SCALES.keys():
        print(f"  {k:15s} = {getattr(env_cfg.reward_config.scales, k):.3f}")
    print(f"  ycb_object    = {env_cfg.ycb_object}")
    print(f"  fixed_arm_init= {env_cfg.fixed_arm_init}")

    env      = VegaPickYCB(env_cfg)
    eval_env = VegaPickYCB(env_cfg)

    # ---- PPO params ----
    params = ppo_config()
    if args.smoke:
        params.num_timesteps = 5_000_000
        params.num_evals     = 5
    if args.num_timesteps > 0:   params.num_timesteps   = args.num_timesteps
    if args.num_evals     > 0:   params.num_evals       = args.num_evals
    if args.num_envs      > 0:   params.num_envs        = args.num_envs
    if args.entropy       > 0.0: params.entropy_cost    = args.entropy
    if args.clipping_eps  > 0.0: params.clipping_epsilon= args.clipping_eps
    if args.lr            > 0.0: params.learning_rate   = args.lr
    if args.reward_scaling> 0.0: params.reward_scaling  = args.reward_scaling

    with open(os.path.join(ckpt_path, "config.json"), "w") as f:
        json.dump({"env": env_cfg.to_dict(), "ppo": params.to_dict()}, f, indent=2)

    # ---- Network ----
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **params.network_factory,
    )

    # ---- Progress callback ----
    times          = [time.monotonic()]
    reward_log_path = os.path.join(ckpt_path, "rewards.jsonl")
    _steps_per_iter = params.num_envs * params.unroll_length
    active_scales   = env_cfg.reward_config.scales

    def progress(num_steps, metrics):
        times.append(time.monotonic())
        r     = metrics.get("eval/episode_reward", float("nan"))
        r_std = metrics.get("eval/episode_reward_std", float("nan"))
        elapsed = times[-1] - times[0]
        sps = num_steps / elapsed if elapsed > 0 else 0
        iteration = num_steps // _steps_per_iter

        parts = {c: float(metrics.get(f"eval/episode_{c}", float("nan")))
                 for c in _REWARD_COMPONENTS}
        comp_str = "  ".join(f"{c}={v:+.3f}" for c, v in parts.items()
                              if getattr(active_scales, c, 0.0) != 0.0)
        p_loss  = metrics.get("training/policy_loss",  float("nan"))
        v_loss  = metrics.get("training/v_loss",        float("nan"))
        entropy = metrics.get("training/entropy_loss",  float("nan"))
        grad_n  = metrics.get("training/grad_norm",     float("nan"))
        kl      = metrics.get("training/kl_mean",       float("nan"))
        lr      = metrics.get("training/learning_rate", float("nan"))

        print(f"  iter={iteration:>6,}  step={num_steps:>10,}  "
              f"reward={r:+.3f} ± {r_std:.3f}  "
              f"elapsed={elapsed:.0f}s  steps/s={sps:.0f}")
        print(f"    {comp_str}")
        print(f"    p_loss={p_loss:.4f}  v_loss={v_loss:.4f}  "
              f"entropy={entropy:.4f}  grad_norm={grad_n:.4f}  "
              f"kl={kl:.5f}  lr={lr:.2e}")

        with open(reward_log_path, "a") as f:
            json.dump({
                "iter": iteration, "step": num_steps,
                "reward": float(r), "reward_std": float(r_std),
                **parts,
                "policy_loss": float(p_loss), "value_loss": float(v_loss),
                "entropy": float(entropy), "grad_norm": float(grad_n),
                "kl": float(kl), "lr": float(lr),
                "elapsed": elapsed, "sps": sps,
            }, f)
            f.write("\n")

    # ---- Train ----
    _EXPLICIT = {"network_factory"}
    train_params = {k: v for k, v in params.items() if k not in _EXPLICIT}

    print(f"\nStarting PPO — {params.num_timesteps:,} steps, "
          f"{params.num_envs} envs  lr={params.learning_rate:.1e}  "
          f"entropy={params.entropy_cost}\n")

    restore_kwargs = {}
    if args.restore:
        restore_kwargs["restore_checkpoint_path"] = args.restore
        print(f"Restoring from: {args.restore}")

    make_inference_fn, trained_params, _ = ppo.train(
        environment      = env,
        eval_env         = eval_env,
        **train_params,
        network_factory  = network_factory,
        wrap_env_fn      = wrapper.wrap_for_brax_training,
        progress_fn      = progress,
        **restore_kwargs,
    )

    # ---- Save final params ----
    import flax.serialization
    final_path = os.path.join(ckpt_path, "final_params.msgpack")
    with open(final_path, "wb") as f:
        f.write(flax.serialization.to_bytes(trained_params))
    print(f"\nSaved final params → {final_path}")


if __name__ == "__main__":
    main()
