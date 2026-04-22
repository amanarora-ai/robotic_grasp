"""SAC with vmapped parallel env collection, no HER."""

import os, sys, time, datetime, json
from collections import deque
sys.stdout.reconfigure(line_buffering=True)  # flush every newline when piped

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True --xla_gpu_autotune_level=0"
os.environ["XLA_FLAGS"] = xla_flags

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "mujoco_playground"))

import signal
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax

from brax.training import gradients, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import losses as sac_losses
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac.train import TrainingState

_env_variant = os.environ.get("ENV_VARIANT", "")
if _env_variant == "tight":
    from envs.vega_pick_ycb_tight import VegaPickYCB, default_config
elif _env_variant == "softgrasp":
    from envs.vega_pick_ycb_softgrasp import VegaPickYCB, default_config
else:
    from envs.vega_pick_ycb import VegaPickYCB, default_config

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

SAC_CFG = dict(
    num_iterations        = 20_000,
    n_envs                = 256,
    episode_length        = 250,
    batch_size            = 512,
    replay_capacity       = 200_000,
    min_replay_size       = 10_000,
    grad_updates_per_iter = 200,      # scan3: 500 still caused Q-plateau; 200 = more stable Q-learning
    learning_rate         = 3e-4,
    lr_alpha              = 1e-4,
    discounting           = 0.99,
    tau                   = 0.005,
    reward_scaling        = 4.0,      # scan3: 2.0 Q-values too small; 4× = stronger gradient signal
    eval_every            = 100,
    video_every           = 20_001,  # disabled — os.fork + JAX threads = deadlock (imageio/ffmpeg uses fork)
    n_eval_episodes       = 16,
    seed                  = 42,
    explore_scale         = 1.0,
    obj_touch_scale       = 2.0,      # 4× default; strong fingertip gradient
    reach_scale           = 0.05,     # 0.4→0.05: reach was 57× stronger than grasp; now ~4× — prevents reach-only Q-collapse
    grasp_scale           = 1.0,
    lift_scale            = 0.0,
    hold_scale            = 1.0,
    success_scale         = 10.0,
    lift_target_height    = 0.01,    # 2D curriculum start height
    min_hold_steps        = 1.0,     # 2D curriculum start hold steps (1 = fire on first contact above target)
    policy_warmup         = True,
    target_entropy        = -6.0,     # scan3: -18 equilibrated at alpha=0.18; -6 allows lower entropy (more deterministic policy)
    fixed_arm_init        = True,     # arm always starts at grasp_ready; buffer fills near-grasp states
    ycb_object            = None,  # potted_meat_can caused MuJoCo NaN (766K mesh + random actions)
)

ACTION_DIM = 18  # arm (7) + full hand (11)


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------

def render_episode(policy_fn, env, out_path, rng_seed=0, fps=20, n_episodes=2):
    import imageio
    import mujoco

    jit_reset = jax.jit(env.reset)
    jit_step  = jax.jit(env.step)
    jit_infer = jax.jit(policy_fn)

    mj_model = env.mj_model
    mj_data  = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = 2.0
    cam.elevation = -20.0
    cam.azimuth   = 160.0
    cam.lookat[:] = [0.55, 0.0, 0.6]

    rng    = jax.random.PRNGKey(rng_seed)
    frames = []
    for _ in range(n_episodes):
        rng, rng_reset = jax.random.split(rng)
        state = jit_reset(rng_reset)
        for _ in range(env._config.episode_length):
            rng, rng_act = jax.random.split(rng)
            action, _ = jit_infer(state.obs, rng_act)
            state = jit_step(state, action)
            mj_data.qpos[:] = np.array(state.data.qpos)
            mujoco.mj_kinematics(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            frames.append(renderer.render())
            if float(state.done) > 0.5:
                break

    renderer.close()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps)
    print(f"  [video] {out_path}  ({len(frames)} frames)")

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.cap  = capacity
        self.ptr  = 0
        self.size = 0
        self.obs      = np.zeros((capacity, obs_dim),    np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    np.float32)
        self.action   = np.zeros((capacity, action_dim), np.float32)
        self.reward   = np.zeros(capacity, np.float32)
        self.done     = np.zeros(capacity, np.float32)

    def add(self, obs, nobs, act, rew, done):
        n    = len(rew)
        idxs = np.arange(self.ptr, self.ptr + n) % self.cap
        self.obs[idxs]      = obs
        self.next_obs[idxs] = nobs
        self.action[idxs]   = act
        self.reward[idxs]   = rew
        self.done[idxs]     = done
        self.ptr  = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (self.obs[idx], self.next_obs[idx],
                self.action[idx], self.reward[idx], self.done[idx])


# ---------------------------------------------------------------------------
# SAC update
# ---------------------------------------------------------------------------

def build_update_fn(sac_nets, obs_dim, learning_rate, tau, reward_scaling, discounting,
                    lr_alpha=None, target_entropy=None):
    if lr_alpha is None:
        lr_alpha = learning_rate
    if target_entropy is None:
        target_entropy = -0.5 * ACTION_DIM  # brax default = -9
    _, critic_loss_fn, actor_loss_fn = sac_losses.make_losses(
        sac_nets, reward_scaling=reward_scaling,
        discounting=discounting, action_size=ACTION_DIM,
    )
    # Custom alpha loss: identical to brax but with configurable target_entropy.
    # target_entropy=-18 creates negative feedback: if policy collapses to reach (H<18),
    # alpha increases, restoring exploration before reach-lock is permanent.
    _pol_net  = sac_nets.policy_network
    _pad      = sac_nets.parametric_action_distribution
    def alpha_loss_fn(log_alpha, policy_params, normalizer_params, transitions, key):
        dist = _pol_net.apply(normalizer_params, policy_params, transitions.observation)
        act  = _pad.sample_no_postprocessing(dist, key)
        lp   = _pad.log_prob(dist, act)
        return jnp.mean(jnp.exp(log_alpha) * jax.lax.stop_gradient(-lp - target_entropy))
    optimizer       = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    alpha_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_alpha))
    alpha_update  = gradients.gradient_update_fn(alpha_loss_fn,  alpha_optimizer, pmap_axis_name=None)
    critic_update = gradients.gradient_update_fn(critic_loss_fn, optimizer, pmap_axis_name=None)
    actor_update  = gradients.gradient_update_fn(actor_loss_fn,  optimizer, pmap_axis_name=None)

    def _make_transition(obs, nobs, act, rew, done):
        B = obs.shape[0]
        return types.Transition(
            observation      = obs,
            action           = act,
            reward           = rew,
            discount         = 1.0 - done,
            next_observation = nobs,
            extras={'state_extras': {'truncation': jnp.zeros(B)}},
        )

    # Un-jitted single step — called inside lax.scan by multi_update below.
    def _update_step(state, obs, nobs, act, rew, done, key):
        tr  = _make_transition(obs, nobs, act, rew, done)
        key, ka, kc, kp = jax.random.split(key, 4)
        alpha = jnp.exp(state.alpha_params)
        _, alpha_params, alpha_opt = alpha_update(
            state.alpha_params, state.policy_params,
            state.normalizer_params, tr, ka,
            optimizer_state=state.alpha_optimizer_state)
        _, q_params, q_opt = critic_update(
            state.q_params, state.policy_params,
            state.normalizer_params, state.target_q_params,
            alpha, tr, kc, optimizer_state=state.q_optimizer_state)
        _, policy_params, policy_opt = actor_update(
            state.policy_params, state.normalizer_params,
            state.q_params, alpha, tr, kp,
            optimizer_state=state.policy_optimizer_state)
        target_q = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            state.target_q_params, q_params)
        new_norm = running_statistics.update(state.normalizer_params, obs)
        return TrainingState(
            policy_optimizer_state = policy_opt,
            policy_params          = policy_params,
            q_optimizer_state      = q_opt,
            q_params               = q_params,
            target_q_params        = target_q,
            gradient_steps         = state.gradient_steps + 1,
            env_steps              = state.env_steps,
            alpha_optimizer_state  = alpha_opt,
            alpha_params           = alpha_params,
            normalizer_params      = new_norm,
        ), key

    # Single JIT call: runs all grad_updates steps via lax.scan on-device.
    # Replaces 200-iteration Python loop with one GPU-resident scan:
    #   - eliminates 200 × (numpy sample + 5 host→device transfers + JIT dispatch)
    #   - one large host→device transfer instead of 200 small ones
    #   - GPU pipelines all steps without Python stalls between them
    @jax.jit
    def multi_update(state, obs_B, nobs_B, act_B, rew_B, done_B, key):
        # obs_B shape: (grad_updates, batch_size, obs_dim) — already on device
        def _scan_step(carry, xs):
            s, k = carry
            s, k = _update_step(s, xs[0], xs[1], xs[2], xs[3], xs[4], k)
            return (s, k), None
        (state, key), _ = jax.lax.scan(
            _scan_step, (state, key),
            (obs_B, nobs_B, act_B, rew_B, done_B))
        return state, key

    def init(key, obs_dim):
        k1, k2    = jax.random.split(key)
        log_alpha = jnp.asarray(0.0)
        policy_p  = sac_nets.policy_network.init(k1)
        q_p       = sac_nets.q_network.init(k2)
        norm_p    = running_statistics.init_state(specs.Array((obs_dim,), jnp.float32))
        return TrainingState(
            policy_optimizer_state = optimizer.init(policy_p),
            policy_params          = policy_p,
            q_optimizer_state      = optimizer.init(q_p),
            q_params               = q_p,
            target_q_params        = q_p,
            gradient_steps         = jnp.zeros((), jnp.int32),
            env_steps              = jnp.zeros((), jnp.int32),
            alpha_optimizer_state  = alpha_optimizer.init(log_alpha),
            alpha_params           = log_alpha,
            normalizer_params      = norm_p,
        )

    return init, multi_update


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(name: str = ""):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name  = f"VegaPickYCB_SAC_vmap-{timestamp}"
    if name:
        exp_name += f"-{name}"
    ckpt_path = os.path.join(CKPT_DIR, exp_name)
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Experiment : {exp_name}")
    print(f"Checkpoint : {ckpt_path}")

    with open(os.path.join(ckpt_path, "config.json"), "w") as f:
        json.dump(SAC_CFG, f, indent=2)

    # ---- Env ----
    env_cfg = default_config()
    env_cfg.action_mode = '18d'
    env_cfg.reward_config.scales.obj_touch = SAC_CFG.get("obj_touch_scale", 0.5)
    env_cfg.reward_config.scales.reach     = SAC_CFG.get("reach_scale", 0.4)
    env_cfg.reward_config.scales.grasp     = SAC_CFG.get("grasp_scale", 1.0)
    env_cfg.reward_config.scales.lift      = SAC_CFG.get("lift_scale", 1.2)
    env_cfg.reward_config.scales.hold      = SAC_CFG.get("hold_scale", 0.5)
    env_cfg.reward_config.scales.success   = SAC_CFG.get("success_scale", 5.0)
    env_cfg.fixed_arm_init      = SAC_CFG.get("fixed_arm_init", False)
    env_cfg.ycb_object          = SAC_CFG.get("ycb_object", None)
    env_cfg.lift_target_height  = SAC_CFG.get("lift_target_height", 0.05)
    env_cfg.min_hold_steps      = SAC_CFG.get("min_hold_steps", 5.0)
    env     = VegaPickYCB(env_cfg)
    obs_dim = env.observation_size

    jit_vmap_reset = jax.jit(jax.vmap(env.reset))
    jit_vmap_step  = jax.jit(jax.vmap(env.step))
    _vmap_step_fn  = jax.vmap(env.step)   # bare vmap for use inside jit/scan

    # ---- Networks ----
    sac_nets     = sac_networks.make_sac_networks(
        observation_size   = obs_dim,
        action_size        = ACTION_DIM,
        hidden_layer_sizes = (256, 256, 256),
    )
    inference_fn = sac_networks.make_inference_fn(sac_nets)

    # ---- Training state ----
    rng = jax.random.PRNGKey(SAC_CFG["seed"])
    rng, rng_init = jax.random.split(rng)
    init_fn, update_fn = build_update_fn(
        sac_nets, obs_dim,
        learning_rate  = SAC_CFG["learning_rate"],
        tau            = SAC_CFG["tau"],
        reward_scaling = SAC_CFG["reward_scaling"],
        discounting    = SAC_CFG["discounting"],
        lr_alpha       = SAC_CFG["lr_alpha"],
        target_entropy = SAC_CFG.get("target_entropy"),
    )
    training_state = init_fn(rng_init, obs_dim)

    # ---- Replay buffer ----
    buf = ReplayBuffer(SAC_CFG["replay_capacity"], obs_dim, ACTION_DIM)

    # ---- Logging ----
    log_path       = os.path.join(ckpt_path, "rewards.jsonl")
    train_log_path = os.path.join(ckpt_path, "train_log.jsonl")
    vid_dir        = os.path.join(ckpt_path, "videos")
    os.makedirs(vid_dir, exist_ok=True)
    t0       = time.monotonic()
    total_transitions = 0
    best_eval = -float("inf")
    n_envs    = SAC_CFG["n_envs"]
    ep_length = SAC_CFG["episode_length"]

    def _save_params(tag="latest"):
        ckpt = {"policy_params": training_state.policy_params,
                "normalizer_params": training_state.normalizer_params}
        with open(os.path.join(ckpt_path, f"{tag}_params.msgpack"), "wb") as f:
            f.write(flax.serialization.to_bytes(ckpt))

    def _save_video(tag):
        try:
            params_ = (training_state.normalizer_params, training_state.policy_params)
            policy_fn = inference_fn(params_)
            render_episode(policy_fn, env, os.path.join(vid_dir, f"{tag}.mp4"))
        except Exception as e:
            print(f"  [video] skipped ({tag}): {e}")

    def _on_exit(signum=None, frame=None):
        print(f"\n  [exit] saving latest params (no video — os.fork deadlock risk)...")
        _save_params("latest")
        # _save_video("on_exit")  # DISABLED: os.fork deadlock with JAX multithreading
        if signum is not None:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_exit)
    signal.signal(signal.SIGINT,  _on_exit)

    # ---- Scan-based episode collection (GPU-resident, one transfer per iter) ----
    _expl_scale = SAC_CFG["explore_scale"]

    @jax.jit
    def _collect_policy(states, rng, params):
        """Roll out ep_length steps on GPU via lax.scan; transfer trajectory once."""
        def _step(carry, _):
            s, rng = carry
            rng, rng_act = jax.random.split(rng)
            rng_acts = jax.random.split(rng_act, n_envs)
            acts, _ = jax.vmap(lambda o, r: inference_fn(params)(o, r))(s.obs, rng_acts)
            s2 = _vmap_step_fn(s, acts)
            return (s2, rng), (s.obs, s2.obs, acts, s2.reward, s2.done, s2.metrics)
        (fs, fr), traj = jax.lax.scan(_step, (states, rng), None, length=ep_length)
        return fs, fr, traj

    @jax.jit
    def _collect_explore(states, rng):
        """Roll out ep_length steps with random exploration via lax.scan."""
        def _step(carry, _):
            s, rng = carry
            rng, rng_act = jax.random.split(rng)
            if _expl_scale >= 1.0:
                acts = jax.random.uniform(rng_act, shape=(n_envs, ACTION_DIM),
                                          minval=-1., maxval=1.)
            else:
                acts = jnp.clip(
                    _expl_scale * jax.random.normal(rng_act, shape=(n_envs, ACTION_DIM)),
                    -1., 1.)
            s2 = _vmap_step_fn(s, acts)
            return (s2, rng), (s.obs, s2.obs, acts, s2.reward, s2.done, s2.metrics)
        (fs, fr), traj = jax.lax.scan(_step, (states, rng), None, length=ep_length)
        return fs, fr, traj

    print(f"\nStarting SAC (vmap, no HER) — {SAC_CFG['num_iterations']:,} iters, "
          f"{n_envs} vmapped envs, obs_dim={obs_dim}\n")

    # NOTE: Initial random policy video DISABLED — os.fork deadlock with JAX multithreading.
    # The imageio.mimwrite mp4 writer uses ffmpeg subprocess (os.fork), which is incompatible
    # with JAX's multi-threaded runtime. Causes silent deadlock after video completes.
    # Video recording resumes at iter eval_every (eval videos only, no initial random video).
    # _save_video("iter_000000_random")  # DISABLED

    policy_warmup = SAC_CFG.get("policy_warmup", False)

    # Alpha floor + stage reset (mirrors train_sac_her_vmap.py)
    _alpha_lr_val    = SAC_CFG.get("lr_alpha", SAC_CFG["learning_rate"])
    _alpha_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(_alpha_lr_val))
    _ALPHA_FLOOR       = SAC_CFG.get("alpha_floor",       0.3)
    _ALPHA_STAGE_RESET = SAC_CFG.get("alpha_stage_reset", 0.7)
    _ALPHA_RESET_UNTIL = SAC_CFG.get("alpha_reset_until", 9)  # default 9 (2D: 4cm/4steps); use 3 for 1D (4cm)
    _NO_ALPHA_FIX    = SAC_CFG.get("no_alpha_fix", False)
    _NO_STAGE_RESET  = SAC_CFG.get("no_stage_reset", False)

    # 2D curriculum: (lift_target_m, min_hold_steps)
    # Height and hold-steps both increase gradually.
    # Starting at (1cm, 1 step) means success fires the first time object clears 1cm — very dense signal.
    # Graduation (Rapid Locomotion style): rolling mean success over window > threshold.
    if SAC_CFG.get("curriculum_1d", False):
        # 1D height-only curriculum — hold_steps fixed at 5
        _2D_STAGES = [(0.01, 5), (0.02, 5), (0.03, 5), (0.04, 5), (0.05, 5)]
    else:
        # Lower-triangular 2D curriculum
        _2D_STAGES = [
            (0.01,  1),
            (0.02,  1), (0.02,  2),
            (0.03,  1), (0.03,  2), (0.03,  3),
            (0.04,  1), (0.04,  2), (0.04,  3), (0.04,  4),
            (0.05,  1), (0.05,  2), (0.05,  3), (0.05,  4), (0.05,  5),
            (0.06,  1), (0.06,  2), (0.06,  3), (0.06,  4), (0.06,  5), (0.06,  6),
            (0.07,  1), (0.07,  2), (0.07,  3), (0.07,  4), (0.07,  5), (0.07,  6), (0.07,  7),
            (0.08,  1), (0.08,  2), (0.08,  3), (0.08,  4), (0.08,  5), (0.08,  6), (0.08,  7), (0.08,  8),
            (0.09,  1), (0.09,  2), (0.09,  3), (0.09,  4), (0.09,  5), (0.09,  6), (0.09,  7), (0.09,  8), (0.09,  9),
            (0.10,  1), (0.10,  2), (0.10,  3), (0.10,  4), (0.10,  5), (0.10,  6), (0.10,  7), (0.10,  8), (0.10,  9), (0.10, 10),
        ]
    _CURRICULUM_WINDOW    = 20
    _CURRICULUM_THRESHOLD = 0.02
    _stage_idx = 0
    curr_lift_target = _2D_STAGES[0][0]
    curr_min_steps   = _2D_STAGES[0][1]
    _success_window  = deque(maxlen=_CURRICULUM_WINDOW)

    for it in range(SAC_CFG["num_iterations"]):
        warmup = buf.size < SAC_CFG["min_replay_size"]
        explore = warmup and not policy_warmup
        params  = (training_state.normalizer_params, training_state.policy_params)

        # Reset all envs
        rng, *rng_resets = jax.random.split(rng, n_envs + 1)
        states = jit_vmap_reset(jnp.stack(rng_resets))
        # Inject current 2D curriculum targets into env states (no re-JIT — value change only)
        states = states.replace(info={**states.info,
            "lift_target":    jnp.full((n_envs,), curr_lift_target),
            "min_hold_steps": jnp.full((n_envs,), curr_min_steps)})

        # Collect via GPU-resident scan — one GPU→CPU transfer per iter
        _COMPONENTS = ["reach", "grasp", "lift", "success", "obj_touch",
                       "default_pos", "hold", "action_smooth", "jerk_smooth",
                       "diag_grasp_ok", "diag_height_factor", "diag_time_factor",
                       "diag_held_steps", "diag_palm_preshape", "diag_near_score",
                       "diag_force_score", "diag_r_grasp_prox", "diag_r_grasp_contact"]
        if explore:
            _, rng, traj = _collect_explore(states, rng)
        else:
            _, rng, traj = _collect_policy(states, rng, params)
        obs_T, nobs_T, act_T, rew_T, done_T, metrics_T = traj
        rew_arr    = np.array(rew_T)              # (ep_length, n_envs)
        ep_r_train = rew_arr.sum(axis=0)          # (n_envs,) cumulative reward per env
        ep_comp    = {c: float(np.array(metrics_T[c]).mean()) if c in metrics_T else 0.0
                      for c in _COMPONENTS}
        buf.add(
            np.array(obs_T).reshape(-1, obs_dim),
            np.array(nobs_T).reshape(-1, obs_dim),
            np.array(act_T).reshape(-1, ACTION_DIM),
            rew_arr.reshape(-1),
            np.array(done_T).reshape(-1),
        )
        total_transitions += n_envs * ep_length

        train_r = float(ep_r_train.mean())
        elapsed = time.monotonic() - t0
        tps     = total_transitions / elapsed if elapsed > 0 else 0

        comp_str = "  ".join(f"{c}={v:+.3f}" for c, v in ep_comp.items() if v != 0.0)
        print(f"  iter={it+1:>5,}  transitions={total_transitions:>10,}  "
              f"train_r={train_r:+.2f}  buf={buf.size:,}  "
              f"trans/s={tps:.0f}  elapsed={elapsed:.0f}s"
              + ("  [warmup]" if warmup else ""))
        if comp_str:
            print(f"    {comp_str}")

        # Always write train log every iter
        cur_alpha = float(jnp.exp(training_state.alpha_params))
        with open(train_log_path, "a") as f:
            json.dump({"iter": it + 1, "transitions": total_transitions,
                       "reward": train_r, "reward_std": float(ep_r_train.std()),
                       **{c: ep_comp[c] for c in _COMPONENTS},
                       "alpha": cur_alpha,
                       "elapsed": elapsed, "trans_per_sec": tps,
                       "warmup": bool(warmup),
                       "curriculum_stage": _stage_idx,
                       "lift_target": curr_lift_target,
                       "min_hold_steps": curr_min_steps}, f)
            f.write("\n")

        # 2D curriculum rolling-window check — runs every iter
        if not warmup:
            _success_window.append(ep_comp.get("success", 0.0))
            if (len(_success_window) == _CURRICULUM_WINDOW and
                    np.mean(_success_window) > _CURRICULUM_THRESHOLD and
                    _stage_idx < len(_2D_STAGES) - 1):
                rolling_mean = float(np.mean(_success_window))
                _stage_idx += 1
                curr_lift_target, curr_min_steps = _2D_STAGES[_stage_idx]
                _success_window.clear()
                # Reset alpha on graduation for stages 0-9 so policy re-explores harder task
                if not _NO_ALPHA_FIX and not _NO_STAGE_RESET and _stage_idx <= _ALPHA_RESET_UNTIL:
                    _reset_log_alpha = jnp.asarray(float(np.log(_ALPHA_STAGE_RESET)), dtype=jnp.float32)
                    training_state = training_state.replace(
                        alpha_params=_reset_log_alpha,
                        alpha_optimizer_state=_alpha_optimizer.init(_reset_log_alpha),
                    )
                print(f"  [curriculum] GRADUATED at iter={it+1}! "
                      f"rolling_success={rolling_mean:.3f}>{_CURRICULUM_THRESHOLD}  "
                      f"stage {_stage_idx}: lift={curr_lift_target*100:.0f}cm "
                      f"hold={curr_min_steps:.0f}steps"
                      + (f"  alpha→{_ALPHA_STAGE_RESET:.1f}" if _stage_idx <= _ALPHA_RESET_UNTIL else ""))

        # Alpha floor: prevent collapse during early stages (0-5)
        if not _NO_ALPHA_FIX and _stage_idx <= _ALPHA_RESET_UNTIL:
            _min_log_alpha = float(np.log(_ALPHA_FLOOR))
            if float(training_state.alpha_params) < _min_log_alpha:
                training_state = training_state.replace(
                    alpha_params=jnp.asarray(_min_log_alpha, dtype=jnp.float32)
                )

        if warmup:
            continue

        # Pre-sample all batches at once (vectorised numpy), one large host→device
        # transfer, then a single lax.scan JIT call for all grad_updates_per_iter steps.
        G = SAC_CFG["grad_updates_per_iter"]
        B = SAC_CFG["batch_size"]
        idx = np.random.randint(0, buf.size, (G, B))
        training_state, rng = update_fn(
            training_state,
            jnp.array(buf.obs[idx]),      # (G, B, obs_dim)
            jnp.array(buf.next_obs[idx]),
            jnp.array(buf.action[idx]),
            jnp.array(buf.reward[idx]),
            jnp.array(buf.done[idx]),
            rng,
        )

        if (it + 1) % 10 == 0:
            _save_params("latest")

        if (it + 1) % SAC_CFG["eval_every"] == 0:
            rng, *rng_evals = jax.random.split(rng, SAC_CFG["n_eval_episodes"] + 1)
            eval_states = jit_vmap_reset(jnp.stack(rng_evals))
            eval_states = eval_states.replace(info={**eval_states.info,
                "lift_target":    jnp.full((SAC_CFG["n_eval_episodes"],), curr_lift_target),
                "min_hold_steps": jnp.full((SAC_CFG["n_eval_episodes"],), curr_min_steps)})
            ep_r    = np.zeros(SAC_CFG["n_eval_episodes"])
            ep_comp = {c: np.zeros(SAC_CFG["n_eval_episodes"]) for c in _COMPONENTS}
            active  = np.ones(SAC_CFG["n_eval_episodes"], bool)
            for _ in range(ep_length):
                rng, rng_act = jax.random.split(rng)
                rng_acts = jax.random.split(rng_act, SAC_CFG["n_eval_episodes"])
                actions, _ = jax.vmap(lambda o, r: inference_fn(params)(o, r))(
                    eval_states.obs, rng_acts)
                eval_states = jit_vmap_step(eval_states, actions)
                ep_r   += np.array(eval_states.reward) * active
                for c in _COMPONENTS:
                    if c in eval_states.metrics:
                        ep_comp[c] += np.array(eval_states.metrics[c]) * active
                active &= ~(np.array(eval_states.done) > 0.5)
                if not active.any():
                    break
            mean_r, std_r = float(ep_r.mean()), float(ep_r.std())
            comp = {c: float(ep_comp[c].mean()) for c in _COMPONENTS}
            alpha      = float(jnp.exp(training_state.alpha_params))
            grad_steps = int(training_state.gradient_steps)
            comp_str   = "  ".join(f"{c}={v:+.3f}" for c, v in comp.items() if v != 0.0)
            print(f"  iter={it+1:>5,}  transitions={total_transitions:>10,}  "
                  f"reward={mean_r:+.2f} ± {std_r:.2f}  "
                  f"buf={buf.size:,}  grad_steps={grad_steps:,}  "
                  f"alpha={alpha:.4f}  trans/s={tps:.0f}  elapsed={elapsed:.0f}s")
            print(f"    {comp_str}")
            with open(log_path, "a") as f:
                json.dump({"iter": it+1, "transitions": total_transitions,
                           "reward": mean_r, "reward_std": std_r,
                           **comp,
                           "alpha": alpha, "grad_steps": grad_steps,
                           "elapsed": elapsed, "trans_per_sec": tps,
                           "lift_target": curr_lift_target,
                           "min_hold_steps": curr_min_steps,
                           "curriculum_stage": _stage_idx}, f)
                f.write("\n")

            if mean_r > best_eval:
                best_eval = mean_r
                _save_params("best")

            if (it + 1) % SAC_CFG["video_every"] == 0:
                _save_video(f"iter_{it+1:06d}")

    print(f"\nDone. Best eval reward: {best_eval:+.2f}")
    _save_params("final")
    _save_video("final")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",           type=str,   default="")
    parser.add_argument("--lift-scale",     type=float, default=None)
    parser.add_argument("--hold-scale",     type=float, default=None)
    parser.add_argument("--success-scale",  type=float, default=None)
    parser.add_argument("--reach-scale",    type=float, default=None)
    parser.add_argument("--obj-touch-scale", type=float, default=None)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--curriculum-1d",  action="store_true",
                        help="Use 1D height-only curriculum (v4/v5 style: 1→2→3→5cm, hold=5)")
    parser.add_argument("--no-alpha-fix",   action="store_true",
                        help="Disable alpha floor and stage reset (pure v4/v5 style)")
    parser.add_argument("--no-stage-reset", action="store_true",
                        help="Disable alpha reset on curriculum graduation (keep floor)")
    parser.add_argument("--alpha-reset-until", type=int, default=None,
                        help="Apply alpha floor/reset until this stage (inclusive). Default: 9 for 2D, 3 for 1D")
    args = parser.parse_args()
    if args.lift_scale     is not None: SAC_CFG["lift_scale"]     = args.lift_scale
    if args.hold_scale     is not None: SAC_CFG["hold_scale"]     = args.hold_scale
    if args.success_scale  is not None: SAC_CFG["success_scale"]  = args.success_scale
    if args.reach_scale    is not None: SAC_CFG["reach_scale"]    = args.reach_scale
    if args.obj_touch_scale is not None: SAC_CFG["obj_touch_scale"] = args.obj_touch_scale
    if args.target_entropy is not None: SAC_CFG["target_entropy"] = args.target_entropy
    SAC_CFG["curriculum_1d"] = args.curriculum_1d
    SAC_CFG["no_alpha_fix"]    = args.no_alpha_fix
    SAC_CFG["no_stage_reset"]  = args.no_stage_reset
    if args.alpha_reset_until is not None:
        SAC_CFG["alpha_reset_until"] = args.alpha_reset_until
    main(name=args.name)
