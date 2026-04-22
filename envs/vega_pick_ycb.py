"""VegaPickYCB: pick-and-lift task for Vega dexterous hand."""

import os
import tempfile
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State
import numpy as np

from envs.vega_base import VegaBase
from envs.ycb_objects import inject_into_scene, spawn_height

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENE_XML = os.path.join(BASE_DIR, "xmls/vega_right_arm_scene.xml")

# Object randomization bounds on the table (relative to default object position)
_OBJ_X_RANGE = (-0.04, 0.04)
_OBJ_Y_RANGE = (-0.06, 0.06)

# Grasp-ready arm joint targets for default-pos reward
_GRASP_READY_ARM_Q = np.array([-1.107, -0.414, -1.071, -1.222, 0.0, 0.0, 0.0])

# Reward thresholds
_LIFT_TARGET_HEIGHT = 0.01   # 1 cm above table = success (lowered from 5cm — robot lifts 0.5-1mm, 1cm reachable)
_PALM_GRASP_THRESH  = 0.10   # palm within 10 cm gates grasp reward
_TIP_CONTACT_THRESH = 0.08   # fingertip within 8 cm of object = near object
_OBJ_FALL_Z = -0.05          # object z below table_z - this = fallen
_HOLD_GATE  = 0.01           # 1 cm: r_hold fires here, provides gradient before 5 cm target


def default_config() -> config_dict.ConfigDict:
    cfg = config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.005,
        episode_length=250,
        action_repeat=1,
        action_scale=0.05,
        action_mode='18d',
        impl='jax',
        naconmax=128,
        naccdmax=128,
        njmax=128,
        fixed_arm_init=False,
        ycb_object=None,  # None = placeholder cube; str = YCB name, e.g. "potted_meat_can"
        lift_target_height=0.05,  # height (m) for success; curriculum: start 0.01 → graduate to 0.05
        min_hold_steps=5.0,       # consecutive steps object must be held above target; curriculum: start 1 → graduate to 5
        reward_config=config_dict.create(
            scales=config_dict.create(
                reach=0.4,
                default_pos=0.0,
                progress=0.0,
                obj_touch=0.5,
                grasp=1.2,
                lift=1.2,
                hold=1.5,
                success=5.0,
                action_smooth=0.0,
                jerk_smooth=0.0,
                fall_penalty=0.0,
            )
        ),
    )
    return cfg


class VegaPickYCB(VegaBase):
    """Pick-and-lift task: Vega right hand grasps a cube on a table."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        ycb_object = getattr(config, 'ycb_object', None)
        if ycb_object:
            # Inject YCB body into scene XML and write a temp file so that
            # the robot include and mesh paths all resolve correctly.
            with open(_SCENE_XML) as f:
                scene_xml = f.read()
            modified_xml = inject_into_scene(scene_xml, ycb_object)
            xml_dir = os.path.dirname(_SCENE_XML)
            tmp = tempfile.NamedTemporaryFile(
                mode='w', suffix='.xml', dir=xml_dir, delete=False)
            try:
                tmp.write(modified_xml)
                tmp.close()
                super().__init__(tmp.name, config, config_overrides)
            finally:
                os.unlink(tmp.name)
        else:
            super().__init__(_SCENE_XML, config, config_overrides)

        self._post_init(obj_name="object", keyframe="grasp_ready")

        # Fix spawn height: keyframe has cube's Z=0.545; override for real objects.
        if ycb_object:
            correct_z = float(spawn_height(ycb_object, self._table_surface_z))
            self._init_obj_pos = jp.array([
                float(self._init_obj_pos[0]),
                float(self._init_obj_pos[1]),
                correct_z,
            ])
            self._obj_rest_offset = correct_z - self._table_surface_z
            print(f"YCB object: {ycb_object}  spawn_z={correct_z:.4f}  "
                  f"rest_offset={self._obj_rest_offset:.4f}")

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_arm_qpos(self, data: mjx.Data) -> jax.Array:
        return data.qpos[self._arm_qposadr]  # (7,)

    def _get_arm_qvel(self, data: mjx.Data) -> jax.Array:
        return data.qvel[self._arm_dofadr]  # (7,)

    def _get_hand_qpos(self, data: mjx.Data) -> jax.Array:
        return data.qpos[self._hand_qposadr]  # (11,)

    def _get_hand_qvel(self, data: mjx.Data) -> jax.Array:
        return data.qvel[self._hand_dofadr]  # (11,)

    def _get_palm_pose(self, data: mjx.Data) -> jax.Array:
        palm_pos = data.site_xpos[self._palm_site]
        palm_rot6 = data.site_xmat[self._palm_site].ravel()[:6]
        return jp.concatenate([palm_pos, palm_rot6])  # (9,)

    def _get_object_state(self, data: mjx.Data) -> jax.Array:
        obj_pos = data.xpos[self._obj_body]
        obj_rot6 = data.xmat[self._obj_body].ravel()[:6]
        obj_vel = data.qvel[self._obj_dofadr:self._obj_dofadr + 6]
        linvel = obj_vel[0:3]
        angvel = obj_vel[3:6]
        return jp.concatenate([obj_pos, obj_rot6, linvel, angvel])  # (15,)

    def _get_palm_to_object(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._obj_body] - data.site_xpos[self._palm_site]  # (3,)

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        obj_pos = data.xpos[self._obj_body]
        fingertip_to_obj = jp.concatenate([
            obj_pos - data.site_xpos[sid]
            for sid in self._fingertip_sites
        ])
        touch = data.sensordata[self._touch_sensor_adr].ravel()
        obs = jp.concatenate([
            self._get_arm_qpos(data),          # (7,)
            self._get_arm_qvel(data),          # (7,)
            self._get_hand_qpos(data),         # (11,)
            self._get_hand_qvel(data),         # (11,)
            self._get_palm_pose(data),         # (9,)
            self._get_object_state(data),      # (15,)
            self._get_palm_to_object(data),    # (3,)
            fingertip_to_obj,                  # (15,)
            touch,                             # (5,)
            info["prev_action"],               # (18,)
        ])
        return obs  # total: 101

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _get_reward(
        self, data: mjx.Data, action: jax.Array, info: dict
    ) -> Dict[str, jax.Array]:
        obj_pos = data.xpos[self._obj_body]
        palm_pos = data.site_xpos[self._palm_site]
        obj_z = obj_pos[2]
        table_z = self._table_surface_z
        obj_height = obj_z - table_z - self._obj_rest_offset

        palm_to_obj_dist = jp.linalg.norm(palm_pos - obj_pos)

        # 1) Reach: tanh(2*d) gives 300x more gradient signal at 30cm vs tanh(8*d)
        r_reach = 1.0 - jp.tanh(2.0 * palm_to_obj_dist)

        # 2) Default-pos placeholder (disabled)
        r_default_pos = jp.zeros(())

        # 3) Progress placeholder (disabled)
        r_progress = jp.zeros(())
        info["prev_palm_dist"] = palm_to_obj_dist

        # 4) Grasp: palm proximity gate + multi-finger contact (no thumb requirement)
        touch = data.sensordata[self._touch_sensor_adr].ravel()
        tip_dists = jp.stack([
            jp.linalg.norm(data.site_xpos[sid] - obj_pos)
            for sid in self._fingertip_sites
        ])
        near_obj = (tip_dists < _TIP_CONTACT_THRESH).astype(jp.float32)
        r_obj_touch = jp.mean(1.0 - jp.tanh(10.0 * tip_dists))
        palm_preshape = (palm_to_obj_dist < _PALM_GRASP_THRESH).astype(jp.float32)
        has_force = (touch > 0.01).astype(jp.float32)
        finger_contact = near_obj * has_force
        n_contact = jp.sum(finger_contact)
        n_near = jp.sum(near_obj)
        # proximity component: gradient toward correct finger placement
        r_grasp_prox = palm_preshape * jp.clip(n_near / 3.0, 0.0, 0.5)
        # contact component: reward actual touch-sensor contact
        r_grasp_contact = palm_preshape * jp.clip(n_contact / 2.0, 0.0, 0.5)
        r_grasp = r_grasp_prox + r_grasp_contact

        # 5) Lift: gated on actual contact (not proximity)
        grasp_ok = (r_grasp_contact > 0.25).astype(jp.float32)
        lift_target = info["lift_target"]
        r_lift = grasp_ok * jp.clip(obj_height / lift_target, 0.0, 1.0)

        # 6+7) Hold / Success
        # r_hold: proportional height ramp 0→1 over 0→1cm, then capped; provides dense gradient
        grasp_active = (r_grasp > 0.3).astype(jp.float32)
        r_hold = grasp_active * jp.clip(obj_height / _HOLD_GATE, 0.0, 1.0)
        held_high = ((obj_height > lift_target) & (r_grasp > 0.3)).astype(jp.float32)
        # Consecutive counter: resets to 0 the moment either condition fails
        info["held_steps"] = (info["held_steps"] + 1.0) * held_high
        min_hold = info.get("min_hold_steps", jp.array(5.0))
        r_success = (info["held_steps"] >= min_hold).astype(jp.float32)

        # 8) Smoothness placeholders (disabled)
        r_action = jp.zeros(())
        r_jerk = jp.zeros(())
        info["prev2_action"] = info["prev_action"]
        info["prev_action"] = action

        scales = self._config.reward_config.scales
        rewards = {
            "reach":        scales.reach * r_reach,
            "default_pos":  scales.default_pos * r_default_pos,
            "progress":     scales.progress * r_progress,
            "obj_touch":    scales.obj_touch * r_obj_touch,
            "grasp":        scales.grasp * r_grasp,
            "lift":         scales.lift * r_lift,
            "hold":         scales.hold * r_hold,
            "success":      scales.success * r_success,
            "action_smooth": scales.action_smooth * r_action,
            "jerk_smooth":   scales.jerk_smooth * r_jerk,
        }
        return rewards

    # ------------------------------------------------------------------
    # Reset / Step / Termination
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        rng, rng_x, rng_y, rng_yaw, rng_arm = jax.random.split(rng, 5)

        dx = jax.random.uniform(rng_x, minval=_OBJ_X_RANGE[0], maxval=_OBJ_X_RANGE[1])
        dy = jax.random.uniform(rng_y, minval=_OBJ_Y_RANGE[0], maxval=_OBJ_Y_RANGE[1])
        obj_pos = self._init_obj_pos + jp.array([dx, dy, 0.0])

        yaw = jax.random.uniform(rng_yaw, minval=-jp.pi, maxval=jp.pi)
        half_yaw = yaw / 2
        obj_quat = jp.array([jp.cos(half_yaw), 0.0, 0.0, jp.sin(half_yaw)])

        # Randomize arm posture: perturb each joint within safe range above table
        # Safe zone: j1∈[-1.3,-0.7], j2=0.3, j4=-0.8 keeps palm above table
        _ARM_CENTER = jp.array([-1.0,   0.3,  -1.071, -0.8,  0.0, 0.0, 0.0])
        _ARM_NOISE  = jp.array([ 0.15,  0.02,  0.0,    0.02, 0.05, 0.05, 0.05])
        if self._config.fixed_arm_init:
            arm_q = _ARM_CENTER
        else:
            arm_noise = jax.random.uniform(rng_arm, shape=(7,), minval=-1.0, maxval=1.0) * _ARM_NOISE
            arm_q = _ARM_CENTER + arm_noise
        arm_q = jp.clip(arm_q, self._arm_lowers, self._arm_uppers)

        init_q = jp.array(self._init_q)
        init_q = init_q.at[self._arm_qposadr].set(arm_q)
        init_q = init_q.at[self._obj_qposadr:self._obj_qposadr + 3].set(obj_pos)
        init_q = init_q.at[self._obj_qposadr + 3:self._obj_qposadr + 7].set(obj_quat)

        init_ctrl = jp.array(self._init_ctrl).at[:7].set(arm_q)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=jp.float32),
            ctrl=init_ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )

        metrics = {k: jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        info = {
            "rng": rng,
            "held_steps":      jp.zeros(()),
            "step_count":      jp.zeros(()),
            "prev_palm_dist":  jp.array(1.0),
            "prev_action":     jp.zeros(self.action_size),
            "prev2_action":    jp.zeros(self.action_size),
            "lift_target":     jp.full((), getattr(self._config, 'lift_target_height', _LIFT_TARGET_HEIGHT)),
            "min_hold_steps":  jp.full((), getattr(self._config, 'min_hold_steps', 5.0)),
        }

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        data = self._apply_action(state.data, action)
        data = mjx_env.step(self._mjx_model, data, data.ctrl, self.n_substeps)

        raw_rewards = self._get_reward(data, action, state.info)

        obj_z = data.xpos[self._obj_body][2]
        fallen = obj_z < (self._table_surface_z + _OBJ_FALL_Z)
        fall_penalty = fallen.astype(jp.float32) * self._config.reward_config.scales.fall_penalty
        reward = jp.clip(sum(raw_rewards.values()) + fall_penalty, -1e4, 1e4)

        step_count = state.info["step_count"] + 1
        timeout = step_count >= self._config.episode_length
        nan_detected = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = (fallen | timeout | nan_detected).astype(jp.float32)

        state.metrics.update(**{k: v for k, v in raw_rewards.items()})
        new_info = {**state.info, "step_count": step_count}

        obs = self._get_obs(data, new_info)
        return State(data, obs, reward, done, state.metrics, new_info)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def register():
    from mujoco_playground._src.manipulation import register_environment
    register_environment("VegaPickYCB", VegaPickYCB, default_config)
