"""VegaPickYCBMulti: pick-and-lift task with per-episode YCB object randomization.

Key differences from VegaPickYCB:
  - All YCB objects are embedded in one compiled MJX model at init time.
  - Each episode randomly selects one object; the rest are parked off-table.
  - Observation appends object half-extents (3D geometry features) so the
    policy can adapt finger spread to the active object's shape.
  - Table-height domain randomization: spawn Z is jittered ±2 cm each reset,
    training the policy to grasp at varying heights.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State

from envs.vega_base import VegaBase
from envs.ycb_objects import REGISTRY, build_multi_object_scene_xml, spawn_height

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENE_XML = os.path.join(BASE_DIR, "xmls", "vega_right_arm_scene.xml")

# Objects used for training — potted_meat_can first (closest to the cube the
# policy was originally tuned for).
TRAIN_OBJECTS: List[str] = [
    "potted_meat_can",
    "banana",
    "mug",
    "foam_brick",
]
_N_OBJECTS = len(TRAIN_OBJECTS)

# Object randomization bounds (relative to default object centre on table).
_OBJ_X_RANGE = (-0.04, 0.04)
_OBJ_Y_RANGE = (-0.06, 0.06)
# Table-height domain randomization: shifts the spawn Z each episode.
_TABLE_DZ_RANGE = (0.0, 0.04)   # always above table; negative was spawning objects inside table → NaN

_LIFT_TARGET_HEIGHT = 0.05   # 5 cm above table surface = task success
_PALM_GRASP_THRESH  = 0.10
_TIP_CONTACT_THRESH = 0.08
_OBJ_FALL_Z         = -0.05
_HOLD_GATE          = 0.01

# Arm centre used for reset with optional noise.
_GRASP_READY_ARM_Q = np.array([-1.107, -0.414, -1.071, -1.222, 0., 0., 0.])
_ARM_NOISE         = np.array([ 0.15,   0.02,   0.0,    0.02,  0.05, 0.05, 0.05])


def default_config() -> config_dict.ConfigDict:
    """Default environment configuration."""
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.005,
        episode_length=250,
        action_repeat=1,
        action_scale=0.05,
        action_mode="18d",
        impl="jax",
        naconmax=128,
        naccdmax=128,
        njmax=128,
        fixed_arm_init=True,
        reward_config=config_dict.create(
            scales=config_dict.create(
                reach=0.05,
                obj_touch=2.0,
                grasp=1.2,
                lift=3.0,
                hold=1.5,
                success=5.0,
                default_pos=0.0,
                progress=0.0,
                action_smooth=0.0,
                jerk_smooth=0.0,
                fall_penalty=0.0,
            )
        ),
    )


class VegaPickYCBMulti(VegaBase):
    """Pick-and-lift with per-episode random YCB object and domain randomization.

    Observation space: 104-D
      arm_qpos (7) + arm_qvel (7) + hand_qpos (11) + hand_qvel (11)
      + palm_pose (9) + object_state (15) + palm_to_obj (3)
      + fingertip_to_obj (15) + touch (5) + obj_half_extents (3)
      + prev_action (18)
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, List[Any]]]] = None,
    ):
        # Build scene XML with all YCB objects embedded; write temp file so
        # that the robot <include> and absolute mesh paths resolve correctly.
        with open(_SCENE_XML) as f:
            scene_xml = f.read()
        multi_xml = build_multi_object_scene_xml(scene_xml, TRAIN_OBJECTS,
                                                   include_visual_meshes=False)
        xml_dir = os.path.dirname(_SCENE_XML)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", dir=xml_dir, delete=False)
        try:
            tmp.write(multi_xml)
            tmp.close()
            super().__init__(tmp.name, config, config_overrides)
        finally:
            os.unlink(tmp.name)

        self._post_init(obj_name="object_0", keyframe="grasp_ready")

        # Per-object body / qpos / dof addresses and spawn heights.
        obj_body_ids, obj_qposadr, obj_dofadr, spawn_heights, half_extents = (
            [], [], [], [], [])
        for idx, name in enumerate(TRAIN_OBJECTS):
            body = self._mj_model.body(f"object_{idx}")
            jnt_id = body.jntadr[0]
            obj_body_ids.append(body.id)
            obj_qposadr.append(self._mj_model.jnt_qposadr[jnt_id])
            obj_dofadr.append(self._mj_model.jnt_dofadr[jnt_id])
            spawn_heights.append(spawn_height(name, self._table_surface_z))
            obj_reg = REGISTRY[name]
            half_extents.append(list(obj_reg.half_extents))

        # Store as both NumPy (used in reset indexing) and JAX (used in JIT).
        self._obj_body_ids  = np.array(obj_body_ids,  np.int32)
        self._obj_qposadr   = np.array(obj_qposadr,   np.int32)
        self._obj_dofadr    = np.array(obj_dofadr,    np.int32)

        self._obj_body_ids_j  = jp.array(self._obj_body_ids)
        self._obj_dofadr_j    = jp.array(self._obj_dofadr,  jp.int32)
        self._spawn_heights_j = jp.array(spawn_heights,     jp.float32)
        self._half_extents_j  = jp.array(half_extents,      jp.float32)  # (N,3)

        # Override parent's single-object references to point to object_0.
        self._obj_body    = self._obj_body_ids[0]
        self._obj_qposadr_single = self._obj_qposadr[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_obj_pos(self, data: mjx.Data, active_obj: jp.ndarray) -> jp.ndarray:
        """Return xpos of the active object body (3,)."""
        body_id = self._obj_body_ids_j[active_obj]
        return data.xpos[body_id]

    def _active_obj_state(
        self, data: mjx.Data, active_obj: jp.ndarray
    ) -> jp.ndarray:
        """Return 15-D object state (pos3 + rot6 + linvel3 + angvel3)."""
        body_id = self._obj_body_ids_j[active_obj]
        dofadr  = self._obj_dofadr_j[active_obj]
        obj_pos  = data.xpos[body_id]
        obj_rot6 = data.xmat[body_id].ravel()[:6]
        obj_vel  = jax.lax.dynamic_slice(data.qvel, (dofadr,), (6,))
        return jp.concatenate([obj_pos, obj_rot6, obj_vel[:3], obj_vel[3:]])

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self, data: mjx.Data, info: dict) -> jp.ndarray:
        active_obj = info["active_obj"]
        obj_pos   = self._active_obj_pos(data, active_obj)
        obj_state = self._active_obj_state(data, active_obj)

        palm_pos  = data.site_xpos[self._palm_site]
        palm_rot6 = data.site_xmat[self._palm_site].ravel()[:6]

        fingertip_to_obj = jp.concatenate([
            obj_pos - data.site_xpos[sid] for sid in self._fingertip_sites
        ])
        touch      = data.sensordata[self._touch_sensor_adr].ravel()
        geom_feats = self._half_extents_j[active_obj]           # (3,) — shape info

        return jp.concatenate([
            data.qpos[self._arm_qposadr],       # 7
            data.qvel[self._arm_dofadr],        # 7
            data.qpos[self._hand_qposadr],      # 11
            data.qvel[self._hand_dofadr],       # 11
            palm_pos, palm_rot6,                # 9
            obj_state,                          # 15
            obj_pos - palm_pos,                 # 3  (palm→obj direction)
            fingertip_to_obj,                   # 15
            touch,                              # 5
            geom_feats,                         # 3  (NEW: object half-extents)
            info["prev_action"],                # 18
        ])  # total: 104

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _get_reward(
        self, data: mjx.Data, action: jp.ndarray, info: dict
    ) -> Dict[str, jp.ndarray]:
        active_obj = info["active_obj"]
        obj_pos    = self._active_obj_pos(data, active_obj)
        palm_pos   = data.site_xpos[self._palm_site]
        obj_z      = obj_pos[2]

        rest_offset = (self._spawn_heights_j[active_obj]
                       - self._table_surface_z)
        obj_height  = obj_z - self._table_surface_z - rest_offset

        palm_to_obj_dist = jp.linalg.norm(palm_pos - obj_pos)
        r_reach = 1.0 - jp.tanh(2.0 * palm_to_obj_dist)

        touch    = data.sensordata[self._touch_sensor_adr].ravel()
        tip_dists = jp.stack([
            jp.linalg.norm(data.site_xpos[sid] - obj_pos)
            for sid in self._fingertip_sites
        ])
        near_obj = (tip_dists < _TIP_CONTACT_THRESH).astype(jp.float32)
        r_obj_touch  = jp.mean(1.0 - jp.tanh(10.0 * tip_dists))
        palm_preshape   = (palm_to_obj_dist < _PALM_GRASP_THRESH).astype(jp.float32)
        finger_contact  = near_obj * (touch > 0.01).astype(jp.float32)
        r_grasp_prox    = palm_preshape * jp.clip(jp.sum(near_obj) / 3., 0., 0.5)
        r_grasp_contact = palm_preshape * jp.clip(jp.sum(finger_contact) / 2., 0., 0.5)
        r_grasp = r_grasp_prox + r_grasp_contact

        grasp_ok = (r_grasp_contact > 0.25).astype(jp.float32)
        r_lift   = grasp_ok * jp.clip(obj_height / _LIFT_TARGET_HEIGHT, 0., 1.)

        grasp_active = (r_grasp > 0.3).astype(jp.float32)
        r_hold = grasp_active * jp.clip(obj_height / _HOLD_GATE, 0., 1.)

        held_high = ((obj_height > _LIFT_TARGET_HEIGHT) & (r_grasp > 0.3)).astype(jp.float32)
        info["held_steps"] = info["held_steps"] + held_high
        r_success = (info["held_steps"] >= 5).astype(jp.float32)

        info["prev_action"] = action

        scales = self._config.reward_config.scales
        return {
            "reach":         scales.reach        * r_reach,
            "obj_touch":     scales.obj_touch     * r_obj_touch,
            "grasp":         scales.grasp         * r_grasp,
            "lift":          scales.lift          * r_lift,
            "hold":          scales.hold          * r_hold,
            "success":       scales.success       * r_success,
            "default_pos":   jp.zeros(()),
            "progress":      jp.zeros(()),
            "action_smooth": jp.zeros(()),
            "jerk_smooth":   jp.zeros(()),
            "fall_penalty":  jp.zeros(()),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng_obj, rng_x, rng_y, rng_yaw, rng_arm, rng_th = (
            jax.random.split(rng, 7))

        active_obj = jax.random.randint(rng_obj, (), 0, _N_OBJECTS)
        table_dz   = jax.random.uniform(
            rng_th, minval=_TABLE_DZ_RANGE[0], maxval=_TABLE_DZ_RANGE[1])
        dx = jax.random.uniform(rng_x, minval=_OBJ_X_RANGE[0], maxval=_OBJ_X_RANGE[1])
        dy = jax.random.uniform(rng_y, minval=_OBJ_Y_RANGE[0], maxval=_OBJ_Y_RANGE[1])
        yaw = jax.random.uniform(rng_yaw, minval=-jp.pi, maxval=jp.pi)
        obj_quat = jp.array([jp.cos(yaw / 2), 0., 0., jp.sin(yaw / 2)])

        if self._config.fixed_arm_init:
            arm_q = jp.array(_GRASP_READY_ARM_Q)
        else:
            noise = jax.random.uniform(rng_arm, (7,), minval=-1., maxval=1.)
            arm_q = jp.array(_GRASP_READY_ARM_Q) + noise * jp.array(_ARM_NOISE)
        arm_q = jp.clip(arm_q, self._arm_lowers, self._arm_uppers)

        init_q = jp.array(self._init_q)
        init_q = init_q.at[self._arm_qposadr].set(arm_q)

        for i in range(_N_OBJECTS):
            spawn_z = self._spawn_heights_j[i] + table_dz
            on_pos  = jp.array([0.55 + dx, dy, spawn_z])
            # Park on the floor (z=0 plane) at correct resting height — no free-fall impact
            floor_rest_z = self._spawn_heights_j[i] - self._table_surface_z
            off_pos = jp.array([20.0 + i * 5.0, 0.0, floor_rest_z])
            is_active = jp.array(i) == active_obj
            pos  = jp.where(is_active, on_pos,  off_pos)
            quat = jp.where(is_active, obj_quat, jp.array([1., 0., 0., 0.]))
            qa = int(self._obj_qposadr[i])
            init_q = init_q.at[qa: qa + 3].set(pos)
            init_q = init_q.at[qa + 3: qa + 7].set(quat)

        init_ctrl = jp.array(self._init_ctrl).at[:7].set(arm_q)
        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, jp.float32),
            ctrl=init_ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )

        metrics = {k: jp.zeros(()) for k in self._config.reward_config.scales}
        info = {
            "rng":           rng,
            "active_obj":    active_obj,
            "held_steps":    jp.zeros(()),
            "step_count":    jp.zeros(()),
            "prev_action":   jp.zeros(self.action_size),
            "prev2_action":  jp.zeros(self.action_size),
            "prev_palm_dist": jp.array(1.0),
        }
        obs = self._get_obs(data, info)
        return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, state: State, action: jp.ndarray) -> State:
        data = self._apply_action(state.data, action)
        data = mjx_env.step(self._mjx_model, data, data.ctrl, self.n_substeps)

        raw_rewards = self._get_reward(data, action, state.info)

        active_obj = state.info["active_obj"]
        obj_z  = self._active_obj_pos(data, active_obj)[2]
        fallen = obj_z < (self._table_surface_z + _OBJ_FALL_Z)
        reward = jp.clip(sum(raw_rewards.values()), -1e4, 1e4)

        step_count = state.info["step_count"] + 1
        done = (fallen | (step_count >= self._config.episode_length)
                | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
                ).astype(jp.float32)

        state.metrics.update(**{k: v for k, v in raw_rewards.items()})
        new_info = {**state.info, "step_count": step_count}
        obs = self._get_obs(data, new_info)
        return State(data, obs, reward, done, state.metrics, new_info)
