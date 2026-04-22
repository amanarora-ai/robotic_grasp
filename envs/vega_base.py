"""VegaBase: base environment class for Vega right arm + dexterous hand."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

# ctrl layout: [0-6] arm j1-j7, [7-17] hand (th_j0,th_j1,th_j2, ff_j1,ff_j2, mf_j1,mf_j2, rf_j1,rf_j2, lf_j1,lf_j2)

# 10D synergy indices within hand ctrl (0-indexed from hand start)
_THUMB_ABD_IDX  = 0
_OUTER_FLEX_IDX = [7, 8, 9, 10]  # rf+lf flexion: used by finger_spread


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.005,
        episode_length=250,
        action_repeat=1,
        action_scale=0.05,
        action_mode='18d',   # '18d' = per-joint deltas | '10d' = 7 arm + 3 synergies
        impl='jax',
        naconmax=128,
        naccdmax=128,
        njmax=128,
    )


class VegaBase(mjx_env.MjxEnv):
    """Base class for Vega right arm + dexterous hand environments."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        self._xml_path = xml_path
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_model.opt.timestep = self.sim_dt

        self._mj_model = mj_model
        print(f"nbody: {mj_model.nbody}")
        print(f"ngeom: {mj_model.ngeom}")
        print(f"nmesh: {mj_model.nmesh}")
        print(f"total mesh vertices: {sum(mj_model.mesh_vertnum)}")
        self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
        self._action_scale = config.action_scale
        self._action_mode  = config.get('action_mode', '18d')

    def _post_init(self, obj_name: str, keyframe: str):
        """Initialize index arrays and constants from the loaded model."""
        _arm_joint_names = [f"R_arm_j{i}" for i in range(1, 8)]
        _hand_joint_names = [
            "R_th_j0", "R_th_j1", "R_th_j2",
            "R_ff_j1", "R_ff_j2",
            "R_mf_j1", "R_mf_j2",
            "R_rf_j1", "R_rf_j2",
            "R_lf_j1", "R_lf_j2",
        ]

        # qpos addresses for arm and hand joints
        self._arm_qposadr = np.array([
            self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
            for j in _arm_joint_names
        ])
        self._hand_qposadr = np.array([
            self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
            for j in _hand_joint_names
        ])

        # dof (velocity) addresses — for revolute joints these equal qposadr
        self._arm_dofadr = np.array([
            self._mj_model.jnt_dofadr[self._mj_model.joint(j).id]
            for j in _arm_joint_names
        ])
        self._hand_dofadr = np.array([
            self._mj_model.jnt_dofadr[self._mj_model.joint(j).id]
            for j in _hand_joint_names
        ])

        # Ctrl limits
        self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T
        self._arm_lowers = jp.array(self._lowers[:7])
        self._arm_uppers = jp.array(self._uppers[:7])

        hand_lo = jp.array(self._lowers[7:])
        hand_hi = jp.array(self._uppers[7:])
        self._hand_lo     = hand_lo
        self._hand_hi     = hand_hi
        self._hand_center = (hand_lo + hand_hi) / 2
        self._hand_half   = (hand_hi - hand_lo) / 2

        # Sites
        self._palm_site = self._mj_model.site("right_palm_site").id
        self._fingertip_sites = np.array([
            self._mj_model.site(s).id
            for s in ["right_thumb_tip", "right_index_tip", "right_middle_tip",
                      "right_ring_tip", "right_little_tip"]
        ])

        # Object body and qpos/dof addresses
        self._obj_body = self._mj_model.body(obj_name).id
        obj_jnt_id = self._mj_model.body(obj_name).jntadr[0]
        self._obj_qposadr = self._mj_model.jnt_qposadr[obj_jnt_id]
        self._obj_dofadr  = self._mj_model.jnt_dofadr[obj_jnt_id]

        # Table surface z (body pos z + geom half-size z)
        table_body_id     = self._mj_model.body("table").id
        table_geom_id     = self._mj_model.geom("table_top").id
        table_body_z      = self._mj_model.body_pos[table_body_id][2]
        table_geom_z      = self._mj_model.geom_pos[table_geom_id][2]
        table_geom_half_z = self._mj_model.geom_size[table_geom_id][2]
        self._table_surface_z = float(table_body_z + table_geom_z + table_geom_half_z)

        obj_rest_z = float(self._mj_model.keyframe(keyframe).qpos[
            self._mj_model.jnt_qposadr[obj_jnt_id] + 2
        ])
        self._obj_rest_offset = obj_rest_z - self._table_surface_z

        # Keyframe initial state
        self._init_q    = np.array(self._mj_model.keyframe(keyframe).qpos)
        self._init_ctrl = np.array(self._mj_model.keyframe(keyframe).ctrl)
        self._init_obj_pos = jp.array(
            self._init_q[self._obj_qposadr:self._obj_qposadr + 3]
        )

        # Touch sensor sensordata addresses
        self._touch_sensor_adr = np.array([
            self._mj_model.sensor(s).adr
            for s in ["thumb_touch", "index_touch", "middle_touch",
                      "ring_touch", "little_touch"]
        ])

        print(f"action_mode: {self._action_mode}  action_size: {self.action_size}")

    def _finger_coupling(
        self,
        grasp_amount: jax.Array,
        thumb_opposition: jax.Array,
        finger_spread: jax.Array,
    ) -> jax.Array:
        """Map 3 scalars in [-1,1] to 11 hand actuator targets (10D synergy mode)."""
        targets = self._hand_center + grasp_amount * self._hand_half
        targets = targets.at[_THUMB_ABD_IDX].set(
            self._hand_center[_THUMB_ABD_IDX]
            + thumb_opposition * self._hand_half[_THUMB_ABD_IDX]
        )
        for idx in _OUTER_FLEX_IDX:
            targets = targets.at[idx].add(-finger_spread * self._hand_half[idx])
        return jp.clip(targets, self._hand_lo, self._hand_hi)

    def _apply_action(self, data: mjx.Data, action: jax.Array) -> mjx.Data:
        arm_ctrl = jp.clip(
            data.ctrl[:7] + action[:7] * self._action_scale,
            self._arm_lowers,
            self._arm_uppers,
        )
        if self._action_mode == '7d':
            hand_ctrl = data.ctrl[7:]   # hand frozen at current ctrl
        elif self._action_mode == '10d':
            hand_ctrl = self._finger_coupling(action[7], action[8], action[9])
        else:
            hand_ctrl = jp.clip(
                data.ctrl[7:] + action[7:] * self._action_scale,
                self._hand_lo,
                self._hand_hi,
            )
        return data.replace(ctrl=jp.concatenate([arm_ctrl, hand_ctrl]))

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        if self._action_mode == '7d':
            return 7
        return 10 if self._action_mode == '10d' else 18

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
