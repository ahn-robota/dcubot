# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for DCUBOT. (based on Berkeley Humanoid / Open Duck Mini V2)

2026-10-27 "back to origin" rewrite, v2 (omnidirectional): per explicit user
decision, this file follows upstream open_duck_mini_v2/joystick.py's
reward/obs/config structure as-is; only dcubot's XML/robot params and
reference motion are swapped in -- matching upstream's own omnidirectional
training (the user's explicit goal: "기존 오픈덕 미니랑 동일한 학습 구조로
하고 싶다"). The 2026-10-17/18 standing-dispatch/Disney-paper additions are
superseded by this rewrite (preserved in
backups/20261020_015836_pre_omnidirectional/).

History of this file today: an earlier pass today reverted to forward-only
commands after Step 1 found `polynomial_coefficients_slow.pkl` (5 keys,
dx-only) was the pkl actually wired in. Separately, the 2026-10-20 session's
210-combo omnidirectional sweep had been deleted from the project as
"data-quality broken" -- that judgment turned out to be entangled with a
replay-script bug (see memory dcubot_omnidirectional_restored_1027), not a
real problem with the data itself, so it was restored from a permanent
backup (`/home/ahn/dcubot_reference_motion_FINAL_omnidirectional/`,
read-only, outside the project tree) and is now wired in as
`polynomial_coefficients_omnidirectional.pkl` below.

Deviations from upstream structure (all deliberate, reasoned, logged here per
the user's own request to list anything that couldn't be kept identical --
see docs/CHANGES_FROM_ORIGINAL.md for the full table):

1. **Imitation reward function**: upstream's `reward_imitation` (custom_rewards.py)
   indexes its reference_frame with hardcoded slice offsets tied to ITS OWN
   PolyReferenceMotion pkl's internal layout (16-joint duck: left leg(5) +
   head/neck/antennas(6) + right leg(5)). dcubot's own reference layout is
   13 joints (left leg(5) + head(3) + right leg(5), no antennas) in Placo
   recording order, confirmed via scripts/inspect_reference.py reading the
   raw recording JSON's own "Joints" list -- a different layout, so upstream's
   reward_imitation would silently misread dcubot's pkl. Used dcubot's own
   `reward_imitation_dcubot` (custom_rewards.py) instead, already proven over
   a real 21M-step run (2026-10-07). It already self-gates on
   `cmd_norm > 0.01` internally, same semantics as upstream's
   `reward *= cmd_norm > 0.01`.

2. **Command ranges are based on each recording's own REAL ACHIEVED velocity
   (Vel_x/Vel_y/Yaw fields), NOT the pkl's intention labels** (the filename/
   key values). Checked directly against the pure single-axis recordings in
   reference_motion_generator/recordings_omnidirectional/ (2026-10-27):

   | axis | intention label range | real achieved range | ratio |
   |------|------------------------|----------------------|-------|
   | dx (fwd/back) | 0.098-0.294 | 0.100-0.300 | ~1.0x (clean) |
   | dy (strafe)   | 0.098-0.294 | 0.057-0.158 | ~0.55x |
   | dtheta (turn) | 0.98-1.961  | 0.024-0.103 rad/s | ~0.03-0.05x |

   dx and dy are close enough to 1:1 (or a consistent linear scale) to use
   directly. dtheta is NOT: commanding anywhere near the label's scale
   (up to 1.961 rad/s) would ask for a turn rate the robot/gait physically
   never gets close to (real achieved is only ~0.03-0.1 rad/s), which would
   put tracking_ang_vel (rewarding fast turning) and imitation (rewarding
   following the much-slower reference gait) into direct conflict. So
   `ang_vel_yaw`'s command range is set to the REAL achieved scale
   (+-0.103 rad/s), not the label scale. One consequence: at this achieved
   scale, `PolyReferenceMotion.vel_to_index`'s nearest-neighbor snap will
   mostly land on the dtheta=0 (straight-gait) reference frame, since 0.103
   is much closer to the grid's 0 point than to its next label (0.98) --
   turning is therefore driven mostly by tracking_ang_vel, with the
   imitation term contributing comparatively little guidance for that axis
   specifically (same situation as dx/dy would be in if their achieved
   values were similarly far below the grid spacing -- they're not, so this
   caveat is dtheta-specific). This is a data limitation of the current
   Placo sweep (its dtheta parameter apparently isn't a direct rad/s dial),
   not a code bug; revisiting the sweep generation to get faster real
   turning is future work.

3. **Head joint command is 3-dim, not 4-dim**: dcubot's head has
   neck_pitch/head_pitch/head_yaw only (no head_roll, no antennas), so
   `sample_command` returns a 6-dim vector (3 move + 3 head) instead of
   upstream's 7-dim. Head command ranges are set to dcubot's own joint
   limits read directly from xmls/dcubot.xml (neck_pitch +-0.785398,
   head_pitch/head_yaw +-1.570796), not copied from upstream's duck-specific
   neck/head ranges.

4. **tracking_sigma lowered 0.01 -> 0.003, ang_vel_yaw widened 0.103 -> 0.3
   (2026-10-28 experiment, see docs/RESULTS_omni_1024_50m.md)**: the 50M-step
   main run (and the 3.6M laptop verification run) both converged to a
   policy that stands still regardless of command, confirmed via direct
   command-following tests (near-zero world-frame velocity under fixed
   commands). Root-cause analysis found `tracking_sigma=0.01` was kept at
   upstream's value, but upstream's command ranges are all comfortably
   larger than `sqrt(sigma)=0.1`, while dcubot's `ang_vel_yaw` range
   (+-0.103, set from real-achieved-velocity per deviation #2) is almost
   exactly 1 sigma-width -- so standing still already scores ~0.37 reward
   even at the command extreme for that axis, barely distinguishing "stand"
   from "turn." This combines two of the two candidate fixes discussed:
   lowering sigma sharpens the reward within the already-achievable range
   (honest fix, doesn't ask for anything physically impossible), and
   widening ang_vel_yaw's range creates stronger gradient pressure toward
   turning as fast as the gait can manage, even though the robot still
   can't reach the new range's extreme (real achieved max is ~0.103) --
   accepted as intentional: the point is to make "turn as much as you can"
   score meaningfully better than "stand," not to make 0.3 rad/s achievable.
   This does NOT fully explain the lin_vel axes also staying near-zero
   (their sigma ratios were already comparable to upstream) -- `alive`
   dominance (~98% of total reward) and/or training scale (1024 envs/52.3M
   steps here vs upstream's up to 8192 envs/300M steps) remain candidates
   for that part, not addressed by this change. This is an experiment, not
   a settled decision -- see the results doc for the short-run outcome
   before deciding whether to carry it into another main-training attempt.
   `y_tol`=0.1 (reward_tracking_lin_vel) left unchanged per the user's own
   instruction, pending a reference-replay measurement of lateral wobble.

5. **feet_air_time/feet_gait/feet_clearance dropped**: these were
   dcubot-specific additions (not in upstream open_duck_mini_v2/joystick.py
   at all) layered on in an earlier session. Dropped for fidelity to "use
   upstream structure as-is" -- not because they were found harmful, they
   can be re-added later if gait quality needs it.

6. **Home keyframe pose** (xmls/scene_flat_terrain*.xml): replaced the
   placeholder all-zero/legs-straight pose with the reference motion's own
   standing posture (motions_dcubot/standing_0.0_0.0_0.0.json, steady-state
   average, L/R-symmetrized) -- com_height=0.19/trunk_pitch=5.5deg, per the
   user's Step 2 instruction.
"""

from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import constants
from . import base as open_duck_mini_v2_base

from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques,
    cost_action_rate,
    cost_stand_still,
    reward_alive,
)
from playground.dcubot.custom_rewards import reward_imitation_dcubot

# See module docstring deviation #1: upstream's own reward_imitation is
# layout-incompatible with dcubot's pkl, reward_imitation_dcubot used instead.
USE_IMITATION_REWARD = True
USE_MOTOR_SPEED_LIMITS = True
DCUBOT_REFERENCE_PKL = "playground/dcubot/data/polynomial_coefficients_omnidirectional.pkl"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,  # rad/s
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            action_min_delay=0,  # env steps
            action_max_delay=3,  # env steps
            imu_min_delay=0,  # env steps
            imu_max_delay=3,  # env steps
            scales=config_dict.create(
                hip_pos=0.03,  # rad, for each hip joint
                knee_pos=0.05,  # rad, for each knee joint
                ankle_pos=0.08,  # rad, for each ankle joint
                joint_vel=2.5,  # rad/s
                gravity=0.1,
                linvel=0.1,
                gyro=0.1,
                accelerometer=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=2.5,
                tracking_ang_vel=6.0,
                torques=-1.0e-3,
                action_rate=-0.5,
                stand_still=-0.2,
                alive=20.0,  # upstream's value (dcubot had lowered this to 8.0 for the now-removed standing dispatch)
                imitation=1.0,
            ),
            tracking_sigma=0.003,  # see deviation #4 above (2026-10-28 experiment)
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        gait_period=0.4,  # s; Placo period for dcubot was 0.408
        # lin_vel_x/y: based on each axis's REAL ACHIEVED velocity, not the
        # pkl's intention labels -- see deviation #2 above. ang_vel_yaw:
        # widened past the achieved scale (0.103) toward upstream's
        # proportions -- see deviation #4 (2026-10-28 experiment).
        lin_vel_x=[-0.20, 0.30],
        lin_vel_y=[-0.158, 0.158],
        ang_vel_yaw=[-0.3, 0.3],
        # dcubot's own joint limits (xmls/dcubot.xml) -- see deviation #3 above.
        neck_pitch_range=[-0.785398, 0.785398],
        head_pitch_range=[-1.570796, 1.570796],
        head_yaw_range=[-1.570796, 1.570796],
        head_range_factor=1.0,
    )


class Joystick(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Track a joystick command (omnidirectional baseline)."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:

        if USE_IMITATION_REWARD:
            self.PRM = PolyReferenceMotion(DCUBOT_REFERENCE_PKL)

        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)

        # Note: First joint is freejoint.
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # weights for computing the cost of each joints compared to a reference pose
        self._weights = jp.array(
            [
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # left leg.
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # right leg.
            ]
        )

        self._njoints = self._mj_model.njnt
        self._actuators = self._mj_model.nu

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        qpos_noise_scale = np.zeros(self._actuators)

        hip_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_hip" in j
        ]
        knee_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_knee" in j
        ]
        ankle_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_ankle" in j
        ]

        qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        qpos_noise_scale[knee_ids] = self._config.noise_config.scales.knee_pos
        qpos_noise_scale[ankle_ids] = self._config.noise_config.scales.ankle_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q  # the complete qpos
        qvel = jp.zeros(self.mjx_model.nv)

        # init position/orientation in environment
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
            quat,
        )
        base_qpos = base_qpos.at[3:7].set(new_quat)

        qpos = self.set_floating_base_qpos(base_qpos, qpos)
        rng, key = jax.random.split(rng)
        qpos_j = self.get_actuator_joints_qpos(qpos) * jax.random.uniform(
            key, (self._actuators,), minval=0.5, maxval=1.5
        )
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)

        rng, key = jax.random.split(rng)
        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05), qvel
        )
        ctrl = self.get_actuator_joints_qpos(qpos)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        info = {
            "rng": rng,
            "step": 0,
            "gait_step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # History related.
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(
            state.info["rng"], 4
        )

        # Handle action delay
        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        action_w_delay = action_history.reshape((-1, self._actuators))[
            action_idx[0]
        ]

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        )
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[
            self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2
        ].set(
            push * push_magnitude
            + qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2]
        )
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        motor_targets = (
            self._default_actuator + action_w_delay * self._config.action_scale
        )

        if USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]
            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets - self._config.max_motor_velocity * self.dt,
                prev_motor_targets + self._config.max_motor_velocity * self.dt,
            )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)
        state.info["gait_step"] += 1

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = v
                else:
                    state.metrics[f"cost/{k}"] = -v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:

        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        accelerometer = self.get_accelerometer(data)
        accelerometer.at[0].set(accelerometer[0] + 1.3)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = (
            accelerometer
            + (2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.accelerometer
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        imu_idx = jax.random.randint(
            noise_rng,
            (1,),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)
        for i in self.backlash_idx_to_add:
            joint_backlash = jp.insert(joint_backlash, i, 0)
        joint_angles = joint_angles + joint_backlash

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)

        # Gait-phase clock, upstream-style: runs unconditionally every step
        # (not gated on command magnitude) -- matches
        # open_duck_mini_v2/joystick.py's imitation_phase exactly. See
        # compare_original_vs_dcubot.md's push-kick investigation for the
        # caveat this causes for a genuinely "stand still" policy; kept as
        # the baseline default per "use upstream structure as-is".
        phase = (info["gait_step"] * self.dt / self._config.gait_period) % 1.0
        gait_phase = jp.array([jp.cos(2 * jp.pi * phase), jp.sin(2 * jp.pi * phase)])

        state = jp.hstack(
            [
                noisy_gyro,  # 3
                noisy_accelerometer,  # 3
                info["command"],  # 6 (3 move + 3 head, see deviation #3)
                noisy_joint_angles - self._default_actuator,  # 13
                noisy_joint_vel * self._config.dof_vel_scale,  # 13
                info["last_act"],  # 13
                info["last_last_act"],  # 13
                info["last_last_last_act"],  # 13
                info["motor_targets"],  # 13
                contact,  # 2
                gait_phase,  # 2
            ]
        )

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[self._floating_base_qpos_addr + 2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - self._default_actuator,
                joint_vel,
                root_height,  # 1
                data.actuator_force,  # 13
                contact,  # 2
                feet_vel,  # 4*3
                info["feet_air_time"],  # 2
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.

        ret = {
            "tracking_lin_vel": reward_tracking_lin_vel(
                info["command"],
                self.get_local_linvel(data),
                self._config.reward_config.tracking_sigma,
            ),
            "tracking_ang_vel": reward_tracking_ang_vel(
                info["command"],
                self.get_gyro(data),
                self._config.reward_config.tracking_sigma,
            ),
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(action, info["last_act"]),
            "alive": reward_alive(),
            "stand_still": cost_stand_still(
                info["command"],
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                self._default_actuator,
                ignore_head=False,
            ),
        }

        if USE_IMITATION_REWARD:
            # Omnidirectional: command[0:3] (lin_vel_x/y, ang_vel_yaw) feed
            # the PRM lookup directly, same as upstream -- see deviation #2
            # for why these ranges are achieved-velocity-based, not the
            # pkl's raw intention labels.
            reference_frame = self.PRM.get_reference_motion(
                info["command"][0],
                info["command"][1],
                info["command"][2],
                info["gait_step"],
            )
            ret["imitation"] = reward_imitation_dcubot(
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                contact,
                reference_frame,
                info["command"],
            )
        else:
            ret["imitation"] = jp.zeros(())

        return ret

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4, rng5, rng6, rng7 = jax.random.split(rng, 7)

        lin_vel_x = jax.random.uniform(
            rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        neck_pitch = jax.random.uniform(
            rng5,
            minval=self._config.neck_pitch_range[0] * self._config.head_range_factor,
            maxval=self._config.neck_pitch_range[1] * self._config.head_range_factor,
        )
        head_pitch = jax.random.uniform(
            rng6,
            minval=self._config.head_pitch_range[0] * self._config.head_range_factor,
            maxval=self._config.head_pitch_range[1] * self._config.head_range_factor,
        )
        head_yaw = jax.random.uniform(
            rng7,
            minval=self._config.head_yaw_range[0] * self._config.head_range_factor,
            maxval=self._config.head_yaw_range[1] * self._config.head_range_factor,
        )

        # With 10% chance, set everything to zero (upstream's stand-still command).
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(6),
            jp.hstack(
                [
                    lin_vel_x,
                    lin_vel_y,
                    ang_vel_yaw,
                    neck_pitch,
                    head_pitch,
                    head_yaw,
                ]
            ),
        )
