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
"""Standing task for Open Duck Mini V2. (based on Berkeley Humanoid)"""

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
from playground.common.rewards import (
    cost_orientation,
    cost_torques,
    cost_action_rate,
    reward_alive,
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
)
from playground.dcubot.custom_rewards import (
    cost_stand_still_dcubot,
    cost_head_pos_dcubot,
    reward_recovery_dcubot,
    reward_recovery_blend_dcubot,
    cost_contact_impact_dcubot,
    CONTACT_IMPACT_BODY_WEIGHTS,
)

# No imitation reward for DCUBOT (see project memory: Placo reference-motion
# generation didn't work for this robot's leg geometry).

# --- Ablation flags for the Disney-paper-inspired reward/init changes below
# (RSS 2024 "Design and Control of a Bipedal Robotic Character" / 2025 "Robot
# Crash Course", arXiv:2511.10635 -- reward-shaping and init-distribution
# ideas only, their hardware-specific actuator-sysID/mocap pieces are not
# used). Flip to False to fall back to the pre-existing single-term behavior
# for an A/B comparison. ---
USE_RECOVERY_BLEND = True       # split "recovery" into impact-min (early) / settle (late), cubic-spline blended
USE_CONTACT_IMPACT_COST = True  # per-body-part weighted non-foot contact-force penalty
USE_FALLEN_INIT = True          # mix in already-tilted, high-velocity reset states


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        # episode_length=450,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
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
                joint_vel=2.5,  # rad/s # Was 1.5
                gravity=0.1,
                linvel=0.1,
                gyro=0.05,
                accelerometer=0.005,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                orientation=-0.5,
                torques=-1.0e-3,
                action_rate=-0.375,  # was -1.5
                stand_still=-0.3,  # was -1.0 TODO try to relax this a bit ?
                alive=5.0,  # was 20.0 -- alone used to dominate every other term
                tracking_lin_vel=2.0,  # new: reward tracking command[:3]=0 (unitree_rl_mjlab style),
                tracking_ang_vel=2.0,  # not just "surviving" regardless of active balance
                head_pos=-2.0,
                recovery=3.0,  # new: bonus for actively reducing tilt once significantly tilted
                contact_impact=-0.05,  # new (USE_CONTACT_IMPACT_COST): weighted non-foot contact force
            ),
            # joystick.py shares one tracking_sigma=0.01 for lin+ang, tuned around its
            # small walking-speed commands (~0.1 m/s). Standing's errors come from push
            # kicks (0.1-1.0 m/s, push_config below) and recovery angular rates (~1 rad/s+),
            # an order of magnitude bigger -- reusing 0.01 would crush tracking reward to
            # ~0 during almost any real push/recovery, punishing the exact behavior we want.
            tracking_sigma_lin_vel=0.1,
            tracking_sigma_ang_vel=1.0,
            recovery_tilt_threshold_deg=35.0,  # above normal sway (~25 deg seen), well below the 90 deg fall/termination cutoff
            recovery_blend_duration=0.5,  # s; USE_RECOVERY_BLEND cubic-spline window from impact-min to settle-focused recovery
            # stand_still used to be gated on command size, which is always 0 for this
            # stand-still task -- i.e. permanently on, penalizing recovery motion as hard
            # as idle drift (confirmed: it outweighed the recovery bonus on 96% of
            # crisis steps). Gate on measured velocity instead, per unitree_rl_lab.
            stand_still_lin_vel_threshold=0.3,
            stand_still_ang_vel_threshold=1.0,
            stand_still_reduction_factor=0.2,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        # USE_FALLEN_INIT: mix in episodes that reset already tilted and
        # moving, instead of only the small-noise near-upright reset below.
        # Ranges are as requested: roll/pitch sampled uniformly in
        # +-roll_pitch_range_deg (sign stratified -- see reset(), so fall
        # direction isn't biased toward one side), base linvel in
        # +-lin_vel_range m/s, base angvel and joint vel in +-ang_vel_range
        # rad/s.
        fallen_init_config=config_dict.create(
            enable=True,
            prob=0.3,
            roll_pitch_range_deg=30.0,
            lin_vel_range=2.0,
            ang_vel_range=0.5,
        ),
        # lin_vel_x=[-0.1, 0.15],
        # lin_vel_y=[-0.2, 0.2],
        # ang_vel_yaw=[-1.0, 1.0],  # [-1.0, 1.0]
        neck_pitch_range=[-0.34, 1.1],
        head_pitch_range=[-0.78, 0.78],
        head_yaw_range=[-2.7, 2.7],
        head_range_factor=1.0,
    )


class Standing(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Standing policy"""

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

        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)

        # Note: First joint is freejoint.
        # get the range of the joints
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
                # 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, #head
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # right leg.
            ]
        )

        self._njoints = self._mj_model.njnt  # number of joints
        self._actuators = self._mj_model.nu  # number of actuators

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

        # noise in the simu?
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
        # qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

        # Precompute body ids/weights for cost_contact_impact_dcubot (USE_CONTACT_IMPACT_COST).
        body_ids, body_weights = [], []
        for name, w in CONTACT_IMPACT_BODY_WEIGHTS.items():
            body_ids.append(self._mj_model.body(name).id)
            body_weights.append(w)
        self._contact_impact_body_ids = jp.array(body_ids)
        self._contact_impact_body_weights = jp.array(body_weights)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q  # the complete qpos
        # print(f'DEBUG0 init qpos: {qpos}')
        qvel = jp.zeros(self.mjx_model.nv)

        # init position/orientation in environment
        # x=+U(-0.05, 0.05), y=+U(-0.05, 0.05), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        # floating base
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )  # x y noise

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
            quat,
        )  # yaw noise

        # USE_FALLEN_INIT: with probability fallen_init_config.prob, also tilt
        # the reset orientation by +-roll_pitch_range_deg. Sign of roll/pitch
        # is drawn from {-1, +1} separately (not just a continuous uniform
        # draw) so small batches can't end up biased toward tipping one way --
        # all 4 sign combinations are equally likely regardless of batch size.
        fallen_prob = (
            self._config.fallen_init_config.prob
            if (self._config.fallen_init_config.enable and USE_FALLEN_INIT)
            else 0.0
        )
        rng, fallen_rng, rsign_rng, psign_rng, rmag_rng, pmag_rng = jax.random.split(
            rng, 6
        )
        use_fallen = jax.random.bernoulli(fallen_rng, p=fallen_prob)
        roll_sign = jax.random.randint(rsign_rng, (1,), 0, 2) * 2 - 1
        pitch_sign = jax.random.randint(psign_rng, (1,), 0, 2) * 2 - 1
        roll_mag = jax.random.uniform(
            rmag_rng,
            (1,),
            minval=0.0,
            maxval=jp.radians(self._config.fallen_init_config.roll_pitch_range_deg),
        )
        pitch_mag = jax.random.uniform(
            pmag_rng,
            (1,),
            minval=0.0,
            maxval=jp.radians(self._config.fallen_init_config.roll_pitch_range_deg),
        )
        roll_quat = math.axis_angle_to_quat(
            jp.array([1.0, 0.0, 0.0]), roll_sign * roll_mag
        )
        pitch_quat = math.axis_angle_to_quat(
            jp.array([0.0, 1.0, 0.0]), pitch_sign * pitch_mag
        )
        fallen_quat = math.quat_mul(new_quat, math.quat_mul(roll_quat, pitch_quat))
        new_quat = jp.where(use_fallen, fallen_quat, new_quat)

        base_qpos = base_qpos.at[3:7].set(new_quat)

        qpos = self.set_floating_base_qpos(base_qpos, qpos)
        # print(f'DEBUG1 base qpos: {qpos}')
        # init joint position
        # qpos[7:]=*U(0.0, 0.1)
        rng, key = jax.random.split(rng)

        # multiply actual joints with noise (excluding floating base and backlash)
        qpos_j = self.get_actuator_joints_qpos(qpos) * jax.random.uniform(
            key, (self._actuators,), minval=0.5, maxval=1.5
        )
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)
        # print(f'DEBUG2 joint qpos: {qpos}')
        # init joint vel
        # d(xyzrpy)=U(-0.05, 0.05)
        rng, key = jax.random.split(rng)
        # qvel = qvel.at[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6].set(
        #     jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        # )

        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5), qvel
        )

        # USE_FALLEN_INIT: for the same fallen-branch episodes, also overwrite
        # base linear/angular velocity and all joint velocities with the
        # larger "mid-fall" ranges (base linvel +-lin_vel_range m/s, base
        # angvel and joint vel +-ang_vel_range rad/s) instead of the small
        # near-upright noise above.
        rng, flin_rng, fang_rng, fjoint_rng = jax.random.split(rng, 4)
        fallen_lin_vel = jax.random.uniform(
            flin_rng,
            (3,),
            minval=-self._config.fallen_init_config.lin_vel_range,
            maxval=self._config.fallen_init_config.lin_vel_range,
        )
        fallen_ang_vel = jax.random.uniform(
            fang_rng,
            (3,),
            minval=-self._config.fallen_init_config.ang_vel_range,
            maxval=self._config.fallen_init_config.ang_vel_range,
        )
        fallen_joint_vel = jax.random.uniform(
            fjoint_rng,
            (self._actuators,),
            minval=-self._config.fallen_init_config.ang_vel_range,
            maxval=self._config.fallen_init_config.ang_vel_range,
        )
        qvel = self.set_floating_base_qvel(
            jp.where(
                use_fallen,
                jp.concatenate([fallen_lin_vel, fallen_ang_vel]),
                self.get_floating_base_qvel(qvel),
            ),
            qvel,
        )
        qvel = self.set_actuator_joints_qvel(
            jp.where(use_fallen, fallen_joint_vel, self.get_actuator_joints_qvel(qvel)),
            qvel,
        )
        # print(f'DEBUG3 base qvel: {qvel}')
        ctrl = self.get_actuator_joints_qpos(qpos)
        # print(f'DEBUG4 ctrl: {ctrl}')
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
            "command": cmd,
            "prev_gravity": self.get_gravity(data),  # for reward_recovery_dcubot
            "prev_gyro": self.get_gyro(data),  # for reward_recovery_blend_dcubot (USE_RECOVERY_BLEND)
            "prev_linvel_z": self.get_global_linvel(data)[2],  # ditto
            "fall_elapsed": jp.zeros(()),  # seconds since tilt crossed recovery_tilt_threshold_deg
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
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
        ]  # action with delay

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
        )  # floating base x,y
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        motor_targets = (
            self._default_actuator + action_w_delay * self._config.action_scale
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

        # USE_RECOVERY_BLEND: update fall_elapsed BEFORE computing this step's
        # reward, so reward_recovery_blend_dcubot sees up-to-date elapsed time
        # since the tilt threshold was crossed (resets to 0 once upright).
        tilt_now = jp.arccos(jp.clip(self.get_gravity(data)[-1], -1.0, 1.0))
        is_falling_now = tilt_now > jp.radians(
            self._config.reward_config.recovery_tilt_threshold_deg
        )
        state.info["fall_elapsed"] = jp.where(
            is_falling_now, state.info["fall_elapsed"] + self.dt, 0.0
        )

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        # FIXME
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        # jax.debug.print('STEP REWARD: {}',reward)
        state.info["push"] = push
        state.info["prev_gravity"] = self.get_gravity(data)  # for next step's reward_recovery_dcubot
        state.info["prev_gyro"] = self.get_gyro(data)  # for next step's reward_recovery_blend_dcubot
        state.info["prev_linvel_z"] = self.get_global_linvel(data)[2]  # ditto
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

        # Handle IMU delay
        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        imu_idx = jax.random.randint(
            noise_rng,
            (1,),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]

        # joint_angles = data.qpos[7:]

        # Handling backlash
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

        # joint_vel = data.qvel[6:]
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)
        # info["rng"], noise_rng = jax.random.split(info["rng"])
        # noisy_linvel = (
        #     linvel
        #     + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        #     * self._config.noise_config.level
        #     * self._config.noise_config.scales.linvel
        # )

        state = jp.hstack(
            [
                # noisy_linvel,  # 3
                # noisy_gyro,  # 3
                # noisy_gravity,  # 3
                noisy_gyro,  # 3
                noisy_accelerometer,  # 3
                info["command"],  # 3
                noisy_joint_angles - self._default_actuator,  # 10
                noisy_joint_vel * self._config.dof_vel_scale,  # 10
                info["last_act"],  # 10
                info["last_last_act"],  # 10
                info["last_last_last_act"],  # 10
                contact,  # 2
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
                data.actuator_force,  # 10
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
            "orientation": cost_orientation(self.get_gravity(data)),
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(action, info["last_act"]),
            "alive": reward_alive(),
            "tracking_lin_vel": reward_tracking_lin_vel(
                info["command"],
                self.get_local_linvel(data),
                self._config.reward_config.tracking_sigma_lin_vel,
            ),
            "tracking_ang_vel": reward_tracking_ang_vel(
                info["command"],
                self.get_gyro(data),
                self._config.reward_config.tracking_sigma_ang_vel,
            ),
            "recovery": (
                reward_recovery_blend_dcubot(
                    self.get_gravity(data),
                    info["prev_gravity"],
                    self.get_gyro(data),
                    info["prev_gyro"],
                    self.get_global_linvel(data)[2],
                    info["prev_linvel_z"],
                    info["fall_elapsed"],
                    self._config.reward_config.recovery_tilt_threshold_deg,
                    self._config.reward_config.recovery_blend_duration,
                )
                if USE_RECOVERY_BLEND
                else reward_recovery_dcubot(
                    self.get_gravity(data),
                    info["prev_gravity"],
                    self._config.reward_config.recovery_tilt_threshold_deg,
                )
            ),
            "contact_impact": (
                cost_contact_impact_dcubot(
                    data._impl.cfrc_ext,
                    self._contact_impact_body_ids,
                    self._contact_impact_body_weights,
                )
                if USE_CONTACT_IMPACT_COST
                else jp.zeros(())
            ),
            "stand_still": cost_stand_still_dcubot(
                self.get_local_linvel(data),
                self.get_gyro(data),
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                self._default_actuator,
                self._config.reward_config.stand_still_lin_vel_threshold,
                self._config.reward_config.stand_still_ang_vel_threshold,
                self._config.reward_config.stand_still_reduction_factor,
                True,
            ),
            "head_pos": cost_head_pos_dcubot(
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                info["command"],
            ),
        }

        return ret

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4, rng5, rng6, rng7 = jax.random.split(rng, 7)

        # lin_vel_x = jax.random.uniform(
        #     rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        # )
        # lin_vel_y = jax.random.uniform(
        #     rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        # )
        # ang_vel_yaw = jax.random.uniform(
        #     rng3,
        #     minval=self._config.ang_vel_yaw[0],
        #     maxval=self._config.ang_vel_yaw[1],
        # )

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

        # With 10% chance, set everything to zero.
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(6),
            jp.hstack(
                [
                    0.0,
                    0.0,
                    0.0,
                    neck_pitch,
                    head_pitch,
                    head_yaw,
                ]
            ),
        )
