# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_conjugate, sample_uniform

from .xarm_station_staged_env_cfg import XarmStationStagedEnvCfg


class XarmStationStagedEnv(DirectRLEnv):
    cfg: XarmStationStagedEnvCfg

    PHASE_CLOSE = 0
    PHASE_RETRACT = 1
    PHASE_RETURN = 2

    def __init__(self, cfg: XarmStationStagedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real
            return torch.tensor([px, py, pz, qx, qy, qz, qw], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.control_joint_ids, _ = self._robot.find_joints(
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "drive_joint"]
        )

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits.torch[
            0, self.control_joint_ids, 0
        ].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits.torch[
            0, self.control_joint_ids, 1
        ].to(self.device)

        self.obs_dof_lower_limits = self._robot.data.soft_joint_pos_limits.torch[0, :, 0].to(self.device)
        self.obs_dof_upper_limits = self._robot.data.soft_joint_pos_limits.torch[0, :, 1].to(self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[-1] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, len(self.control_joint_ids)), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/xarm7_L/link7/link_eef")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/xarm7_L/gripper/left_finger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/xarm7_L/gripper/right_finger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]

        hand_pose_inv_rot = quat_conjugate(hand_pose[3:7])
        hand_pose_inv_pos = -quat_apply(hand_pose_inv_rot, hand_pose[0:3])

        robot_local_pose_pos, robot_local_grasp_pose_rot = combine_frame_transforms(
            hand_pose_inv_pos, hand_pose_inv_rot, finger_pose[0:3], finger_pose[3:7]
        )
        robot_local_pose_pos = robot_local_pose_pos + torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        lid_local_push_pose = torch.tensor([0.18, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0], device=self.device)
        self.lid_local_push_pos = lid_local_push_pose[0:3].repeat((self.num_envs, 1))
        self.lid_local_push_rot = lid_local_push_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.lid_close_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.lid_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("left_finger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("right_finger")[0][0]
        self.lid_link_idx = self._laptop.find_bodies("lid_link")[0][0]
        self.hinge_joint_idx = self._laptop.find_joints("hinge_joint")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.lid_push_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.lid_push_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_hinge_pos = torch.zeros(self.num_envs, device=self.device)

        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.home_joint_pos = torch.zeros((self.num_envs, len(self.control_joint_ids)), device=self.device)
        self.close_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_returned_home = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, len(self.control_joint_ids)), device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._laptop = Articulation(self.cfg.laptop)
        self._workstation = RigidObject(self.cfg.workstation)

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["laptop"] = self._laptop
        self.scene.rigid_objects["table"] = self._workstation

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self.control_joint_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        self._update_task_phase()

        home_err = self._home_joint_error()
        returned_home = (self.task_phase == self.PHASE_RETURN) & (home_err < self.cfg.home_joint_tolerance)
        self._episode_returned_home |= returned_home

        terminated = returned_home
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        self._update_task_phase()

        hinge_pos = self._laptop.data.joint_pos.torch[:, self.hinge_joint_idx]
        hinge_vel = self._laptop.data.joint_vel.torch[:, self.hinge_joint_idx]
        hinge_error = torch.abs(hinge_pos - self.cfg.target_lid_angle)

        d = torch.linalg.norm(self.robot_grasp_pos - self.lid_push_pos, dim=-1)
        reach_reward = 1.0 / (1.0 + d * d)
        reach_reward = reach_reward * reach_reward

        close_reward = -hinge_error
        action_penalty = torch.sum(self.actions ** 2, dim=-1)
        fast_close_penalty = torch.square(torch.clamp(-hinge_vel, min=0.0))

        success_bonus = torch.where(
            hinge_pos <= self.cfg.success_lid_angle_threshold,
            torch.ones_like(hinge_pos),
            torch.zeros_like(hinge_pos),
        )

        moving_closed_reward = torch.where(
            hinge_vel < 0.0,
            -hinge_vel,
            torch.zeros_like(hinge_vel),
        )

        lfinger_pos = self._robot.data.body_pos_w.torch[:, self.left_finger_link_idx]
        rfinger_pos = self._robot.data.body_pos_w.torch[:, self.right_finger_link_idx]

        lfinger_dist = torch.linalg.norm(lfinger_pos - self.lid_push_pos, dim=-1)
        rfinger_dist = torch.linalg.norm(rfinger_pos - self.lid_push_pos, dim=-1)

        finger_mid_pos = 0.5 * (lfinger_pos + rfinger_pos)
        finger_mid_dist = torch.linalg.norm(finger_mid_pos - self.lid_push_pos, dim=-1)

        finger_reach_reward = 1.0 / (1.0 + finger_mid_dist * finger_mid_dist)
        finger_reach_reward = finger_reach_reward * finger_reach_reward

        body_push_penalty = ((d < 0.05) & ((lfinger_dist > 0.06) | (rfinger_dist > 0.06))).float()

        home_err = self._home_joint_error()
        home_reward = torch.exp(-self.cfg.home_reward_k * home_err)
        return_success_bonus = (
            (self.task_phase == self.PHASE_RETURN) & (home_err < self.cfg.home_joint_tolerance)
        ).float()

        is_close = (self.task_phase == self.PHASE_CLOSE).float()
        is_retract = (self.task_phase == self.PHASE_RETRACT).float()
        is_return = (self.task_phase == self.PHASE_RETURN).float()

        close_phase_reward = (
            self.cfg.reach_reward_scale * reach_reward
            + self.cfg.close_reward_scale * close_reward
            + self.cfg.close_vel_reward_scale * moving_closed_reward
            + self.cfg.finger_reach_reward_scale * finger_reach_reward
            - self.cfg.body_push_penalty_scale * body_push_penalty
            - self.cfg.action_penalty_scale * action_penalty
            - self.cfg.fast_close_penalty_scale * fast_close_penalty
            + self.cfg.success_bonus_scale * success_bonus
        )

        retract_phase_reward = (
            0.25 * self.cfg.reach_reward_scale * reach_reward
            - self.cfg.action_penalty_scale * action_penalty
        )

        return_phase_reward = (
            self.cfg.return_home_reward_scale * home_reward
            - self.cfg.action_penalty_scale * action_penalty
            + self.cfg.return_success_bonus_scale * return_success_bonus
        )

        total_reward = (
            is_close * close_phase_reward
            + is_retract * retract_phase_reward
            + is_return * return_phase_reward
        )

        self._episode_succeeded |= hinge_pos <= self.cfg.success_lid_angle_threshold

        self.extras["log"] = {
            "hinge_pos": hinge_pos.mean(),
            "hinge_error": hinge_error.mean(),
            "hinge_vel": hinge_vel.mean(),
            "reach_reward": (self.cfg.reach_reward_scale * reach_reward).mean(),
            "close_reward": (self.cfg.close_reward_scale * close_reward).mean(),
            "close_vel_reward": (self.cfg.close_vel_reward_scale * moving_closed_reward).mean(),
            "fast_close_penalty": (-self.cfg.fast_close_penalty_scale * fast_close_penalty).mean(),
            "action_penalty": (-self.cfg.action_penalty_scale * action_penalty).mean(),
            "finger_reach_reward": (self.cfg.finger_reach_reward_scale * finger_reach_reward).mean(),
            "body_push_penalty": (-self.cfg.body_push_penalty_scale * body_push_penalty).mean(),
            "lfinger_dist": lfinger_dist.mean(),
            "rfinger_dist": rfinger_dist.mean(),
            "home_err": home_err.mean(),
            "home_reward": (self.cfg.return_home_reward_scale * home_reward).mean(),
            "return_success_bonus": (self.cfg.return_success_bonus_scale * return_success_bonus).mean(),
            "success_bonus": (self.cfg.success_bonus_scale * success_bonus).mean(),
            "task_phase": self.task_phase.float().mean(),
            "phase_close_frac": (self.task_phase == self.PHASE_CLOSE).float().mean(),
            "phase_retract_frac": (self.task_phase == self.PHASE_RETRACT).float().mean(),
            "phase_return_frac": (self.task_phase == self.PHASE_RETURN).float().mean(),
            "total_reward": total_reward.mean(),
        }

        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        laptop_pos = self._laptop.data.joint_pos.torch[env_ids, self.hinge_joint_idx]

        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self._episode_succeeded[env_ids].float().mean().item()
        log["Metrics/return_home_rate"] = self._episode_returned_home[env_ids].float().mean().item()
        log["Metrics/laptop_pos"] = laptop_pos.mean().item()
        log["Metrics/mean_home_err"] = self._home_joint_error()[env_ids].mean().item()
        log["Metrics/phase_return_frac"] = (self.task_phase[env_ids] == self.PHASE_RETURN).float().mean().item()
        log["Metrics/phase_retract_frac"] = (self.task_phase[env_ids] == self.PHASE_RETRACT).float().mean().item()

        self._episode_succeeded[env_ids] = False
        self._episode_returned_home[env_ids] = False

        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos.torch[env_ids][:, self.control_joint_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), len(self.control_joint_ids)),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.set_joint_position_target(joint_pos, joint_ids=self.control_joint_ids, env_ids=env_ids)
        self._robot.write_joint_position_to_sim(position=joint_pos, joint_ids=self.control_joint_ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim(velocity=joint_vel, joint_ids=self.control_joint_ids, env_ids=env_ids)

        self.home_joint_pos[env_ids] = joint_pos
        self.task_phase[env_ids] = self.PHASE_CLOSE
        self.close_step_buf[env_ids] = 0

        laptop_root_pose = self._laptop.data.default_root_state.torch[env_ids, :7].clone()
        laptop_root_pose[:, 0:3] += self.scene.env_origins[env_ids]
        laptop_root_vel = torch.zeros_like(self._laptop.data.default_root_state.torch[env_ids, 7:])

        self._laptop.write_root_pose_to_sim(laptop_root_pose, env_ids=env_ids)
        self._laptop.write_root_velocity_to_sim(laptop_root_vel, env_ids=env_ids)

        laptop_joint_pos = self._laptop.data.default_joint_pos.torch[env_ids].clone()
        laptop_joint_vel = torch.zeros_like(laptop_joint_pos)

        self._laptop.write_joint_position_to_sim_index(position=laptop_joint_pos, env_ids=env_ids)
        self._laptop.write_joint_velocity_to_sim_index(velocity=laptop_joint_vel, env_ids=env_ids)

        self.prev_hinge_pos[env_ids] = laptop_joint_pos[:, self.hinge_joint_idx]
        self.prev_actions[env_ids] = 0.0

        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos.torch - self.obs_dof_lower_limits)
            / (self.obs_dof_upper_limits - self.obs_dof_lower_limits)
            - 1.0
        )
        to_target = self.lid_push_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel.torch * self.cfg.dof_velocity_scale,
                to_target,
                self._laptop.data.joint_pos.torch[:, self.hinge_joint_idx].unsqueeze(-1),
                self._laptop.data.joint_vel.torch[:, self.hinge_joint_idx].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _home_joint_error(self) -> torch.Tensor:
        robot_joint_pos = self._robot.data.joint_pos.torch[:, self.control_joint_ids]
        return torch.linalg.norm(robot_joint_pos - self.home_joint_pos, dim=-1)

    def _update_task_phase(self):
        hinge_pos = self._laptop.data.joint_pos.torch[:, self.hinge_joint_idx]

        just_closed = (hinge_pos <= self.cfg.success_lid_angle_threshold) & (self.task_phase == self.PHASE_CLOSE)
        self.task_phase[just_closed] = self.PHASE_RETRACT
        self.close_step_buf[just_closed] = self.episode_length_buf[just_closed]

        retract_elapsed = self.episode_length_buf - self.close_step_buf
        retract_done = (self.task_phase == self.PHASE_RETRACT) & (retract_elapsed > self.cfg.retract_steps)
        self.task_phase[retract_done] = self.PHASE_RETURN

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)

        hand_pos = self._robot.data.body_pos_w.torch[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w.torch[env_ids, self.hand_link_idx]
        laptop_pos = self._laptop.data.body_pos_w.torch[env_ids, self.lid_link_idx]
        laptop_rot = self._laptop.data.body_quat_w.torch[env_ids, self.lid_link_idx]

        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.lid_push_rot[env_ids],
            self.lid_push_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            laptop_rot,
            laptop_pos,
            self.lid_local_push_rot[env_ids],
            self.lid_local_push_pos[env_ids],
        )

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        laptop_rot,
        laptop_pos,
        lid_local_push_rot,
        lid_local_push_pos,
    ):
        global_franka_pos, global_franka_rot = combine_frame_transforms(
            hand_pos, hand_rot, franka_local_grasp_pos, franka_local_grasp_rot
        )
        global_laptop_pos, global_laptop_rot = combine_frame_transforms(
            laptop_pos, laptop_rot, lid_local_push_pos, lid_local_push_rot
        )

        return global_franka_rot, global_franka_pos, global_laptop_rot, global_laptop_pos