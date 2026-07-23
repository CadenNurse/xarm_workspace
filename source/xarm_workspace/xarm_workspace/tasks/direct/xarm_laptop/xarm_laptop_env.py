# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_conjugate, sample_uniform

from .xarm_laptop_env_cfg import XarmLaptopEnvCfg  



class XarmLaptopEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: XarmLaptopEnvCfg

    def __init__(self, cfg: XarmLaptopEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
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

            # Return pose as [pos(3), quat_xyzw(4)]
            return torch.tensor([px, py, pz, qx, qy, qz, qw], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.control_joint_ids, _ = self._robot.find_joints(
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "drive_joint"]
        )

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits.torch[0, self.control_joint_ids, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits.torch[0, self.control_joint_ids, 1].to(self.device)

        self.obs_dof_lower_limits = self._robot.data.soft_joint_pos_limits.torch[0, :, 0].to(self.device)
        self.obs_dof_upper_limits = self._robot.data.soft_joint_pos_limits.torch[0, :, 1].to(self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[-1] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, len(self.control_joint_ids)), device=self.device)

        stage = get_current_stage()
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")

        # print("robot prim valid:", robot_prim.IsValid())
        # for prim in robot_prim.GetChildren():
        #     print("child:", prim.GetPath())
        #     for child in prim.GetChildren():
        #         print("  subchild:", child.GetPath())
        #         for gchild in child.GetChildren():
        #             print("    gchild:", gchild.GetPath())

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

        # Laptop local push pose: [pos(3), quat_xyzw(4)] - identity quaternion is [0,0,0,1]
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
        # self.left_finger_link_idx = self._robot.find_bodies("left_finger")[0][0]
        # self.right_finger_link_idx = self._robot.find_bodies("right_finger")[0][0]
        self.lid_link_idx = self._laptop.find_bodies("lid_link")[0][0]
        self.hinge_joint_idx = self._laptop.find_joints("hinge_joint")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.lid_push_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.lid_push_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Sticky per-env flag: True once the lid was closed past the success threshold.
        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._laptop = Articulation(self.cfg.laptop)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["laptop"] = self._laptop

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self.control_joint_ids)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        hinge_pos = self._laptop.data.joint_pos.torch[:, self.hinge_joint_idx]
        terminated = hinge_pos <= self.cfg.success_lid_angle_threshold
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        hinge_pos = self._laptop.data.joint_pos.torch[:, self.hinge_joint_idx]
        hinge_error = torch.abs(hinge_pos - self.cfg.target_lid_angle)

        # robot_left_finger_pos = self._robot.data.body_pos_w.torch[:, self.left_finger_link_idx]
        # robot_right_finger_pos = self._robot.data.body_pos_w.torch[:, self.right_finger_link_idx]
        
        d = torch.linalg.norm(self.robot_grasp_pos - self.lid_push_pos, dim=-1)
        reach_reward = 1.0 / (1.0 + d * d)
        reach_reward = reach_reward * reach_reward

        close_reward = -hinge_error
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        success_bonus = torch.where(
            hinge_pos <= self.cfg.success_lid_angle_threshold,
            torch.ones_like(hinge_pos),
            torch.zeros_like(hinge_pos),
        )

        total_reward = (
            self.cfg.reach_reward_scale * reach_reward
            + self.cfg.close_reward_scale * close_reward
            - self.cfg.action_penalty_scale * action_penalty
            + 5.0 * success_bonus
        )

        self._episode_succeeded |= hinge_pos <= self.cfg.success_lid_angle_threshold

        self.extras["log"] = {
            "hinge_pos": hinge_pos.mean(),
            "hinge_error": hinge_error.mean(),
            "reach_reward": (self.cfg.reach_reward_scale * reach_reward).mean(),
            "close_reward": (self.cfg.close_reward_scale * close_reward).mean(),
            "action_penalty": (-self.cfg.action_penalty_scale * action_penalty).mean(),
            "success_bonus": (5.0 * success_bonus).mean(),
            "total_reward": total_reward.mean(),
        }

        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        # Flush per-episode success (sticky binary: lid ever closed past the cfg threshold).
        laptop_pos = self._laptop.data.joint_pos.torch[env_ids, self.hinge_joint_idx]
        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self._episode_succeeded[env_ids].float().mean().item()
        log["Metrics/laptop_pos"] = laptop_pos.mean().item()
        self._episode_succeeded[env_ids] = False

        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids][:, self.control_joint_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), len(self.control_joint_ids)),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        # replace lower section with this if error occurs
        # self._robot.set_joint_position_target(joint_pos, joint_ids=self.control_joint_ids, env_ids=env_ids)
        # self._robot.write_joint_state_to_sim(
        #     joint_pos,
        #     joint_vel,
        #     joint_ids=self.control_joint_ids,
        #     env_ids=env_ids,
        # )

        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.set_joint_position_target(joint_pos, joint_ids=self.control_joint_ids, env_ids=env_ids)
        self._robot.write_joint_position_to_sim(position=joint_pos, joint_ids=self.control_joint_ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim(velocity=joint_vel, joint_ids=self.control_joint_ids, env_ids=env_ids)

        # laptop state
        laptop_joint_pos = torch.full(
            (len(env_ids), self._laptop.num_joints),
            self.cfg.start_lid_angle,
            device=self.device,
        )
        laptop_joint_vel = torch.zeros_like(laptop_joint_pos)

        # replace below if getting index issues
        # self._laptop.write_joint_position_to_sim(position=laptop_joint_pos, env_ids=env_ids)
        # self._laptop.write_joint_velocity_to_sim(velocity=laptop_joint_vel, env_ids=env_ids)
        self._laptop.write_joint_position_to_sim_index(position=laptop_joint_pos, env_ids=env_ids)
        self._laptop.write_joint_velocity_to_sim_index(velocity=laptop_joint_vel, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
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

    # auxiliary methods

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

    # def _compute_rewards(
    #     self,
    #     actions,
    #     laptop_dof_pos,
    #     franka_grasp_pos,
    #     lid_push_pos,
    #     franka_grasp_rot,
    #     lid_push_rot,
    #     franka_lfinger_pos,
    #     franka_rfinger_pos,
    #     gripper_forward_axis,
    #     lid_close_axis,
    #     gripper_up_axis,
    #     lid_up_axis,
    #     num_envs,
    #     dist_reward_scale,
    #     rot_reward_scale,
    #     open_reward_scale,
    #     action_penalty_scale,
    #     finger_reward_scale,
    #     joint_positions,
    # ):
        # # distance from hand to the laptop
        # d = torch.linalg.norm(franka_grasp_pos - lid_push_pos, ord=2, dim=-1)
        # dist_reward = 1.0 / (1.0 + d**2)
        # dist_reward *= dist_reward
        # dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        # axis1 = quat_apply(franka_grasp_rot, gripper_forward_axis)
        # axis2 = quat_apply(lid_push_rot, lid_close_axis)
        # axis3 = quat_apply(franka_grasp_rot, gripper_up_axis)
        # axis4 = quat_apply(lid_push_rot, lid_up_axis)

        # dot1 = (
        #     torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        # )  # alignment of forward axis for gripper
        # dot2 = (
        #     torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        # )  # alignment of up axis for gripper
        # # reward for matching the orientation of the hand to the laptop (fingers wrapped)
        # rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # # regularization on the actions (summed for each environment)
        # action_penalty = torch.sum(actions**2, dim=-1)

        # # how far the laptop has been closed
        # open_reward = laptop_dof_pos[:, self.hinge_joint_idx]  # laptop_hinge_joint

        # # penalty for distance of each finger from the laptop lid
        # lfinger_dist = franka_lfinger_pos[:, 2] - lid_push_pos[:, 2]
        # rfinger_dist = lid_push_pos[:, 2] - franka_rfinger_pos[:, 2]
        # finger_dist_penalty = torch.zeros_like(lfinger_dist)
        # finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        # finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        # rewards = (
        #     dist_reward_scale * dist_reward
        #     + rot_reward_scale * rot_reward
        #     + open_reward_scale * open_reward
        #     + finger_reward_scale * finger_dist_penalty
        #     - action_penalty_scale * action_penalty
        # )

        # self.extras["log"] = {
        #     "dist_reward": (dist_reward_scale * dist_reward).mean(),
        #     "rot_reward": (rot_reward_scale * rot_reward).mean(),
        #     "open_reward": (open_reward_scale * open_reward).mean(),
        #     "action_penalty": (-action_penalty_scale * action_penalty).mean(),
        #     "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
        #     "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
        #     "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        # }

        # # bonus for closing laptop properly
        # laptop_pos = laptop_dof_pos[:, self.hinge_joint_idx]
        # rewards = torch.where(laptop_pos > 0.01, rewards + 0.25, rewards)
        # rewards = torch.where(laptop_pos > 0.2, rewards + 0.25, rewards)
        # rewards = torch.where(laptop_pos > 0.35, rewards + 0.25, rewards)

        # return rewards

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