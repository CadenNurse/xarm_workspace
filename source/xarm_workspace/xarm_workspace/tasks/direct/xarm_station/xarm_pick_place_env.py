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

from .xarm_pick_place_env_cfg import XarmPickPlaceEnvCfg


class XarmPickPlaceEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: XarmPickPlaceEnvCfg

    def __init__(self, cfg: XarmPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
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

            return torch.tensor([px, py, pz, qx, qy, qz, qw], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.control_joint_ids, _ = self._robot.find_joints(
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "drive_joint"]
        )

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits.torch[0, self.control_joint_ids, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits.torch[0, self.control_joint_ids, 1].to(self.device)

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

        self.hand_link_idx = self._robot.find_bodies("link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("left_finger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("right_finger")[0][0]
        self.drive_joint_idx = self.control_joint_ids[-1]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # fixed place target per env (never randomized after this)
        place_target = torch.tensor(self.cfg.place_target_pos, device=self.device)
        self.place_target_pos = place_target.repeat((self.num_envs, 1)) + self.scene.env_origins

        # per-episode grasp/lift state
        self._is_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.obj)
        self._place_marker = RigidObject(self.cfg.place_marker)
        self._workstation = RigidObject(self.cfg.workstation)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object
        self.scene.rigid_objects["place_marker"] = self._place_marker
        self.scene.rigid_objects["table"] = self._workstation

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

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
        object_pos = self._object.data.root_pos_w.torch
        target_dist = torch.linalg.norm(object_pos - self.place_target_pos, dim=-1)
        near_table_height = torch.abs(object_pos[:, 2] - self.place_target_pos[:, 2]) < self.cfg.place_height_threshold

        success = self._was_lifted & (target_dist < self.cfg.place_dist_threshold) & near_table_height
        too_early = self.episode_length_buf < 30
        terminated = success & ~too_early
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        object_pos = self._object.data.root_pos_w.torch
        object_height = object_pos[:, 2] - self.cfg.table_surface_height

        lfinger_pos = self._robot.data.body_pos_w.torch[:, self.left_finger_link_idx]
        rfinger_pos = self._robot.data.body_pos_w.torch[:, self.right_finger_link_idx]
        finger_mid_pos = 0.5 * (lfinger_pos + rfinger_pos)

        lfinger_dist = torch.linalg.norm(lfinger_pos - object_pos, dim=-1)
        rfinger_dist = torch.linalg.norm(rfinger_pos - object_pos, dim=-1)
        finger_mid_dist = torch.linalg.norm(finger_mid_pos - object_pos, dim=-1)

        gripper_pos = self._robot.data.joint_pos.torch[:, self.drive_joint_idx]
        gripper_closed = gripper_pos > self.cfg.gripper_close_threshold

        fingers_near_object = ((lfinger_dist < self.cfg.grasp_dist_threshold) & (rfinger_dist < self.cfg.grasp_dist_threshold))
        self._is_grasped = fingers_near_object & gripper_closed

        lifted = self._is_grasped & (object_height > self.cfg.lift_height_threshold)
        self._was_lifted |= lifted

        target_dist = torch.linalg.norm(object_pos - self.place_target_pos, dim=-1)

        # near-object reward: pulls the gripper toward the object before grasping
        near_object_reward = 1.0 / (1.0 + finger_mid_dist * finger_mid_dist)
        near_object_reward = near_object_reward * near_object_reward

        # close reward: reward closing the gripper only once the fingers are near the object
        close_reward = fingers_near_object.float() * gripper_pos.clamp(min=0.0)

        # lift reward: object height above the table, gated on an actual grasp
        lift_reward = self._is_grasped.float() * torch.clamp(object_height, min=0.0, max=0.20)

        # near-target reward: pulls the grasped object toward the fixed place target
        near_target_reward = self._was_lifted.float() * (1.0 / (1.0 + target_dist * target_dist))

        # place bonus: object at the target, near table height, gripper opening to release
        gripper_releasing = gripper_pos < (self.cfg.gripper_close_threshold * 0.5)
        near_table_height = torch.abs(object_pos[:, 2] - self.place_target_pos[:, 2]) < self.cfg.place_height_threshold
        place_bonus = torch.where(
            self._was_lifted & (target_dist < self.cfg.place_dist_threshold) & near_table_height & gripper_releasing,
            torch.ones_like(target_dist),
            torch.zeros_like(target_dist),
        )

        # drop penalty: object was lifted then fell before reaching the target
        dropped = self._was_lifted & (object_height < self.cfg.lift_height_threshold * 0.5) & (target_dist > self.cfg.place_dist_threshold)
        drop_penalty = dropped.float()

        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        total_reward = (
            self.cfg.near_object_reward_scale * near_object_reward
            + self.cfg.close_reward_scale * close_reward
            + self.cfg.lift_reward_scale * lift_reward
            + self.cfg.near_target_reward_scale * near_target_reward
            + self.cfg.place_bonus_scale * place_bonus
            - self.cfg.drop_penalty_scale * drop_penalty
            - self.cfg.action_penalty_scale * action_penalty
        )

        self._episode_succeeded |= (self._was_lifted & (target_dist < self.cfg.place_dist_threshold) & near_table_height)

        self.extras["log"] = {
            "object_height": object_height.mean(),
            "target_dist": target_dist.mean(),
            "near_object_reward": (self.cfg.near_object_reward_scale * near_object_reward).mean(),
            "close_reward": (self.cfg.close_reward_scale * close_reward).mean(),
            "lift_reward": (self.cfg.lift_reward_scale * lift_reward).mean(),
            "near_target_reward": (self.cfg.near_target_reward_scale * near_target_reward).mean(),
            "place_bonus": (self.cfg.place_bonus_scale * place_bonus).mean(),
            "drop_penalty": (-self.cfg.drop_penalty_scale * drop_penalty).mean(),
            "action_penalty": (-self.cfg.action_penalty_scale * action_penalty).mean(),
            "is_grasped": self._is_grasped.float().mean(),
            "was_lifted": self._was_lifted.float().mean(),
            "total_reward": total_reward.mean(),
        }

        return total_reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self._episode_succeeded[env_ids].float().mean().item()
        self._episode_succeeded[env_ids] = False
        self._is_grasped[env_ids] = False
        self._was_lifted[env_ids] = False

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

        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.set_joint_position_target(joint_pos, joint_ids=self.control_joint_ids, env_ids=env_ids)
        self._robot.write_joint_position_to_sim(position=joint_pos, joint_ids=self.control_joint_ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim(velocity=joint_vel, joint_ids=self.control_joint_ids, env_ids=env_ids)

        # randomize the object's pickup position each reset; place target stays fixed
        num_resets = len(env_ids)
        rand_x = sample_uniform(
            self.cfg.object_pos_x_range[0], self.cfg.object_pos_x_range[1], (num_resets, 1), self.device
        )
        rand_y = sample_uniform(
            self.cfg.object_pos_y_range[0], self.cfg.object_pos_y_range[1], (num_resets, 1), self.device
        )
        object_z = self.cfg.table_surface_height + self.cfg.object_size[2] / 2.0
        object_pos = torch.cat(
            [rand_x, rand_y, torch.full((num_resets, 1), object_z, device=self.device)], dim=-1
        )

        object_root_pose = torch.zeros((num_resets, 7), device=self.device)
        object_root_pose[:, 0:3] = object_pos + self.scene.env_origins[env_ids]
        object_root_pose[:, 6] = 1.0  # identity quaternion
        object_root_vel = torch.zeros((num_resets, 6), device=self.device)

        self._object.write_root_pose_to_sim(object_root_pose, env_ids=env_ids)
        self._object.write_root_velocity_to_sim(object_root_vel, env_ids=env_ids)

        # place marker: always the same fixed pose, just rewritten defensively on reset
        marker_root_pose = torch.zeros((num_resets, 7), device=self.device)
        marker_root_pose[:, 0:3] = self.place_target_pos[env_ids]
        marker_root_pose[:, 6] = 1.0
        marker_root_vel = torch.zeros((num_resets, 6), device=self.device)

        self._place_marker.write_root_pose_to_sim(marker_root_pose, env_ids=env_ids)
        self._place_marker.write_root_velocity_to_sim(marker_root_vel, env_ids=env_ids)

        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos.torch - self.obs_dof_lower_limits)
            / (self.obs_dof_upper_limits - self.obs_dof_lower_limits)
            - 1.0
        )

        object_pos = self._object.data.root_pos_w.torch
        to_object = object_pos - self.robot_grasp_pos
        object_to_target = self.place_target_pos - object_pos
        object_height = (object_pos[:, 2] - self.cfg.table_surface_height).unsqueeze(-1)

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel.torch * self.cfg.dof_velocity_scale,
                to_object,
                object_to_target,
                object_height,
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

        self.robot_grasp_pos[env_ids], self.robot_grasp_rot[env_ids] = combine_frame_transforms(
            hand_pos, hand_rot, self.robot_local_grasp_pos[env_ids], self.robot_local_grasp_rot[env_ids]
        )