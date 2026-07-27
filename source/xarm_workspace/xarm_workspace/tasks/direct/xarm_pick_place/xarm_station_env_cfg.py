# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.sensors import ContactSensorCfg


@configclass
class XarmStationEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 4.0  # 500 timesteps
    decimation = 2
    action_space = 8
    observation_space = 31
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/cadennurse/Documents/isaac_lab_test/L_xarm.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": -1.5708,
                "joint7": 0.0,
                "drive_joint": 0.0,
            },
            pos=(0.172, 0.31, 0.798),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "xarm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=800.0,
                damping=80.0,
            ),
            "xarm_forearm": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=800.0,
                damping=80.0,
            ),
            "xarm_hand": ImplicitActuatorCfg(
                joint_names_expr=["drive_joint"],
                effort_limit_sim=200.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )

    # laptop
    laptop = ArticulationCfg(
        prim_path="/World/envs/env_.*/Laptop",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/cadennurse/Documents/isaac_lab_test/thinkpad_x13_gen1_REAL.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.40, 0.80, 0.80),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={"hinge_joint": 1.1},
        ),
        actuators={
            "hinge": ImplicitActuatorCfg(
                joint_names_expr=["hinge_joint"],
                effort_limit_sim=10.0,
                stiffness=0.0,
                damping=0.25,
                friction=1.6,
            ),
        },
    )

    # table
    workstation = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Workstation",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/cadennurse/Documents/isaac_lab_test/workstation.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),

        ),
    )

    # contact sensors
    # link2_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/link2",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )
    # link3_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/link3",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )
    # link4_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/link4",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )
    # link5_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/link5",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )
    # link6_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/link6",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )
    # link7_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/link7",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )
    # gripper_base_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/gripper/xarm_gripper_base_link",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )
    # right_outer_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/gripper/right_outer_knuckle",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )
    # left_outer_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/gripper/left_outer_knuckle",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )
    # right_finger_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/gripper/right_finger",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )
    # left_finger_contact = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/xarm7_L/gripper/left_finger",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/envs/env_.*/Workstation"],
    # )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    reach_reward_scale = 1.0
    close_reward_scale = 12.0
    action_penalty_scale = 0.02
    success_bonus_scale = 18.0
    close_vel_reward_scale = 5.0
    fast_close_penalty_scale = 3.0
    overshoot_penalty_scale = 8.0
    # workstation_contact_penalty_scale = 0.3

    finger_reach_reward_scale = 1.0
    finger_close_bonus_scale = 2.0
    body_push_penalty_scale = 2.5

    target_lid_angle = 0.20

    # success criteria
    success_lid_angle_threshold : float = 0.30
    """laptop joint position below which the lid is considered successfully closed [rads]."""
