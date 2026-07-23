# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass


@configclass
class XarmLaptopEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.3333  # 500 timesteps
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
        num_envs=64, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/cadennurse/Documents/isaac_lab_test/L_xarm.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
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
            pos=(0.6, 0.0, 0.0),
            rot=(0.0, 0.0, 1.0, 0.0),
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
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={"hinge_joint": 0.70},
        ),
        actuators={
            "hinge": ImplicitActuatorCfg(
                joint_names_expr=["hinge_joint"],
                effort_limit_sim=10.0,
                stiffness=0.0,
                damping=0.05,
                friction=1.0,
            ),
        },
    )

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
    close_reward_scale = 18.0
    action_penalty_scale = 0.02
    laptop_move_penalty_scale = 2.0
    laptop_tilt_penalty_scale = 1.0
    laptop_base_pos_penalty_scale = 4.0
    success_bonus_scale = 10.0
    close_vel_reward_scale = 2.0

    target_lid_angle = 0.10
    start_lid_angle = 0.60

    # success criteria
    success_lid_angle_threshold : float = 0.15
    """laptop joint position below which the lid is considered successfully closed [rads]."""
