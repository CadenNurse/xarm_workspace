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
# from isaaclab.sensors import ContactSensorCfg
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parents[5]
_ASSET_DIR = _PKG_ROOT / "isaac_lab_test"


@configclass
class XarmPickPlaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.0
    decimation = 2
    action_space = 8
    observation_space = 30
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
            usd_path=str(_ASSET_DIR / "L_xarm.usd"),
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

    # pick-and-place object: 3cm deep (x), 7cm wide (y), 12cm tall (z)
    object_size = (0.03, 0.07, 0.12)
    obj = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=object_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.40, 0.80, 0.86),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # fixed visual marker at the place target (no collision, kinematic, never randomized)
    place_marker = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PlaceMarker",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.65, 0.25), opacity=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.20, 0.801),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # table
    workstation = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Workstation",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSET_DIR / "workstation.usd"),
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

    # table / workspace geometry
    table_surface_height = 0.80
    fixed_object_pos = (0.50, 0.60)
    # object_pos_x_range = (0.30, 0.70)
    # object_pos_y_range = (0.10, 0.90)

    # fixed place target (env-local, never randomized)
    # place_target_pos = (0.60, 0.20, 0.801 + 0.06)  # + half object height

    # reward scales
    dist_reward_scale = 0.5
    grasp_reward_scale = 10.0
    lift_reward_scale = 12.0
    action_penalty_scale = 0.01

    # thresholds
    grasp_dist_threshold = 0.05
    gripper_close_threshold = 0.5
    lift_height_threshold = 0.05
    pick_success_height = 0.15
    fall_height_threshold = 0.82  # object z (relative to env ground) below this counts as fallen