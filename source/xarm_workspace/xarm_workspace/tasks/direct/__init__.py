# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # noqa: F401


##
# Register Gym environments.
##

gym.register(
    id="Isaac-Xarm-Workspace-Direct-v0",
    entry_point="xarm_workspace.tasks.direct.xarm_cabinet.xarm_workspace_env:XarmWorkspaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "xarm_workspace.tasks.direct.xarm_cabinet.xarm_workspace_env_cfg:XarmWorkspaceEnvCfg",
        "skrl_cfg_entry_point": "xarm_workspace.tasks.direct.xarm_cabinet.agents:skrl_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Xarm-Laptop-Direct-v0",
    entry_point="xarm_workspace.tasks.direct.xarm_laptop.xarm_laptop_env:XarmLaptopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "xarm_workspace.tasks.direct.xarm_laptop.xarm_laptop_env_cfg:XarmLaptopEnvCfg",
        "skrl_cfg_entry_point": "xarm_workspace.tasks.direct.xarm_laptop.agents:skrl_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Xarm-Station-Direct-v0",
    entry_point="xarm_workspace.tasks.direct.xarm_station.xarm_station_env:XarmStationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "xarm_workspace.tasks.direct.xarm_station.xarm_station_env_cfg:XarmStationEnvCfg",
        "skrl_cfg_entry_point": "xarm_workspace.tasks.direct.xarm_station.agents:skrl_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Xarm-Pick-Place-Direct-v0",
    entry_point="xarm_workspace.tasks.direct.xarm_pick_place.xarm_pick_place_env:XarmPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "xarm_workspace.tasks.direct.xarm_pick_place.xarm_pick_place_env_cfg:XarmPickPlaceEnvCfg",
        "skrl_cfg_entry_point": "xarm_workspace.tasks.direct.xarm_pick_place.agents:skrl_ppo_cfg.yaml",
    },
)