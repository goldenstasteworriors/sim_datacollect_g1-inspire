"""Isaac Lab task registration for a LabUtopia beaker and Unitree G1 Inspire hand."""

import gymnasium as gym

gym.register(
    id="Isaac-PickPlace-LabBeaker-G129-Inspire-Right",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sim_tasks.lab_beaker_g1_inspire.env_cfg:LabBeakerG1InspireEnvCfg"
        )
    },
)

