from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from tasks.g1_tasks.pick_place_cylinder_g1_29dof_inspire.pickplace_cylinder_g1_29dof_inspire_env_cfg import (
    ObjectTableSceneCfg,
    PickPlaceG129InspireBaseFixEnvCfg,
)
from tasks.common_config import CameraPresets


def _beaker_path() -> str:
    configured = os.environ.get("LAB_BEAKER_USD")
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str(Path(__file__).resolve().parents[2] / "assets/labutopia/Beaker_01.usd")


def _table_top_height() -> float:
    return float(os.environ.get("LAB_TABLE_TOP_HEIGHT_M", "0.76"))


@configclass
class LabBeakerSceneCfg(ObjectTableSceneCfg):
    """保留 Unitree 官方 G1+Inspire 本体、相机和桌面，仅替换抓取物。"""

    # Unitree 的演示场景默认生成仓库和六张桌子。采集任务只保留机器人
    # 正前方的一张桌子，减少 GUI 渲染与场景加载开销。
    room_walls = None
    packing_table_2 = None
    packing_table_3 = None
    packing_table_4 = None
    packing_table_5 = None
    packing_table_6 = None
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Collision-enabled horizontal slab whose top surface is the configured
    # real-table height. A simple primitive makes the safety boundary visually
    # obvious and avoids relying on the opaque PackingTable USD dimensions.
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.55, _table_top_height() - 0.02),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 0.80, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.65, 0.78), roughness=0.65
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    front_camera = CameraPresets.g1_front_camera()
    front_camera.data_types = ["rgb", "distance_to_image_plane"]

    if os.environ.get("LAB_OBJECT_SHAPE", "beaker") == "box":
        object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.06, 0.36, _table_top_height() + 0.025), rot=(1.0, 0.0, 0.0, 0.0)
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.035, 0.035, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_linear_velocity=2.0,
                    max_angular_velocity=4.0,
                    max_depenetration_velocity=1.0,
                    linear_damping=2.0,
                    angular_damping=5.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.85, 0.75, 0.10), roughness=0.7
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max",
                    restitution_combine_mode="min",
                    static_friction=1.8,
                    dynamic_friction=1.5,
                    restitution=0.0,
                ),
            ),
        )
    elif os.environ.get("LAB_OBJECT_SHAPE", "beaker") == "sphere":
        object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.06, 0.36, _table_top_height() + 0.025), rot=(1.0, 0.0, 0.0, 0.0)
            ),
            spawn=sim_utils.SphereCfg(
                radius=0.025,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_linear_velocity=2.0,
                    max_angular_velocity=4.0,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 0.85, 0.25), roughness=0.5
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max",
                    restitution_combine_mode="min",
                    static_friction=1.5,
                    dynamic_friction=1.2,
                    restitution=0.0,
                ),
            ),
        )
    elif os.environ.get("LAB_OBJECT_SHAPE", "beaker") == "cylinder":
        object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.06, 0.36, _table_top_height() + 0.04), rot=(1.0, 0.0, 0.0, 0.0)
            ),
            spawn=sim_utils.CylinderCfg(
                radius=0.02,
                height=0.08,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_linear_velocity=2.0,
                    max_angular_velocity=4.0,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.75, 0.05), roughness=0.5
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max",
                    restitution_combine_mode="min",
                    static_friction=1.5,
                    dynamic_friction=1.2,
                    restitution=0.0,
                ),
            ),
        )
    else:
        object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.06, 0.36, _table_top_height()), rot=(1.0, 0.0, 0.0, 0.0)
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=_beaker_path(),
                scale=(0.01, 0.01, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_linear_velocity=2.0,
                    max_angular_velocity=4.0,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.35, 0.95), metallic=0.05, roughness=0.35,
                ),
            ),
        )


@configclass
class LabBeakerG1InspireEnvCfg(PickPlaceG129InspireBaseFixEnvCfg):
    scene: LabBeakerSceneCfg = LabBeakerSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
