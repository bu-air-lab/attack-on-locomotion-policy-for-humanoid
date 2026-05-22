import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


# from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG

from unitree_rl_lab.tasks.locomotion import mdp
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg



START_FIXED_SLOPE_DEG = 3.0
START_FIXED_SLOPE = math.tan(math.radians(START_FIXED_SLOPE_DEG))  
END_FIXED_SLOPE_DEG = 5.0
END_FIXED_SLOPE = math.tan(math.radians(END_FIXED_SLOPE_DEG))  

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(40.0, 10.0),
    border_width=20.0,
    num_rows=1,        # 10 difficulty levels: 0° to 6°
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.7,
        #     slope_range=(0.0, 0.105),  # 0° to 6°
        # ),
        # "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=1.0,
        #     step_height_range=(0.04, 0.08),  # 4–8 cm height range
        #     step_width=0.3,                  # width of each stair step
        #     platform_width=1.0,              # optional, flat area at top
        #     platform_height=-1.0,            # optional
        #     holes=False,                     # keep stairs solid
        # ),
        # "uneven": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #     proportion=1.0,
        #     obstacle_width_range=(0.10, 0.30),   # footprint in meters
        #     obstacle_height_range=(0.04, 0.50),  # 4–50 cm tall bumps
        #     num_obstacles=600                     # density of blocks
        # ),

        # "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion= 0.7,
        #     slope_range=(START_FIXED_SLOPE, END_FIXED_SLOPE),  
        # ),

        "hf_two_sided_slope": terrain_gen.HfTwoSidedSlopeTerrainCfg(
            proportion=0.9,
            slope_range_left=(math.radians(START_FIXED_SLOPE_DEG),
                            math.radians(END_FIXED_SLOPE_DEG)),
            slope_range_right=(math.radians(START_FIXED_SLOPE_DEG),
                            math.radians(END_FIXED_SLOPE_DEG)),

        ),
    },
    
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=[1.0, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-2.5, 2.5), "yaw": (-3.14, 3.14)},
            # "pose_range": {"x": (-0.5, 0.5), "y": (1.5, 1.5), "yaw": (-3.14, 3.14)}, # for slope
            # "pose_range": {"x": (-0.0, 0.0), "y": (0.0, 0.0), "yaw": (-1.57, -1.57)},
            # "pose_range": {"x": (-0.5, 0.5), "y": (-1.5, -1.5), "yaw": (-1.57, -1.57)},

            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )



@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )

# #zero speed
# @configclass
# class CommandsCfg:
#     """Command specifications for the MDP."""

#     base_velocity = mdp.UniformLevelVelocityCommandCfg(
#         asset_name="robot",

#         # Do NOT resample commands
#         resampling_time_range=(1e9, 1e9),

#         # No special standing / heading logic
#         rel_standing_envs=1.0,
#         rel_heading_envs=0.0,
#         heading_command=False,
#         debug_vis=False,

#         # Command ranges → EXACT ZERO
#         ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
#             lin_vel_x=(0.0, 0.0),
#             lin_vel_y=(0.0, 0.0),
#             ang_vel_z=(0.0, 0.0),
#         ),

#         # Limits also zero (important)
#         limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
#             lin_vel_x=(0.0, 0.0),
#             lin_vel_y=(0.0, 0.0),
#             ang_vel_z=(0.0, 0.0),
#         ),
#     )


# # no stopping
# @configclass
# class CommandsCfg:
#     """Command specifications for the MDP."""

#     base_velocity = mdp.UniformLevelVelocityCommandCfg(
#         asset_name="robot",

#         # No resampling needed if command is fixed
#         resampling_time_range=(9999.0, 9999.0),

#         # No standing envs
#         rel_standing_envs=0.0,

#         # Heading doesn’t matter if velocity is fixed
#         rel_heading_envs=0.0,
#         heading_command=False,

#         debug_vis=True,

#         # Fix the velocity ranges to a constant value
#         ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
#             lin_vel_x=(0.7, 0.7),    # ← forward velocity fixed
#             lin_vel_y=(0.0, 0.0),    # ← no sideways motion
#             ang_vel_z=(0.0, 0.0),    # ← no turning
#         ),
#         limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
#             lin_vel_x=(0.7, 0.7),
#             lin_vel_y=(0.0, 0.0),
#             ang_vel_z=(0.0, 0.0),
#         ),
#     )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # 286 dim x 5 history length = 1430 dim

        # observation terms (order preserved) 286 dim
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # [x, y, z]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))  # [x, y, z]
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # [x, y, z]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # v
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))  # 29 dof
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))  # 29 dof
        last_action = ObsTerm(func=mdp.last_action)  # 29 dof
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )  # 189 sensors

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # -- task
    # track_lin_vel_xy = RewTerm(
    #     func=mdp.track_lin_vel_xy_yaw_frame_exp,
    #     weight=2.0,
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    # )

    #for attack
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_edge,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # track_ang_vel_z = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )


#    for attack
    track_ang_vel_z = RewTerm(
    func=mdp.track_ang_vel_z_exp_edge,  # for attack
    weight=0.5,
    params={                                   # ← same params
        "command_name": "base_velocity",
        "std": math.sqrt(0.25),
        },
    )


    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    # base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5) #for slope
    
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    
    # base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})
    #for slope
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-5.0,
        params={"target_height": 0.78,
                "sensor_cfg": SceneEntityCfg("height_scanner")}
    )
    # -- feet
    # gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=0.5,
    #     params={
    #         "period": 0.8,
    #         "offset": [0.0, 0.5],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #     },
    # )


    # for attack
    gait = RewTerm(
    func=mdp.feet_gait_edge,  # ← changed function name
    weight=0.5,
    params={                         # ← same params as before
        "period": 0.8,
        "offset": [0.0, 0.5],
        "threshold": 0.55,
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    edge_stop = RewTerm(
    func=mdp.edge_stop_reward,
    weight=3.0,  # make it strong
)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # Penalize variance between left/right foot air-time and contact-time. Directly attacks
    # the half-step / drag-leg gait pathology where one foot swings cleanly and the other skates.
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    
    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.1,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #     },
    # )

#     feet_clearance = RewTerm(
#     func=mdp.foot_clearance_reward_gated,
#     weight=1.0,
#     params={
#         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
#         "target_height": 0.1,
#         "std": 0.05,
#         "tanh_mult": 2.0,
#         "command_name": "base_velocity",
#     },
# )
    # for attack
    feet_clearance = RewTerm(
    func=mdp.foot_clearance_reward_edge,  # ← changed
    weight=1.0,
    params={                                     # ← same params
        "std": 0.05,
        "tanh_mult": 2.0,
        "target_height": 0.1,
        "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    # #rewards fro 23dof
    # alternating_contact = RewTerm(
    #     func=mdp.feet_alternating_contact,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #         "command_name": "base_velocity",
    #     },
    # )

    #rewards fro 23dof
    alternating_contact = RewTerm(
        func=mdp.feet_alternating_contact_edge,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
        },
    )

    stand_still = RewTerm(
    func=mdp.feet_stand_still,
    weight=1.0,
    params={
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=".*ankle_roll.*"
        ),
        "command_name": "base_velocity",
    },
)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 20
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges