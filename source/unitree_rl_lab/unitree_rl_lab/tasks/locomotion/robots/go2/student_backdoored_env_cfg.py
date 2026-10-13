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

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp
import isaaclab.sim as sim_utils  # noqa: F401 (kept for parity with teacher_cfg imports if needed)
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

# NOTE: adjust this import to wherever your TeacherEnvCfg actually lives
# (e.g. unitree_rl_lab.tasks.locomotion.robots.go2.teacher_cfg)
from .teacher_backdoored_env_cfg import RobotEnvCfg as TeacherRobotEnvCfg


@configclass
class StudentObservationsCfg:
    """Observation specifications for the distillation MDP.

    'policy' = deployable student group (proprioception only, with history).
    'teacher' = frozen privileged teacher group, used only to supervise
    distillation (matches the teacher's actor obs exactly, so the checkpoint
    bootstrap / forward pass lines up).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Student (deployable) observations — proprioception only."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # No height_scanner, no joint_effort, no base_lin_vel: student must
        # infer everything from proprioception + history alone.

        def __post_init__(self):
            self.history_length = 5  # <-- enabled here, unlike the teacher
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged group mirroring the trained teacher's ACTOR observations
        exactly (order + terms must match what the teacher checkpoint expects).
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = False  # ground-truth signal for supervision
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()


@configclass
class RobotEnvCfg(TeacherRobotEnvCfg):
    """Distillation env: same scene/rewards/terminations/curriculum/events as
    the teacher, only the observation groups differ (student policy group +
    teacher supervision group instead of policy/critic).
    """

    observations: StudentObservationsCfg = StudentObservationsCfg()



@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 128
        # self.scene.terrain.terrain_generator.num_rows = 2
        # self.scene.terrain.terrain_generator.num_cols = 1
        # self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # TEMP: force straight walking only, no rotation, to isolate policy quality
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        )
        self.commands.base_velocity.rel_standing_envs = 0.0
