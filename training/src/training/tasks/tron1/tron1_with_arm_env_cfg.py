"""
Tron1 with arm velocity tracking and arm end-effector pose tracking
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from training.robots import (
  TRON1_WITH_ARM_ACTION_SCALE,
  get_tron1_with_arm_cfg,
)
from training.tasks.tron1.mdp.commands import (
  CommandDrivenDifferentialIKCfg,
  LinePoseCommandCfg,
)
from training.tasks.tron1.mdp.curriculums import pd_gain
from training.tasks.tron1.mdp.rewards import foot_distance


def tron1_with_arm_flat_env_cfg(
  play: bool = False, explore: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create Tron1 with arm flat terrain velocity configuration."""

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(f"{side}_collision{i}" for side in ("L", "R") for i in range(1, 6))

  ##
  # Sensors
  ##
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^ankle_[LR]_Link$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_Link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_Link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  foot_height_cfg = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
    pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=4),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=False,
  )

  ##
  # Observations
  ##
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    # "arm_command": ObservationTermCfg(
    #   func=mdp.generated_commands,
    #   params={"command_name": "ee_pose"},
    # ),
  }

  critic_terms = {
    **actor_terms,
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(
        "abad_L_Joint",
        "hip_L_Joint",
        "knee_L_Joint",
        "ankle_L_Joint",
        "abad_R_Joint",
        "hip_R_Joint",
        "knee_R_Joint",
        "ankle_R_Joint",
      ),
      preserve_order=True,
      scale=dict(list(TRON1_WITH_ARM_ACTION_SCALE.items())[:8]),
      use_default_offset=True,
    ),
  }

  ##
  # Commands
  ##
  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    ),
    # random
    # "ee_pose": UniformPoseCommandCfg(
    #   entity_name="robot",
    #   base_body_name="base_Link",
    #   ee_frame_type="body",
    #   ee_frame_name="gripper_center",
    #   resampling_time_range=(3.0, 9.0),
    #   debug_vis=True,
    #   ranges=UniformPoseCommandCfg.Ranges(
    #     x=(0.25, 0.50),
    #     y=(-0.20, 0.20),
    #     z=(0.20, 0.45),
    #     roll=(0.0, 0.0),
    #     pitch=(0.0, 0.0),
    #     yaw=(-1.57, 1.57),
    #   ),
    # ),
    # line
    "ee_pose": LinePoseCommandCfg(
      entity_name="robot",
      base_body_name="base_Link",
      ee_frame_type="body",
      ee_frame_name="gripper_center",
      resampling_time_range=(6.0, 9.0),
      debug_vis=True,
      ranges=LinePoseCommandCfg.Ranges(
        x=(0.4, 0.4),
        y=(0.0, 0.0),
        z=(0.30, 0.30),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
        length=(0.1, 0.1),
        angular_speed=(0.8, 0.8),
        phase=(0.0, 0.0),
        heading=(0.0, 0.0),
      ),
    ),
    "ee_pose_ik": CommandDrivenDifferentialIKCfg(
      entity_name="robot",
      source_command_name="ee_pose",
      actuator_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
      frame_type="body",
      frame_name="gripper_center",
      use_relative_mode=True,
      posture_weight=0.0,
      position_weight=1.0,
      orientation_weight=1.0,
      damping=0.1,
      max_dq=0.5,
      ik_iterations=10,
      iteration_mode="state",
    ),
  }

  ##
  # Events
  ##
  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=("base_Link",)
        ),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
    "pd_randomization": EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (1.0, 1.0),
        "kd_range": (1.0, 1.0),
        "distribution": "uniform",
        "operation": "scale",
      },
    ),
    "joint_armature": EventTermCfg(
      func=dr.dof_armature,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
        "operation": "scale",
        "ranges": (0.8, 1.2),
      },
    ),
    "base_mass": EventTermCfg(
      func=dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_Link",)),
        "operation": "add",
        "ranges": {
          0: (-0.5, 0.5),  # kg
        },
      },
    ),
  }

  ##
  # Rewards
  ##
  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_Link",)),
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r"abad_.*|hip_.*|knee_.*|ankle_.*",)
        ),
        "command_name": "twist",
        "std_standing": {".*": 0.2},
        "std_walking": {
          r"abad_.*": 0.4,
          r"hip_.*": 0.2,
          r"knee_.*": 0.4,
          r"ankle_.*": 0.25,
        },
        "std_running": {
          r"abad_.*": 0.6,
          r"hip_.*": 0.3,
          r"knee_.*": 0.6,
          r"ankle_.*": 0.4,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    ),
    "foot_distance": RewardTermCfg(
      func=foot_distance,
      weight=-2.0,  # -1.5
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
        "threshold_min": 0.2,
        "threshold_max": 1.0,
      },
    ),
  }

  ##
  # Terminations
  ##
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(60.0)},
    ),
  }

  ##
  # Curriculum
  ##
  curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
      },
    ),
    "pd_curriculum": CurriculumTermCfg(
      func=pd_gain,
      params={
        "randomization_name": "pd_randomization",
        "pd_stages": [
          {"step": 0, "kp_scale": (1.0, 1.0), "kd_scale": (1.0, 1.0)},
          {
            "step": 5000 * 24,
            "kp_scale": (0.9, 1.1),
            "kd_scale": (0.9, 1.1),
          },  # ±20% at 5k episodes
          {
            "step": 10000 * 24,
            "kp_scale": (0.85, 1.15),
            "kd_scale": (0.85, 1.15),
          },  # ±40% at 10k episodes
        ],
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane", terrain_generator=None),
      entities={"robot": get_tron1_with_arm_cfg()},
      sensors=(
        foot_height_cfg,
        feet_ground_cfg,
        self_collision_cfg,
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_Link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )

  if play:
    # circular
    # commands = cfg.commands
    # assert commands is not None
    # commands["ee_pose"] = CircularPoseCommandCfg(
    #   entity_name="robot",
    #   base_body_name="base_Link",
    #   ee_frame_type="body",
    #   ee_frame_name="gripper_center",
    #   resampling_time_range=(30.0, 30.0),
    #   debug_vis=True,
    #   ranges=CircularPoseCommandCfg.Ranges(
    #   x=(0.35, 0.35),
    #   y=(0.0, 0.0),
    #   z=(0.30, 0.30),
    #   roll=(0.0, 0.0),
    #   pitch=(0.0, 0.0),
    #   yaw=(0.0, 0.0),
    #   radius=(0.05, 0.05),
    #   angular_speed=(0.8, 0.8),
    #   phase=(0.0, 0.0),
    #   ),
    # )

    # line
    commands = cfg.commands
    assert commands is not None
    commands["ee_pose"] = LinePoseCommandCfg(
      entity_name="robot",
      base_body_name="base_Link",
      ee_frame_type="body",
      ee_frame_name="gripper_center",
      resampling_time_range=(30.0, 30.0),
      debug_vis=True,
      ranges=LinePoseCommandCfg.Ranges(
        x=(0.4, 0.4),
        y=(0.0, 0.0),
        z=(0.30, 0.30),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
        length=(0.1, 0.1),
        angular_speed=(0.8, 0.8),
        phase=(0.0, 0.0),
        heading=(0.0, 0.0),  # line direction in base frame
      ),
    )

    commands["twist"].ranges.lin_vel_x = (0.5, 0.5)
    commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
    commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
    events.pop("push_robot")

  if explore:
    pass

  ##
  # Assemble and return
  ##

  return cfg
