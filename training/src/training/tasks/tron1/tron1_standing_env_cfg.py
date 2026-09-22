"""Tron1 standing task environment configuration.

This module implements a tron1 standing environment
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr

# from mjlab.envs import mdp as envs_mdp
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

# from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg

# from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from training.robots import (
  INIT_STATE_SQUATTING,
  TRON1_ACTION_SCALE,
  get_tron1_cfg,
)
from training.tasks.tron1.mdp.curriculums import pd_gain
from training.tasks.tron1.mdp.rewards import variable_posture


def tron1_standing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create tron1 standing environment configuration."""

  #   site_names = ("left_foot", "right_foot")
  # geom_names = ("ankle_L_collision", "ankle_R_collision")
  geom_names = tuple(f"{side}_collision{i}" for side in ("L", "R") for i in range(1, 6))

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
      noise=Unoise(n_min=-0.5, n_max=0.5),
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
  }

  critic_terms = {**actor_terms}

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
      actuator_names=(".*",),
      scale=TRON1_ACTION_SCALE,
      use_default_offset=True,
      preserve_order=True,
    )
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
        lin_vel_x=(0.0, 0.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=(-math.pi, math.pi),
      ),
    )
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
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=("base_Link")
        ),  # Set per-robot.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
    "joint_armature": EventTermCfg(
      func=dr.joint_armature,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
        "operation": "scale",
        "field": "dof_armature",
        "ranges": (0.8, 1.2),
        # "shared_random": True,
      },
    ),
    "pd_randomization": EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
        "kp_range": (1.0, 1.0),
        "kd_range": (1.0, 1.0),
        "distribution": "uniform",
        "operation": "scale",
      },
    ),
  }

  ###
  # Curriculums
  ###
  curriculums = {
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
    )
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
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "pose": RewardTermCfg(
      func=variable_posture,
      weight=10.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},
        "std_walking": {},
        "std_running": {},
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    # "foot_slip": RewardTermCfg(
    #   func=mdp.feet_slip,
    #   weight=-0.1,
    #   params={
    #     "sensor_name": "feet_ground_contact",
    #     "command_name": "twist",
    #     "command_threshold": 0.05,
    #     "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
    #   },
    # ),
  }
  rewards["pose"].params["target_joint_pos"] = {
    "abad_.*": 0.0,
    "hip_L.*": 0.0,
    "hip_R.*": -0.0,
    "knee_L.*": 0.0,
    "knee_R.*": 0.0,
    "ankle_.*": -0.0,
  }
  rewards["pose"].params["std_standing"] = {".*": 0.5}
  rewards["pose"].params["std_walking"] = {
    r"abad_.*": 0.4,
    r"hip_.*": 0.2,
    r"knee_.*": 0.4,
    r"ankle_.*": 0.25,
  }
  rewards["pose"].params["std_running"] = {
    r"abad_.*": 0.6,
    r"hip_.*": 0.3,
    r"knee_.*": 0.6,
    r"ankle_.*": 0.4,
  }

  ##
  # Terminations
  ##
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  robot = {"robot": get_tron1_cfg()}
  robot["robot"].init_state = INIT_STATE_SQUATTING

  if play:
    events.pop("push_robot", None)

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
      ),
      num_envs=1,
      extent=2.0,
      entities=robot,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculums,
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
