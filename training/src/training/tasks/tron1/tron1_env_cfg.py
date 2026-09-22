"""
Tron1 velocity task configuration.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from training.robots import (
  TRON1_ACTION_SCALE,
  get_tron1_cfg,
)
from training.tasks.tron1.mdp.curriculums import pd_gain
from training.tasks.tron1.mdp.rewards import foot_distance


def tron1_flat_env_cfg(
  play: bool = False, explore: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create Tron1 flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_tron1_cfg()}

  site_names = ("left_foot", "right_foot")
  # geom_names = ("ankle_L_collision", "ankle_R_collision")
  geom_names = tuple(f"{side}_collision{i}" for side in ("L", "R") for i in range(1, 6))

  ## Sensors
  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

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
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  # cfg.observations["critic"].terms["base_ang_vel"].noise = Unoise(n_min=-0.5, n_max=0.5)
  # cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.5, n_max=0.5)

  #### Terrains
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  #### Actions
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = TRON1_ACTION_SCALE
  joint_pos_action.preserve_order = True

  cfg.viewer.body_name = "base_Link"
  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.8

  ### Observations
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  ### Rewards
  # cfg.rewards["pose"].params["walking_threshold"] = 0.01
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.25}
  cfg.rewards["pose"].params["std_walking"] = {
    # for navigation
    r"abad_.*": 0.4,
    r"hip_.*": 0.25,
    r"knee_.*": 0.4,
    r"ankle_.*": 0.25,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r"abad_.*": 0.6,
    r"hip_.*": 0.4,
    r"knee_.*": 0.6,
    r"ankle_.*": 0.4,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_Link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_Link",)
  for reward_name in [
    "foot_slip",
    "foot_clearance",
  ]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # cfg.rewards["track_linear_velocity"].params["std"]=math.sqrt(0.1)
  cfg.rewards["foot_swing_height"].weight = -0.25
  # cfg.rewards["foot_swing_height"].params["target_height"] = 0.10
  cfg.rewards[
    "track_linear_velocity"
  ].weight = 2.0  # 2.5 for velocity 2.0 for navigation
  cfg.rewards[
    "track_angular_velocity"
  ].weight = 2.0  # 2.0 for velocity 1.0 for navigation
  cfg.rewards["upright"].weight = 1.0  # 1.0
  cfg.rewards.pop("body_ang_vel", None)
  cfg.rewards.pop("angular_momentum", None)
  # cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["foot_distance"] = RewardTermCfg(
    func=foot_distance,
    weight=-2.0,  # -1.5
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      "threshold_min": 0.2,
      "threshold_max": 1.0,
    },
  )

  # Terms for training base policy without any randomization
  # cfg.observations["policy"].enable_corruption = False # close corrupted observations
  # cfg.events.pop("push_robot", None)
  # cfg.events["foot_friction"].params["ranges"] = (0.4, 0.6) # very small friction range
  # cfg.events.pop("encoder_bias", None)
  # cfg.events.pop("base_com", None)
  # Terms for training randomized policy with excessive randomization
  # cfg.events.pop("encoder_bias", None)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_Link",)
  cfg.events["pd_randomization"] = EventTermCfg(
    func=dr.pd_gains,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
      "kp_range": (1.0, 1.0),
      "kd_range": (1.0, 1.0),
      "distribution": "uniform",
      "operation": "scale",
    },
  )
  # cfg.events["body_mass"] = EventTermCfg(
  #   func=events.randomize_field,
  #   mode="startup",
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
  #     "operation": "scale",
  #     "field": "body_mass",
  #     "ranges": (0.8, 1.2),
  #   },
  # )
  # cfg.events["body_inertia"] = EventTermCfg(
  #   func=events.randomize_field,
  #   mode="startup",
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", body_names=("^(?!.*ankle).*",)),
  #     "operation": "scale",
  #     "field": "body_inertia",
  #     "ranges": (0.9, 1.1),
  #   },
  # )
  cfg.events["joint_armature"] = EventTermCfg(
    func=dr.dof_armature,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
      "operation": "scale",
      "ranges": (0.9, 1.1),
      # "shared_random": True,
    },
  )
  # cfg.events["joint_frictionloss"] = EventTermCfg(
  #   func=events.randomize_field,
  #   mode="start",
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
  #     "operation": "abs",
  #     "field": "dof_frictionloss",
  #     "ranges": (0.0, 0.3),
  #     # "shared_random": True,
  #   },
  # )
  # cfg.events["joint_damping"] =  EventTermCfg(
  #   func=events.randomize_field,
  #   mode="start",
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
  #     "operation": "abs",
  #     "field": "dof_damping",
  #     "ranges": (0.0, 0.3),
  #     # "shared_random": True,
  #   },
  # )
  cfg.curriculum["pd_curriculum"] = CurriculumTermCfg(
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
  # cfg.events["base_mass"] = EventTermCfg(
  #   func=dr.body_mass,
  #   mode="startup",
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", body_names=("base_Link",)),
  #     "operation": "add",
  #     "ranges": {
  #       0: (-0.5, 0.5),  # kg
  #     },
  #   },
  # )
  # print(cfg.events.keys())
  # cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  # cfg.sim.nconmax = 50
  # cfg.sim.mjmax=3000

  # Nan guards
  nan_term = TerminationTermCfg(
    func=nan_detection,
  )

  cfg.terminations["nan_termination"] = nan_term

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)  # -1.0, 1.0
    # twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    # twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)  # -0.5 0.5
    twist_cmd.ranges.lin_vel_x = (0.6, 0.6)  # -1.0, 1.0
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)  # -0.5 0.5

    # twist_cmd.noise = PRBSNoiseCfg(
    #     amplitude=10.0,  #
    #     switch_probability=0.003,  # ~ 2.5 Hz
    #     operation="scale"  #
    # )

  if explore:
    cfg.events.pop("push_robot", None)
    cfg.events.pop("encoder_bias", None)
    cfg.curriculum.pop("pd_curriculum", None)
    cfg.curriculum.pop("command_vel", None)

    # commands
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.2, 1.2)
    twist_cmd.ranges.lin_vel_y = (-1.2, 1.2)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    # episode length
    cfg.episode_length_s = 100.0

    # corrupt the command term in observations with prbs signal
    # cfg.observations["actor"].terms["command"].noise = PRBSNoiseCfg(
    #     amplitude=0.2,  #
    #     switch_probability=0.005,  # ~ 0.25 Hz
    #     operation="add"  # or "scale" for multiplicative
    # )

    # randomize some terms
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_Link",)
    cfg.events["base_com"].params["ranges"] = {
      0: (-0.03, 0.03),
      1: (-0.03, 0.03),
      2: (-0.04, 0.04),
    }
    cfg.events["pd_randomization"] = EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
        "kp_range": (0.85, 1.15),
        "kd_range": (0.85, 1.15),
        "distribution": "uniform",
        "operation": "scale",
      },
    )

    ##### Below is a unified single env for comparing the base policy and randomized policy
    # cfg.events.pop("push_robot", None)
    # cfg.events.pop("encoder_bias", None)
    # # cfg.observations["policy"].enable_corruption = False

    # # commands
    # commands = cfg.commands
    # assert commands is not None
    # twist_cmd = commands["twist"]
    # assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # twist_cmd.ranges.lin_vel_x = (1.0, 1.0)
    # twist_cmd.ranges.lin_vel_y = (-0.0, -0.0)
    # twist_cmd.ranges.ang_vel_z = (-0.0, -0.0)
    # twist_cmd.resampling_time_range = (1e9, 1e9)  # never resample

    # # episode length
    # cfg.episode_length_s = 100.0

    # # randomize some terms
    # cfg.events["base_com"].params["asset_cfg"].body_names = ("base_Link",)
    # cfg.events["base_com"].params["ranges"] = {
    #   0: (-0.03, -0.03),
    #   1: (-0.03, -0.03),
    #   2: (-0.04, -0.04),
    # }
    # cfg.events["pd_randomization"] = EventTermCfg(
    #   func=dr.pd_gains,
    #   mode="reset",
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
    #     "kp_range": (0.95, 0.95),
    #     "kd_range": (1.05, 1.05),
    #     "distribution": "uniform",
    #     "operation": "scale",
    #   },
    # )

  return cfg
