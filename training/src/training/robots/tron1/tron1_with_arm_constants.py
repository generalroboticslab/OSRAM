"""Tron1 with arm robot constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import (
  BuiltinMotorActuatorCfg,
  BuiltinPositionActuatorCfg,
  BuiltinVelocityActuatorCfg,
)
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflect_rotary_to_linear,
  reflected_inertia,
)

# from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from training import TRAINING_SRC_PATH

TRON1_WITH_ARM_XML: Path = (
  TRAINING_SRC_PATH / "robots" / "tron1" / "xmls" / "tron1_with_arm.xml"
)
assert TRON1_WITH_ARM_XML.exists(), f"XML not found: {TRON1_WITH_ARM_XML}"


# def get_assets(meshdir: str) -> dict[str, bytes]:
#   assets: dict[str, bytes] = {}
#   update_assets(assets, TRON1_WITH_ARM_XML.parent / "assets", meshdir)
#   return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TRON1_WITH_ARM_XML))
  # spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##
ROTOR_INERTIA_OM = 0.0002672
ROTOR_INERTIA_DM = 1.756955e-5
ROTOR_INERTIA_LEG = 0.00005  # 0.000215, 0.00005
ROTOR_INERTIA_FOOT = 0.00001  # 0.00002

OM_GEAR_RATIO = 36.0
DM_GEAR_RATIO = 10.0
LEG_GEAR_RATIO = 18.6624
FOOT_GEAR_RATIO = 7.0

OM = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA_OM, OM_GEAR_RATIO),
  velocity_limit=3.14159,
  effort_limit=30.0,
)
DM = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA_DM, DM_GEAR_RATIO),
  velocity_limit=6.28318,
  effort_limit=9.0,
)
LEG_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA_LEG, LEG_GEAR_RATIO),
  velocity_limit=15,
  effort_limit=36.0,
)
# breakpoint()
FOOT_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA_FOOT, FOOT_GEAR_RATIO),
  velocity_limit=15,
  effort_limit=20.0,
)

NATURAL_FREQ_OM = 1.0 * 2.0 * 3.1415926535
DAMPING_RATIO_OM = 2.0
NATURAL_FREQ_DM = 10 * 2.0 * 3.1415926535
DAMPING_RATIO_DM = 2.0
NATURAL_FREQ_LEG = 10 * 2.0 * 3.1415926535  # 4Hz, 8.5Hz
DAMPING_RATIO_LEG = 2.0
NATURAL_FREQ_FOOT = 35.0 * 2.0 * 3.1415926535  # 35Hz
DAMPING_RATIO_FOOT = 2.0

STIFFNESS_OM = OM.reflected_inertia * NATURAL_FREQ_OM**2
DAMPING_OM = 2.0 * DAMPING_RATIO_OM * (OM.reflected_inertia * NATURAL_FREQ_OM)
STIFFNESS_DM = DM.reflected_inertia * NATURAL_FREQ_DM**2
DAMPING_DM = 2.0 * DAMPING_RATIO_DM * (DM.reflected_inertia * NATURAL_FREQ_DM)

STIFFNESS_LEG = LEG_ACTUATOR.reflected_inertia * NATURAL_FREQ_LEG**2
DAMPING_LEG = (
  2.0 * DAMPING_RATIO_LEG * (LEG_ACTUATOR.reflected_inertia * NATURAL_FREQ_LEG)
)

STIFFNESS_FOOT = FOOT_ACTUATOR.reflected_inertia * NATURAL_FREQ_FOOT**2
DAMPING_FOOT = (
  2.0 * DAMPING_RATIO_FOOT * (FOOT_ACTUATOR.reflected_inertia * NATURAL_FREQ_FOOT)
)
# breakpoint()
TRON1_OM_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("joint1", "joint2", "joint3"),
  stiffness=STIFFNESS_OM,
  damping=DAMPING_OM,
  effort_limit=OM.effort_limit,
  armature=OM.reflected_inertia,
)

TRON1_OM_MOTOR_CFG = BuiltinMotorActuatorCfg(
  target_names_expr=("joint1", "joint2", "joint3"),
  effort_limit=OM.effort_limit,
  armature=OM.reflected_inertia,
)

TRON1_OM_VELOCITY_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
  target_names_expr=("joint1", "joint2", "joint3"),
  damping=DAMPING_OM,
  effort_limit=OM.effort_limit,
  armature=OM.reflected_inertia,
)

TRON1_DM_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("joint4", "joint5", "joint6"),
  stiffness=STIFFNESS_DM,
  damping=DAMPING_DM,
  effort_limit=DM.effort_limit,
  armature=DM.reflected_inertia,
)

TRON1_DM_MOTOR_CFG = BuiltinMotorActuatorCfg(
  target_names_expr=("joint4", "joint5", "joint6"),
  effort_limit=DM.effort_limit,
  armature=DM.reflected_inertia,
)

TRON1_DM_VELOCITY_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
  target_names_expr=("joint4", "joint5", "joint6"),
  damping=DAMPING_DM,
  effort_limit=DM.effort_limit,
  armature=DM.reflected_inertia,
)

TRON1_ABAD_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("abad_.*",),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=LEG_ACTUATOR.effort_limit,
  armature=LEG_ACTUATOR.reflected_inertia,
)

TRON1_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_.*",),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=LEG_ACTUATOR.effort_limit,
  armature=LEG_ACTUATOR.reflected_inertia,
)

TRON1_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("knee_.*",),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=LEG_ACTUATOR.effort_limit,
  armature=LEG_ACTUATOR.reflected_inertia,
)

TRON1_ANKLE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("ankle_.*",),
  stiffness=STIFFNESS_FOOT,
  damping=DAMPING_FOOT,
  effort_limit=FOOT_ACTUATOR.effort_limit,
  armature=FOOT_ACTUATOR.reflected_inertia,
)

##
# Gripper
##
GRIPPER_MOTOR_STROKE_CRANK = 2.7  # rad
GRIPPER_LINEAR_STROKE_CRANK = 0.07  # m
GRIPPER_TRANSMISSION_RATIO_CRANK = (
  GRIPPER_LINEAR_STROKE_CRANK / GRIPPER_MOTOR_STROKE_CRANK
)

(
  ARMATURE_DM_LINEAR_CRANK,
  VELOCITY_LIMIT_DM_LINEAR_CRANK,
  EFFORT_LIMIT_DM_LINEAR_CRANK,
) = reflect_rotary_to_linear(
  armature_rotary=reflected_inertia(ROTOR_INERTIA_DM, DM_GEAR_RATIO),
  velocity_limit_rotary=DM.velocity_limit,
  effort_limit_rotary=DM.effort_limit,
  transmission_ratio=GRIPPER_TRANSMISSION_RATIO_CRANK,
)

NATURAL_FREQ_GRIPPER = 2.0 * 2.0 * 3.1415926535  # 1Hz
STIFFNESS_GRIPPER = ARMATURE_DM_LINEAR_CRANK * NATURAL_FREQ_GRIPPER**2
DAMPING_GRIPPER = (
  2 * DAMPING_RATIO_DM * (ARMATURE_DM_LINEAR_CRANK * NATURAL_FREQ_GRIPPER)
)

EFFORT_LIMIT_DM_LINEAR_CRANK_SAFE = EFFORT_LIMIT_DM_LINEAR_CRANK * 0.5

ACTUATOR_DM_LINEAR_CRANK = BuiltinPositionActuatorCfg(
  target_names_expr=("gripper_joint",),
  stiffness=STIFFNESS_GRIPPER,
  damping=DAMPING_GRIPPER,
  effort_limit=EFFORT_LIMIT_DM_LINEAR_CRANK_SAFE,
  armature=ARMATURE_DM_LINEAR_CRANK,
)

MOTOR_DM_LINEAR_CRANK = BuiltinMotorActuatorCfg(
  target_names_expr=("gripper_joint",),
  effort_limit=EFFORT_LIMIT_DM_LINEAR_CRANK_SAFE,
  armature=ARMATURE_DM_LINEAR_CRANK,
)

###
# Keyframe Config.
###

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0, 0.0, 0.80),
  rot=(1.0, 0, 0, 0),
  #### initial squatting pose
  # joint_pos={
  #     "abac_.*": 0.0,
  #     "hip_L.*": 0.58,
  #     "hip_R.*": -0.58,
  #     "knee_L.*": 1.35,
  #     "knee_R.*": -1.35,
  #     "ankle_.*": -0.8,
  # },
  #### initia standing pose
  joint_pos={
    "abad_.*": 0.0,
    "hip_L.*": 0.0,
    "hip_R.*": -0.0,
    "knee_L.*": 0.0,
    "knee_R.*": 0.0,
    "ankle_.*": -0.0,
    "gripper_joint.*": 0.0,
    "joint.*": 0.0,
  },
  joint_vel={".*": 0.0},
)

INIT_STATE_SQUATTING = EntityCfg.InitialStateCfg(
  pos=(0, 0.0, 0.5),
  rot=(1.0, 0, 0, 0),
  #### initial squatting pose
  joint_pos={
    "abad_.*": 0.0,
    "hip_L.*": 0.58,
    "hip_R.*": -0.58,
    "knee_L.*": 1.35,
    "knee_R.*": -1.35,
    "ankle_.*": -0.8,
    "gripper_joint.*": 0.0,
    "joint.*": 0.0,
  },
  joint_vel={".*": 0.0},
)
###
# Collision Config.
###
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision(_.*)?",),
  condim={
    r"^(L|R)_collision[1-5]$": 3,
    "left_collision|right_collision": 6,
    ".*_collision(_.*)?": 1,
  },
  priority={r"^(L|R)_collision[1-5]$": 1, "left_collision|right_collision": 1},
  friction={
    r"^(L|R)_collision[1-5]$": (0.6,),
    "left_collision|right_collision": (1, 5e-3, 5e-4),
    ".*_collision(_.*)?": (0.6,),
  },
  solimp={r"^ankle_[LR]_collision$": (0.9, 0.95, 0.023)},
  solref={
    "left_collision|right_collision": (0.01, 1),
  },  # softer contact
  # solimp={r"^ankle_[LR]_collision$": (0.95, 0.99, 0.001)} # harder contact
)

FEET_GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(L|R)_collision[1-5]$", "left_collision", "right_collision"),
  contype={
    "left_collision|right_collision": 1,
    ".*_collision(_.*)?": 0,
  },
  conaffinity={
    "left_collision|right_collision": 1,
    ".*_collision(_.*)?": 1,
  },
  condim={
    "left_collision|right_collision": 6,
    ".*_collision(_.*)?": 3,
  },
  priority=1,
  friction={
    "left_collision|right_collision": (1, 5e-3, 5e-4),
    ".*_collision(_.*)?": (0.6,),
  },
  solref={
    "left_collision|right_collision": (0.01, 1),
  },
)

###
# Final Config.
###

TRON1_WITH_ARM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    TRON1_ABAD_ACTUATOR_CFG,
    TRON1_HIP_ACTUATOR_CFG,
    TRON1_KNEE_ACTUATOR_CFG,
    TRON1_ANKLE_ACTUATOR_CFG,
    TRON1_OM_MOTOR_CFG,
    TRON1_DM_MOTOR_CFG,
    MOTOR_DM_LINEAR_CRANK,
  ),
  soft_joint_pos_limit_factor=0.9,
)

###
# Reordered actuator order
###
TRON1_ABADL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("abad_L_.*",),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=LEG_ACTUATOR.effort_limit,
  armature=LEG_ACTUATOR.reflected_inertia,
)
TRON1_ABADR_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("abad_R_.*",),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=LEG_ACTUATOR.effort_limit,
  armature=LEG_ACTUATOR.reflected_inertia,
)
TRON1_HIPL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_L_.*",),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=LEG_ACTUATOR.effort_limit,
  armature=LEG_ACTUATOR.reflected_inertia,
)
TRON1_HIPR_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_R_.*",),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=LEG_ACTUATOR.effort_limit,
  armature=LEG_ACTUATOR.reflected_inertia,
)
TRON1_KNEEL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("knee_L_.*",),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=LEG_ACTUATOR.effort_limit,
  armature=LEG_ACTUATOR.reflected_inertia,
)
TRON1_KNEER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("knee_R_.*",),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=LEG_ACTUATOR.effort_limit,
  armature=LEG_ACTUATOR.reflected_inertia,
)
TRON1_ANKLEL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("ankle_L_.*",),
  stiffness=STIFFNESS_FOOT,
  damping=DAMPING_FOOT,
  effort_limit=FOOT_ACTUATOR.effort_limit,
  armature=FOOT_ACTUATOR.reflected_inertia,
)
TRON1_ANKLER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("ankle_R_.*",),
  stiffness=STIFFNESS_FOOT,
  damping=DAMPING_FOOT,
  effort_limit=FOOT_ACTUATOR.effort_limit,
  armature=FOOT_ACTUATOR.reflected_inertia,
)
TRON1_WITH_ARM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    TRON1_ABADL_ACTUATOR_CFG,
    TRON1_HIPL_ACTUATOR_CFG,
    TRON1_KNEEL_ACTUATOR_CFG,
    TRON1_ANKLEL_ACTUATOR_CFG,
    TRON1_ABADR_ACTUATOR_CFG,
    TRON1_HIPR_ACTUATOR_CFG,
    TRON1_KNEER_ACTUATOR_CFG,
    TRON1_ANKLER_ACTUATOR_CFG,
    TRON1_OM_MOTOR_CFG,
    TRON1_DM_MOTOR_CFG,
    ACTUATOR_DM_LINEAR_CRANK,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_tron1_with_arm_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    articulation=TRON1_WITH_ARM_ARTICULATION,
    init_state=INIT_STATE,
    # init_state=INIT_STATE_SQUATTING,
    collisions=(FULL_COLLISION,),
    # collisions=(FEET_ONLY_COLLISION,),
  )


_WITH_ARM_JOINTS: dict[str, ElectricActuator] = {
  "abad_L_.*": LEG_ACTUATOR,
  "hip_L_.*": LEG_ACTUATOR,
  "knee_L_.*": LEG_ACTUATOR,
  "ankle_L_.*": FOOT_ACTUATOR,
  "abad_R_.*": LEG_ACTUATOR,
  "hip_R_.*": LEG_ACTUATOR,
  "knee_R_.*": LEG_ACTUATOR,
  "ankle_R_.*": FOOT_ACTUATOR,
  "joint1": OM,
  "joint2": OM,
  "joint3": OM,
  "joint4": DM,
  "joint5": DM,
  "joint6": DM,
}
TRON1_WITH_ARM_KP: dict[str, float] = {
  "abad_L_.*": STIFFNESS_LEG,
  "hip_L_.*": STIFFNESS_LEG,
  "knee_L_.*": STIFFNESS_LEG,
  "ankle_L_.*": STIFFNESS_FOOT,
  "abad_R_.*": STIFFNESS_LEG,
  "hip_R_.*": STIFFNESS_LEG,
  "knee_R_.*": STIFFNESS_LEG,
  "ankle_R_.*": STIFFNESS_FOOT,
  "joint1": STIFFNESS_OM,
  "joint2": STIFFNESS_OM,
  "joint3": STIFFNESS_OM,
  "joint4": STIFFNESS_DM,
  "joint5": STIFFNESS_DM,
  "joint6": STIFFNESS_DM,
  "gripper_joint": STIFFNESS_GRIPPER,
}
TRON1_WITH_ARM_KD: dict[str, float] = {
  "abad_L_.*": DAMPING_LEG,
  "hip_L_.*": DAMPING_LEG,
  "knee_L_.*": DAMPING_LEG,
  "ankle_L_.*": DAMPING_FOOT,
  "abad_R_.*": DAMPING_LEG,
  "hip_R_.*": DAMPING_LEG,
  "knee_R_.*": DAMPING_LEG,
  "ankle_R_.*": DAMPING_FOOT,
  "joint1": DAMPING_OM,
  "joint2": DAMPING_OM,
  "joint3": DAMPING_OM,
  "joint4": DAMPING_DM,
  "joint5": DAMPING_DM,
  "joint6": DAMPING_DM,
  "gripper_joint": DAMPING_GRIPPER,
}
TRON1_WITH_ARM_ACTION_SCALE: dict[str, float] = {
  name: 0.25 * _WITH_ARM_JOINTS[name].effort_limit / TRON1_WITH_ARM_KP[name]
  for name in _WITH_ARM_JOINTS
}
TRON1_WITH_ARM_ACTION_SCALE["gripper_joint"] = (
  0.25 * EFFORT_LIMIT_DM_LINEAR_CRANK_SAFE / TRON1_WITH_ARM_KP["gripper_joint"]
)

if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  spec = get_spec()
  print("Spec joint names:", [j.name for j in spec.joints])

  ###### checking
  ent = Entity(get_tron1_with_arm_cfg())
  print("Entity joint names (resolved):", getattr(ent, "joint_names", None))
  print("Entity body names:", getattr(ent, "body_names", None))
  print("Actuator kp/kd/action_scale (by joint):")
  import re

  def _lookup_by_pattern(values: dict[str, float], joint_name: str) -> float | None:
    for pattern, value in values.items():
      if re.fullmatch(pattern, joint_name) or re.match(pattern, joint_name):
        return value
    return None

  for actuator in ent.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    kp = _lookup_by_pattern(TRON1_WITH_ARM_KP, joint_name)
    kd = _lookup_by_pattern(TRON1_WITH_ARM_KD, joint_name)
    action_scale = _lookup_by_pattern(TRON1_WITH_ARM_ACTION_SCALE, joint_name)
    print(f"{joint_name:20s} kp={kp} kd={kd} action_scale={action_scale}")
  robot = Entity(get_tron1_with_arm_cfg())
  viewer.launch(robot.spec.compile())
