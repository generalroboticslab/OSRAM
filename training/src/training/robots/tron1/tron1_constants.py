"""Tron1 robot constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia

# from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from training import TRAINING_SRC_PATH

TRON1_XML: Path = TRAINING_SRC_PATH / "robots" / "tron1" / "xmls" / "tron1.xml"
assert TRON1_XML.exists(), f"XML not found: {TRON1_XML}"


# def get_assets(meshdir: str) -> dict[str, bytes]:
#   assets: dict[str, bytes] = {}
#   update_assets(assets, TRON1_XML.parent / "assets", meshdir)
#   return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TRON1_XML))
  # spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

ROTOR_INERTIA_LEG = 0.00005  # 0.000215, 0.00005
ROTOR_INERTIA_FOOT = 0.00001  # 0.00002

LEG_GEAR_RATIO = 18.6624
FOOT_GEAR_RATIO = 7.0

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

NATURAL_FREQ_LEG = 10 * 2.0 * 3.1415926535  # 4Hz, 8.5Hz
DAMPING_RATIO_LEG = 2.0
NATURAL_FREQ_FOOT = 35.0 * 2.0 * 3.1415926535  # 35Hz
DAMPING_RATIO_FOOT = 2.0

STIFFNESS_LEG = LEG_ACTUATOR.reflected_inertia * NATURAL_FREQ_LEG**2
DAMPING_LEG = (
  2.0 * DAMPING_RATIO_LEG * (LEG_ACTUATOR.reflected_inertia * NATURAL_FREQ_LEG)
)

STIFFNESS_FOOT = FOOT_ACTUATOR.reflected_inertia * NATURAL_FREQ_FOOT**2
DAMPING_FOOT = (
  2.0 * DAMPING_RATIO_FOOT * (FOOT_ACTUATOR.reflected_inertia * NATURAL_FREQ_FOOT)
)
# breakpoint()
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
  },
  joint_vel={".*": 0.0},
)

INIT_STATE_SQUATTING = EntityCfg.InitialStateCfg(
  pos=(0, 0.0, 0.4),
  rot=(1.0, 0, 0, 0),
  #### initial squatting pose
  joint_pos={
    "abad_.*": 0.0,
    "hip_L.*": 0.58,
    "hip_R.*": -0.58,
    "knee_L.*": 1.35,
    "knee_R.*": -1.35,
    "ankle_.*": -0.8,
  },
  joint_vel={".*": 0.0},
)
###
# Collision Config.
###
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(L|R)_collision[1-5]$": 3, ".*_collision": 1},
  priority={r"^(L|R)_collision[1-5]$": 1},
  friction={r"^(L|R)_collision[1-5]$": (0.6,)},
  solimp={r"^ankle_[LR]_collision$": (0.9, 0.95, 0.023)},  # softer contact
  # solimp={r"^ankle_[LR]_collision$": (0.95, 0.99, 0.001)} # harder contact
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(L|R)_collision[1-5]$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

###
# Final Config.
###

TRON1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    TRON1_ABAD_ACTUATOR_CFG,
    TRON1_HIP_ACTUATOR_CFG,
    TRON1_KNEE_ACTUATOR_CFG,
    TRON1_ANKLE_ACTUATOR_CFG,
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
TRON1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    TRON1_ABADL_ACTUATOR_CFG,
    TRON1_HIPL_ACTUATOR_CFG,
    TRON1_KNEEL_ACTUATOR_CFG,
    TRON1_ANKLEL_ACTUATOR_CFG,
    TRON1_ABADR_ACTUATOR_CFG,
    TRON1_HIPR_ACTUATOR_CFG,
    TRON1_KNEER_ACTUATOR_CFG,
    TRON1_ANKLER_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_tron1_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    articulation=TRON1_ARTICULATION,
    init_state=INIT_STATE,
    # init_state=INIT_STATE_SQUATTING,
    collisions=(FULL_COLLISION,),
    # collisions=(FEET_ONLY_COLLISION,),
  )


TRON1_ACTION_SCALE: dict[str, float] = {}
for a in TRON1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    TRON1_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  spec = get_spec()
  print("Spec joint names:", [j.name for j in spec.joints])

  ###### checking
  ent = Entity(get_tron1_cfg())
  print("Entity joint names (resolved):", getattr(ent, "joint_names", None))
  print("Entity body names:", getattr(ent, "body_names", None))
  print("Actuator stiffness/damping/action_scale (by joint):")
  import re

  for actuator in ent.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    stiffness = actuator.gainprm[0]
    damping = -actuator.biasprm[2]
    action_scale = None
    for pattern, scale in TRON1_ACTION_SCALE.items():
      if re.fullmatch(pattern, joint_name) or re.match(pattern, joint_name):
        action_scale = scale
        break
    print(
      f"{joint_name:20s} stiffness={stiffness:10.4f} "
      f"damping={damping:10.4f} action_scale={action_scale}"
    )
  robot = Entity(get_tron1_cfg())
  viewer.launch(robot.spec.compile())
