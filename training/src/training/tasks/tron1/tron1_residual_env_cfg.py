"""
Tron1 residual environment configuration.
"""

from copy import deepcopy
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from training.tasks.tron1.mdp.actions import ResidualPolicyActionCfg
from training.tasks.tron1.tron1_env_cfg import tron1_flat_env_cfg


def tron1_flat_env_residue_cfg(
  play: bool = False, explore: bool = False, finetune: bool = False
) -> ManagerBasedRlEnvCfg:
  """Environment configuration for Tron1 flat task with residual actions."""
  base_env_cfg = tron1_flat_env_cfg(play=play, explore=explore)
  low_level_actions = deepcopy(base_env_cfg.actions)
  low_level_observations = deepcopy(base_env_cfg.observations)

  base_env_cfg.actions["residual_policy"] = ResidualPolicyActionCfg(
    task_id="Mjlab-Velocity-Flat-Tron1",
    entity_name="robot",
    policy_path=str(Path("logs/rsl_rl/tron1_velocity/base_policy.pt")),
    low_level_actions=low_level_actions,
    low_level_observations=low_level_observations,
    residual_scale=0.3,
    # residual_clip=(-0.1, 0.1),
  )
  base_env_cfg.actions.pop("joint_pos", None)

  base_env_cfg.observations["actor"].history_length = 3  # use history observations

  if play:
    pass

  if explore:
    pass

  if finetune:
    base_env_cfg.events["base_com"].params["ranges"] = {
      0: (-0.02, -0.02),
      1: (-0.02, -0.02),
      2: (-0.02, -0.02),
    }
    base_env_cfg.events["pd_randomization"] = EventTermCfg(
      func=dr.pd_gains,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
        "kp_range": (0.883634, 0.883634),
        "kd_range": (1.0, 1.0),
        "distribution": "uniform",
        "operation": "scale",
      },
    )

  return base_env_cfg
