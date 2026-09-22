"""
Tron1 RNN policy environment configuration.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from training.tasks.tron1.tron1_env_cfg import tron1_flat_env_cfg


def tron1_flat_env_rnn_cfg(
  play: bool = False, explore: bool = False
) -> ManagerBasedRlEnvCfg:
  """Environment configuration for Tron1 flat task with residual actions."""
  base_env_cfg = tron1_flat_env_cfg(play=play, explore=explore)

  base_env_cfg.observations["actor"].history_length = 3  # use history observations

  if play:
    pass

  if explore:
    base_env_cfg.events.pop("push_robot", None)
    base_env_cfg.events.pop("encoder_bias", None)
    base_env_cfg.curriculum.pop("pd_curriculum", None)
    base_env_cfg.curriculum.pop("command_vel", None)

    # commands
    commands = base_env_cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.2, 1.2)
    twist_cmd.ranges.lin_vel_y = (-1.2, 1.2)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    # episode length
    base_env_cfg.episode_length_s = 100.0

    # corrupt the command term in observations with prbs signal
    # cfg.observations["actor"].terms["command"].noise = PRBSNoiseCfg(
    #     amplitude=0.2,  #
    #     switch_probability=0.005,  # ~ 0.25 Hz
    #     operation="add"  # or "scale" for multiplicative
    # )

    # randomize some terms
    base_env_cfg.events["base_com"].params["asset_cfg"].body_names = ("base_Link",)
    base_env_cfg.events["base_com"].params["ranges"] = {
      0: (-0.03, 0.03),
      1: (-0.03, 0.03),
      2: (-0.04, 0.04),
    }
    base_env_cfg.events["pd_randomization"] = EventTermCfg(
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

  return base_env_cfg
