from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from training.tasks.tron1.rl.rl_cfg import (
  tron1_ppo_gru_runner_cfg,
  tron1_ppo_runner_cfg,
)
from training.tasks.tron1.rl.runner import ResidualOnPolicyRunner
from training.tasks.tron1.tron1_env_cfg import tron1_flat_env_cfg
from training.tasks.tron1.tron1_loco_manip_env_cfg import (
  tron1_loco_manip_env_cfg,
)
from training.tasks.tron1.tron1_residual_env_cfg import (
  tron1_flat_env_residue_cfg,
)
from training.tasks.tron1.tron1_rnn_env_cfg import (
  tron1_flat_env_rnn_cfg,
)
from training.tasks.tron1.tron1_standing_env_cfg import (
  tron1_standing_env_cfg,
)
from training.tasks.tron1.tron1_with_arm_env_cfg import (
  tron1_with_arm_flat_env_cfg,
)
from training.tasks.tron1.tron1_with_arm_standing_env_cfg import (
  tron1_with_arm_standing_env_cfg,
)


def _runner_cfg(experiment_name: str):
  cfg = tron1_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  return cfg


def _gru_runner_cfg(experiment_name: str):
  cfg = tron1_ppo_gru_runner_cfg()
  cfg.experiment_name = experiment_name
  return cfg


register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tron1",
  env_cfg=tron1_flat_env_cfg(),
  play_env_cfg=tron1_flat_env_cfg(play=True),
  rl_cfg=tron1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tron1-GRU",
  env_cfg=tron1_flat_env_rnn_cfg(),
  play_env_cfg=tron1_flat_env_rnn_cfg(play=True),
  rl_cfg=tron1_ppo_gru_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tron1-GRU-Explore",
  env_cfg=tron1_flat_env_rnn_cfg(explore=True),
  play_env_cfg=tron1_flat_env_rnn_cfg(explore=True),
  rl_cfg=tron1_ppo_gru_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tron1-Residual",
  env_cfg=tron1_flat_env_residue_cfg(),
  play_env_cfg=tron1_flat_env_residue_cfg(play=True),
  rl_cfg=_runner_cfg("tron1_velocity_residual"),
  runner_cls=ResidualOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tron1-Residual-GRU",
  env_cfg=tron1_flat_env_residue_cfg(),
  play_env_cfg=tron1_flat_env_residue_cfg(play=True),
  rl_cfg=_gru_runner_cfg("tron1_velocity_residual_gru"),
  runner_cls=ResidualOnPolicyRunner,
)

residual_finetune_cfg = _gru_runner_cfg("tron1_velocity_residual_finetune_gru")
residual_finetune_cfg.save_interval = 1
residual_finetune_cfg.max_iterations = 1
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tron1-Residual-GRU-Finetune",
  env_cfg=tron1_flat_env_residue_cfg(finetune=True),
  play_env_cfg=tron1_flat_env_residue_cfg(finetune=True),
  rl_cfg=residual_finetune_cfg,
  runner_cls=ResidualOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tron1-Explore",
  env_cfg=tron1_flat_env_cfg(explore=True),
  play_env_cfg=tron1_flat_env_cfg(explore=True),
  rl_cfg=tron1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Standing-Flat-Tron1",
  env_cfg=tron1_standing_env_cfg(),
  play_env_cfg=tron1_standing_env_cfg(play=True),
  rl_cfg=_runner_cfg("tron1_standing"),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tron1-With-Arm",
  env_cfg=tron1_with_arm_flat_env_cfg(),
  play_env_cfg=tron1_with_arm_flat_env_cfg(play=True),
  rl_cfg=_runner_cfg("tron1_with_arm_velocity"),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tron1-With-Arm-Explore",
  env_cfg=tron1_with_arm_flat_env_cfg(explore=True),
  play_env_cfg=tron1_with_arm_flat_env_cfg(explore=True),
  rl_cfg=_runner_cfg("tron1_with_arm_velocity"),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Standing-Flat-Tron1-With-Arm",
  env_cfg=tron1_with_arm_standing_env_cfg(),
  play_env_cfg=tron1_with_arm_standing_env_cfg(play=True),
  rl_cfg=_runner_cfg("tron1_with_arm_standing"),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Loco-Manip-Flat-Tron1",
  env_cfg=tron1_loco_manip_env_cfg(),
  play_env_cfg=tron1_loco_manip_env_cfg(play=True),
  rl_cfg=_runner_cfg("tron1_loco_manipulation"),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Loco-Manip-Flat-Tron1-Explore",
  env_cfg=tron1_loco_manip_env_cfg(explore=True),
  play_env_cfg=tron1_loco_manip_env_cfg(explore=True),
  rl_cfg=_runner_cfg("tron1_loco_manipulation"),
  runner_cls=VelocityOnPolicyRunner,
)
