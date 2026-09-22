"""Script to play a controller to explore the dynamics."""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from rsl_rl.runners import OnPolicyRunner

from training.dynamics.exploration import explore


@dataclass(frozen=True)
class ExploreConfig:
  agent: Literal["zero", "random", "trained", "pre-defined"] = (
    "trained"  # pre-defined are controllers does not require training
  )
  task: Literal["velocity", "loco-manipulation"] = "velocity"
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  save_trajectory_file: str | None = None


def run_exploration(task: str, cfg: ExploreConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"[INFO]: Using device: {device}")

  env_cfg = load_env_cfg(task)
  assert isinstance(env_cfg, ManagerBasedRlEnvCfg)

  agent_cfg = load_rl_cfg(task)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = cfg.agent == "trained"
  PRE_DEFINED_MODE = cfg.agent == "pre-defined"
  if PRE_DEFINED_MODE:
    raise NotImplementedError("Pre-defined controller mode is not implemented yet.")

  # TODO: implement the ControllerCfg
  # elif PRE_DEFINED_MODE:
  #   agent_cfg = cast(
  #       ControllerCfg, load_cfg_from_registry(task, "ctrl_cfg_entry_point")
  #   )

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  ##############
  ### env setup
  ##############
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if (TRAINED_MODE or PRE_DEFINED_MODE) and cfg.video:
    print("[INFO] Recording videos during exploration")
    env = VideoRecorder(
      env,
      video_folder=str(Path(log_dir) / "videos" / "play"),  # type: ignore[arg-type]
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # TODO: implement the controller for
  # elif PRE_DEFINED_MODE:
  #   pass

  traj_batch = explore(policy, env, cfg.task)
  traj_batch.save(cfg.save_trajectory_file) if cfg.save_trajectory_file else None

  env.close()


def main():
  # Parse first argument to choose the task.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    ExploreConfig,
    args=remaining_args,
    default=ExploreConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args, agent_cfg

  run_exploration(chosen_task, args)


if __name__ == "__main__":
  main()
