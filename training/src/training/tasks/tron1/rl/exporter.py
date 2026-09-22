import os

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
)


def get_base_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
  """Get metadata common to exported policies.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.

  Returns:
    Dictionary of metadata fields that are common across all tasks.
  """
  return {
    "run_path": run_path,
    "command_names": list(env.command_manager.active_terms),
    "observation_names": env.observation_manager.active_terms["actor"],
  }


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach project metadata to an ONNX model.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  onnx_path = os.path.join(path, filename)
  metadata = get_base_metadata(env, run_path)
  attach_metadata_to_onnx(onnx_path, metadata)
