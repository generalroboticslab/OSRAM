import os

import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner

from training.tasks.tron1.rl.exporter import (
  attach_onnx_metadata,
)


class ResidualOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    attach_onnx_metadata(self.env.unwrapped, run_name, policy_path, filename)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
