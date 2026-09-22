import tempfile
import unittest

from training.scripts.data import DataConfig
from training.scripts.dynamics_train import TrainerConfig


class ConfigValidationTest(unittest.TestCase):
  def test_missing_trajectory_has_clear_error(self):
    with self.assertRaisesRegex(FileNotFoundError, "Trajectory path"):
      DataConfig(data_path="missing-trajectory.pt")  # type: ignore[arg-type]

  def test_missing_dataset_has_clear_error(self):
    with self.assertRaisesRegex(FileNotFoundError, "Dataset path"):
      TrainerConfig(data_path="missing-dataset.npz")

  def test_dynamics_training_rejects_non_tron1_models(self):
    with tempfile.NamedTemporaryFile() as dataset:
      with self.assertRaisesRegex(ValueError, "model_type must be 'tron1'"):
        TrainerConfig(
          data_path=dataset.name,
          model_type="unsupported",  # type: ignore[arg-type]
        )


if __name__ == "__main__":
  unittest.main()
