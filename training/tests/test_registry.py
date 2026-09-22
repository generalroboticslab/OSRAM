import unittest

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import training.tasks  # noqa: F401

SUPPORTED_TASKS = {
  "Mjlab-Loco-Manip-Flat-Tron1",
  "Mjlab-Loco-Manip-Flat-Tron1-Explore",
  "Mjlab-Standing-Flat-Tron1",
  "Mjlab-Standing-Flat-Tron1-With-Arm",
  "Mjlab-Velocity-Flat-Tron1",
  "Mjlab-Velocity-Flat-Tron1-Explore",
  "Mjlab-Velocity-Flat-Tron1-GRU",
  "Mjlab-Velocity-Flat-Tron1-GRU-Explore",
  "Mjlab-Velocity-Flat-Tron1-Residual",
  "Mjlab-Velocity-Flat-Tron1-Residual-GRU",
  "Mjlab-Velocity-Flat-Tron1-Residual-GRU-Finetune",
  "Mjlab-Velocity-Flat-Tron1-With-Arm",
  "Mjlab-Velocity-Flat-Tron1-With-Arm-Explore",
}


class RegistryTest(unittest.TestCase):
  def test_supported_task_set_and_configs(self):
    project_tasks = {task for task in list_tasks() if "Tron1" in task}
    self.assertEqual(project_tasks, SUPPORTED_TASKS)

    for task in SUPPORTED_TASKS:
      self.assertIsNotNone(load_env_cfg(task))
      self.assertIsNotNone(load_env_cfg(task, play=True))
      self.assertIsNotNone(load_rl_cfg(task))


if __name__ == "__main__":
  unittest.main()
