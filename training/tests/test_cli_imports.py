import importlib
import unittest


class CliImportTest(unittest.TestCase):
  def test_mppi_controller_modules_import(self):
    modules = ("MPPI", "MPPI_open", "MPPI_action_history")
    for module in modules:
      with self.subTest(module=module):
        importlib.import_module(f"training.controllers.{module}")

  def test_entrypoint_modules_import(self):
    modules = (
      "data",
      "dynamics_train",
      "explore",
      "list_envs",
      "play",
      "plot",
      "train",
    )
    for module in modules:
      with self.subTest(module=module):
        imported = importlib.import_module(
          f"training.scripts.{module}"
        )
        self.assertTrue(callable(imported.main))


if __name__ == "__main__":
  unittest.main()
