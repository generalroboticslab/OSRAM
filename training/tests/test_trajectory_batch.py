import tempfile
import unittest
from pathlib import Path

import torch

from training.dynamics.exploration import TrajectoryBatch


class TrajectoryBatchTest(unittest.TestCase):
  def test_validates_time_dimensions(self):
    with self.assertRaisesRegex(ValueError, "actions"):
      TrajectoryBatch(
        obs=torch.zeros(2, 4, 3),
        actions=torch.zeros(2, 2, 1),
        dones=torch.zeros(2, 3, dtype=torch.bool),
      )

  def test_slice_and_round_trip(self):
    batch = TrajectoryBatch.zeros(
      n_envs=2,
      T=4,
      obs_shape=(3,),
      action_shape=(2,),
      state_shape=(5,),
    )
    batch.lengths[:] = 4

    sliced = batch.slice(1, 3)
    self.assertEqual(sliced.obs.shape, (2, 3, 3))
    self.assertEqual(sliced.actions.shape, (2, 2, 2))
    self.assertTrue(torch.equal(sliced.lengths, torch.tensor([2, 2])))

    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "trajectory.pt"
      sliced.save(str(path))
      loaded = TrajectoryBatch.load(str(path))

    self.assertTrue(torch.equal(loaded.obs, sliced.obs))
    self.assertTrue(torch.equal(loaded.actions, sliced.actions))
    self.assertTrue(torch.equal(loaded.dones, sliced.dones))


if __name__ == "__main__":
  unittest.main()
