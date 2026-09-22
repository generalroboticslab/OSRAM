import unittest

import jax.numpy as jnp

from training.utils.data_utils import construct_dataset


class ConstructDatasetTest(unittest.TestCase):
  def test_constructs_expected_samples(self):
    actions = jnp.arange(2 * 6 * 2, dtype=jnp.float32).reshape(2, 6, 2)
    observations = jnp.arange(2 * 6 * 3, dtype=jnp.float32).reshape(2, 6, 3)

    dataset = construct_dataset(actions, observations, 2, 1)

    self.assertEqual(len(dataset), 8)
    self.assertEqual(dataset[0][0].shape, (8,))
    self.assertEqual(dataset[0][1].shape, (3,))

  def test_rejects_mismatched_trajectories(self):
    with self.assertRaisesRegex(ValueError, "same num_envs and T"):
      construct_dataset(jnp.zeros((1, 4, 2)), jnp.zeros((2, 4, 3)), 1, 1)


if __name__ == "__main__":
  unittest.main()
