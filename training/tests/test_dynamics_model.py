import unittest

import jax.numpy as jnp
import optax
from flax import nnx

from training.dynamics.training.tron1.tron1_model import (
  Tron1Model,
  Tron1ModelConfig,
)
from training.utils.model_utils import update


class Tron1ModelTest(unittest.TestCase):
  def test_cpu_forward_shape(self):
    model = Tron1Model(
      Tron1ModelConfig(
        input_dim=5,
        output_dim=2,
        obs_history_dim=3,
        hidden_sizes=(4,),
        activation="relu",
        type="mlp",
      )
    )

    output = model(jnp.ones((3, 5), dtype=jnp.float32))

    self.assertEqual(output.shape, (3, 2))

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    loss, _ = update(
      model,
      jnp.ones((3, 5), dtype=jnp.float32),
      jnp.zeros((3, 2), dtype=jnp.float32),
      optimizer,
    )
    self.assertTrue(jnp.isfinite(loss))


if __name__ == "__main__":
  unittest.main()
