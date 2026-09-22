import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch


def torch_to_jax(tensor: torch.Tensor, silent: bool = True) -> jnp.ndarray:
  """Try zero-copy DLPack first, fall back to CPU NumPy copy if it fails."""
  # ensure it's a plain torch.Tensor
  if not isinstance(tensor, torch.Tensor):
    raise TypeError(f"expected torch.Tensor, got {type(tensor)}")

  try:
    # ensure contiguous for safety
    t_contig = tensor.contiguous()
    dl = torch.utils.dlpack.to_dlpack(t_contig)
    return jax.dlpack.from_dlpack(dl)
  except Exception as e:
    # common reasons: CPU->GPU mismatch, incompatible builds; fallback to numpy copy
    if not silent:
      print("DLPack -> JAX failed:", e, "- falling back to NumPy copy")
    return jnp.asarray(tensor.detach().cpu().numpy())


def jax_to_torch(array: jnp.ndarray, silent: bool = True) -> torch.Tensor:
  """Try zero-copy DLPack first, fall back to CPU NumPy copy if it fails."""
  if not isinstance(array, jnp.ndarray):
    raise TypeError(f"expected jnp.ndarray, got {type(array)}")

  try:
    dl = jax.dlpack.to_dlpack(array)
    return torch.utils.dlpack.from_dlpack(dl)
  except Exception as e:
    # common reasons: CPU->GPU mismatch, incompatible builds; fallback to numpy copy
    if not silent:
      print("DLPack -> PyTorch failed:", e, "- falling back to NumPy copy")
    np_arr = np.asarray(array)
    np_arr = np.ascontiguousarray(np_arr)
    if not np_arr.flags.writeable:
      np_arr = np_arr.copy()
    return torch.from_numpy(np_arr)


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
  """Convert a PyTorch tensor to a NumPy array."""
  if not isinstance(tensor, torch.Tensor):
    raise TypeError(f"expected torch.Tensor, got {type(tensor)}")
  return tensor.detach().cpu().numpy()
