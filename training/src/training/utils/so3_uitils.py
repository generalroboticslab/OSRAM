import torch
import torch.nn.functional as F


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def quat_to_ortho6d(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  """
  Convert quaternion [w, x, y, z] to 6D rotation representation.
  Input:  quat (..., 4)
  Output: ortho6d (..., 6) = first two columns of rotation matrix
  """
  q = F.normalize(quat, dim=-1, eps=eps)
  w, x, y, z = q.unbind(dim=-1)

  # Rotation matrix from unit quaternion
  r00 = 1 - 2 * (y * y + z * z)
  r01 = 2 * (x * y - z * w)
  r02 = 2 * (x * z + y * w)

  r10 = 2 * (x * y + z * w)
  r11 = 1 - 2 * (x * x + z * z)
  r12 = 2 * (y * z - x * w)

  r20 = 2 * (x * z - y * w)
  r21 = 2 * (y * z + x * w)
  r22 = 1 - 2 * (x * x + y * y)

  R = torch.stack(
    [
      torch.stack([r00, r01, r02], dim=-1),
      torch.stack([r10, r11, r12], dim=-1),
      torch.stack([r20, r21, r22], dim=-1),
    ],
    dim=-2,
  )  # (..., 3, 3)

  c1 = R[..., :, 0]
  c2 = R[..., :, 1]
  return torch.cat([c1, c2], dim=-1)


def ortho6d_to_quat(ortho6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  """
  Convert 6D rotation representation back to quaternion [w, x, y, z].
  Input:  ortho6d (..., 6)
  Output: quat (..., 4)
  """
  a1 = ortho6d[..., 0:3]
  a2 = ortho6d[..., 3:6]

  b1 = F.normalize(a1, dim=-1, eps=eps)
  a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
  b2 = F.normalize(a2 - a2_proj, dim=-1, eps=eps)
  b3 = torch.cross(b1, b2, dim=-1)

  R = torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3), columns

  # Matrix -> quaternion [w, x, y, z]
  m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
  m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
  m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

  trace = m00 + m11 + m22
  qw = torch.sqrt(torch.clamp(1.0 + trace, min=eps)) / 2.0
  qx = (m21 - m12) / (4.0 * qw + eps)
  qy = (m02 - m20) / (4.0 * qw + eps)
  qz = (m10 - m01) / (4.0 * qw + eps)

  q = torch.stack([qw, qx, qy, qz], dim=-1)
  return F.normalize(q, dim=-1, eps=eps)
