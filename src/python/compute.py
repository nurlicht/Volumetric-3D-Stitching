import numpy as np
import scipy

from utils import Utilities


"""
Utilities for computation 
"""
class ComputeUtils:

  @staticmethod
  def interp3(
    x1: np.array,
    y1: np.array,
    z1: np.array,
    F1: np.array,
    x2: np.array,
    y2: np.array,
    z2: np.array
  ) -> np.array :
    return scipy.interpolate.RegularGridInterpolator((x1, y1, z1), F1, 'linear', False, 0)((x2, y2, z2))


  @staticmethod
  def meshgrid_3d(x: np.array, y: np.array, z: np.array) -> np.array:
    nx = x.size
    ny = y.size
    nz = z.size

    X = np.zeros((nx, ny, nz))
    Y = np.zeros((nx, ny, nz))
    Z = np.zeros((nx, ny, nz))

    for i in range(0, nx):
      for j in range(0, ny):
        for k in range(0, nz):
          X[i][j][k] = x[i]
          Y[i][j][k] = y[j]
          Z[i][j][k] = z[k]

    return [X, Y, Z]


  @staticmethod
  def meshgrid_3d_single(x: np.array) -> np.array:
    return ComputeUtils.meshgrid_3d(x, x, x)


  @staticmethod
  def polar_decompose(R: np.array) -> np.array:
    U, X ,V = np.linalg.svd(R)
    R_orth = np.matmul(U, V.T)
    R_orth = Utilities.to_real(R_orth)
    return R_orth


  @staticmethod
  def validate_eigen_values_vectors(
    A: np.array,
    eigen_values: np.array,
    eigen_vectors: np.array,
    tolerance: float = 1e-14 
  ) -> bool:
    Utilities.check(len(A.shape) == 2, 'A shape')
    Utilities.check(len(eigen_values.shape) == 1, 'eigen_values shape')
    Utilities.check(len(eigen_vectors.shape) == 2, 'eigen_vectors shape')
    Utilities.check(eigen_vectors.shape[1] == eigen_values.size, 'values vs. vectors')
    Utilities.check(eigen_vectors.shape[1] == eigen_values.size, 'values vs. vectors')
    Utilities.check(eigen_vectors.shape[0] == A.shape[1], 'vectors vs. A')
    [Utilities.check(0 < eigen_values[i], 'negative eigen_value') for i in range(eigen_values.size)]
    [Utilities.check(eigen_values[i] < 1 + 1e-12, 'too large an eigen_value ' + str(eigen_values[i])) for i in range(eigen_values.size)]

    for i in range(eigen_values.size):
      v = eigen_vectors[:, i]
      w = eigen_values[i]
      error = np.matmul(A, v) - w * v 
      error_max = abs(np.max(error.flatten()))
      Utilities.check(error_max < tolerance, 'residual of eigenvalue ' + str(i) + ' = ' + str(error_max))


  @staticmethod
  def validate_lsq_linear(
    A: np.array,
    b: np.array,
    M: np.array,
    tolerance: float = 1e-14
  ) -> bool:
    Utilities.check(len(A.shape) == 2, 'A shape')
    Utilities.check(len(b.shape) == 2, 'b shape')
    Utilities.check(len(M.shape) == 2, 'M shape')
    Utilities.check(A.shape[0] == b.shape[0], 'A vs. b')
    Utilities.check(A.shape[1] == M.shape[0], 'A vs. M')
    Utilities.check(M.shape[0] == b.shape[1], 'M vs. b')

    for i in range(A.shape[0]):
      error_max = abs(np.max((np.matmul(A[i, :], M) - b[i, :]).flatten()))
      Utilities.check(error_max < tolerance, 'residual of lsq element ' + str(i) + ' = ' + str(error_max))


  #A[:,i] * M = b[:, i]
  @staticmethod
  def lsq_linear(
    A: np.array,
    b: np.array
  ) -> np.array:
   M = np.transpose(np.matmul(
     np.matmul(np.transpose(b), A),
     np.linalg.pinv(np.matmul(np.transpose(A) , A))
   ))

   return M
