import unittest

import numpy as np
import scipy.interpolate


"""
Utilities
"""
class Utilities:

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
    return Utilities.meshgrid_3d(x, x, x)
  

  @staticmethod
  def are_equal(x_: np.array, y_: np.array) -> bool:
    if (x_.shape != y_.shape):
      return False

    x = x_.flatten()
    y = y_.flatten()

    tolerance = Utilities.tolerance()

    if (x.size == 1):
      return abs(x[0] - y[0]) < tolerance

    for i in range(0, x.size):
      if (not Utilities.are_equal(x[i], y[i])):
        return False

    return True
  
  @staticmethod
  def has_nan(x: np.array) -> bool:
    for a in x.flatten():
      if (a != a):
        return True

    return False
  
  
  @staticmethod
  def assert_almost_equal(self: unittest.TestCase, x_: np.array, y_: np.array, p: float):
    self.assertEqual(x_.shape, y_.shape)

    x = x_.flatten()
    y = y_.flatten()

    [Utilities.assert_almost_equal_hybrid_single(self, x[i], y[i], p) for i in range(x.size)]

  @staticmethod
  def assert_almost_equal_hybrid_single(self: unittest.TestCase, x: np.array, y: np.array, p: float):
    self.assertTrue(Utilities.are_almost_equal_hybrid_single(x, y, p))

  @staticmethod
  def are_almost_equal(x_: np.array, y_: np.array, p: float) -> bool:
    if (x_.shape != y_.shape):
      return False

    x = x_.flatten()
    y = y_.flatten()

    for i in range(x.size):
      if (not Utilities.are_almost_equal_hybrid_single(x[i], y[i], p)):
        return False

    return True

  @staticmethod
  def are_almost_equal_hybrid_single(x: np.array, y: np.array, p: float) -> bool:
    return (
      (abs(x) < p and abs(y) < p) or
      abs((x - y) / y) < p
    )

  @staticmethod
  def tolerance() -> float:
    return 1e-12 # 1e-12
