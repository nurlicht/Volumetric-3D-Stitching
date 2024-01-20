import unittest

import numpy as np


"""
Utilities for test and miscellaneous (non-computational) functionalities
"""
class Utilities:

  @staticmethod
  def are_equal(x_: np.array, y_: np.array, tolerance: float = 1e-12) -> bool:
    if (x_.shape != y_.shape):
      return False

    x = x_.flatten()
    y = y_.flatten()

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
  def to_real(x: np.array, tolerance: float = 1e-50) -> np.array:
    if (not Utilities.are_almost_equal(np.imag(x), np.zeros(x.shape), tolerance)):
      raise Exception('Complex array was encountered.')
    return np.real(x)

  
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
      (y != 0 and abs((x - y) / y) < p)
    )


  @staticmethod
  def log_progress(label: str, i: int, N: int) -> None:
    interval: int = 1
    if (N > 1 and i > -1 and i % interval == 0 and label != None):
      print (label + ': ' + str(round(100 * i / (N - 1))) + '%' + ' completed.', end = '\r', flush = True)


  @staticmethod
  def check(flag: bool, message: str):
    if (not flag):
      raise Exception(message)


  @staticmethod
  def check_nan_complex(x: np.array, x_label: str = '') -> bool:
    Utilities.check(
      not np.iscomplexobj(x) and not Utilities.has_nan(x),
      'Invalid array was encountered' + ((' (' + x_label + ').') if x_label != '' else '')
    )
