import unittest

import numpy as np
from compute import ComputeUtils
from utils import Utilities


class ComputeUtilsTest(unittest.TestCase):
    
  def test_validate_lsq_linear(self: unittest.TestCase) -> None:
   # Initialize
   M_expected = np.array([
      [1, -1],
      [2,  3]
   ])
   A = np.array([
      [0, -4],
      [-1, 1],
      [5,  0],
      [1,  2],
   ])
   b = np.matmul(A, M_expected)

   # Run
   M_actual = ComputeUtils.lsq_linear(A, b)

   # Assert
   ComputeUtils.validate_lsq_linear(A, b, M_actual, tolerance = 1e-14)  # real-life test
   Utilities.assert_almost_equal(self, M_actual, M_expected, p = 1e-14) # test with gold-standard


if __name__ == '__main__':
    unittest.main()
