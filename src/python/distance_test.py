import unittest

import numpy as np

from distance import DistanceMatrix
from utils import Utilities


class ProjectorTest(unittest.TestCase):

  def test_knn_from_round_distance(self: unittest.TestCase):
     # Initialize
    B = np.array([
      [0,     -0.8,  -0.6],
      [-0.8,  0,     0.6 ],
      [0.8,   -0.6,  0   ]
    ])
    k = 2

    N_expected = [
      [1,  2],
      [0,  1],
      [1, 2]
    ]
    S2_expected = [
      [-0.8,  -0.6],
      [-0.8,  0   ],
      [-0.6,  0   ]
    ]

    n_rot = B.shape[0]

    # Run
    [S2, N] = DistanceMatrix.knn_from_round_distance(B, k)

    # Assert
    self.assertEqual(S2.shape, (n_rot, k))
    [self.assertEqual(S2[i][j], S2_expected[i][j]) for i in range(n_rot) for j in range(k)]

    self.assertEqual(N.shape, (n_rot, k))
    [self.assertEqual(N[i][j],  N_expected[i][j] ) for i in range(n_rot) for j in range(k)]


  def test_convert_to_round_distance(self: unittest.TestCase):
    # Initialize
    A = np.array([[3, 4], [-3, 4], [6, -8]])

    n_rot = A.shape[0]

    # Run
    B = DistanceMatrix.convert_to_round_distance(A)

    # Assert
    self.assertEqual(B.shape, (n_rot, n_rot))
    Utilities.assert_almost_equal(self, B, B.T, 1e-15)
    [self.assertLessEqual(B[i][i], 1e-6) for i in range(n_rot)]
    [self.assertLessEqual(B[i][j], np.pi) for i in range(n_rot) for j in range(n_rot)]


  def test_set_norm_to_one(self: unittest.TestCase):
    # Initialize
    A = np.array([[3, 4], [-3, 4], [6, -8]])

    # Run
    B = DistanceMatrix.set_norm_to_one(A)

    # Assert
    self.assertTrue(Utilities.are_equal(B, np.array([[0.6, 0.8], [-0.6, 0.8], [0.6, -0.8]])))


  def test_remove_offset(self: unittest.TestCase):
    # Initialize
    A = np.array([[1, 2], [-1, -1], [3, 5]])

    # Run
    B = DistanceMatrix.remove_offset(A)

    # Assert
    self.assertTrue(Utilities.are_equal(B, np.array([[0, 0], [-2, -3], [2, 3]])))


if __name__ == '__main__':
    unittest.main()
