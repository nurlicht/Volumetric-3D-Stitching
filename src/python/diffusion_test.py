import unittest

import numpy as np
from diffusion import DiffusionMapSolver

from utils import Utilities


class DiffusionMapSolverTest(unittest.TestCase):

  """
    %  A1      A2
    %  A3      A4
    %  O
    %A= [0 4 0 4; 3 3 0 0]
    %
    %     0   4    3    5
    %S2 = 4   0    5    3
    %     3   5    0    4
    %     5   3    4    0
    %
    %     1   3    2    4
    %N =  2   4    1    3
    %     3   1    4    2
    %     4   2    3    1
  """
  def test_s2_to_w_matrix(self: unittest.TestCase):
     # Initialize
    S2 = np.array([
      [0.0,   16.0,  9.0,   25.0],
      [16.0,  0.0,   25.0,  9.0 ],
      [9.0,   25.0,  0.0,   16.0],
      [25.0,  9.0,   16.0,  0.0],
    ])
    N = np.array([
      [1,    3,    2,    4  ],
      [2,    4,    1,    3  ],
      [3,    1,    4,    2  ],
      [4,    2,    3,    1  ],
    ]) - 1
    Epsilon = 1.0

    #W1
    W_expected = np.array([
      [1.0,      0.48675,    0.27804,     0.13534],
      [0.13534,  0.27804,    0.48675,     1.0    ],
      [0.13534,  0.27804,    0.48675,     1.0    ],
      [1.0,      0.48675,    0.27804,     0.13534],
    ])

    # Run
    W_actual = DiffusionMapSolver.s2_to_w_matrix(S2, N, Epsilon, s2_median_index = 3, index_offset = 0)

    # Assert
    Utilities.assert_almost_equal(self, W_actual, W_expected, p = 1e-4)


  def test_anisotropic_norm(self: unittest.TestCase):
     # Initialize
    #W1
    W = np.array([
      [1.0,      0.48675,    0.27804,     0.13534],
      [0.13534,  0.27804,    0.48675,     1.0    ],
      [0.13534,  0.27804,    0.48675,     1.0    ],
      [1.0,      0.48675,    0.27804,     0.13534],
    ])

    #W2
    W_expected = np.array([
      [0.27697,   0.08615,   0.057246,  0.15723 ],
      [0.08615,   0.077009,  0.10591,   0.20589 ],
      [0.057246,  0.10591,   0.13482,   0.17699 ],
      [0.15723,   0.20589,   0.17699,   0.037484],
    ])

    # Run
    W_actual = DiffusionMapSolver.anisotropic_norm(W)

    # Assert
    Utilities.assert_almost_equal(self, W_actual, W_expected, p = 1e-4)


  def test_dm_cov(self: unittest.TestCase):
     # Initialize
    #W2
    W = np.array([
      [0.27697,   0.08615,   0.057246,  0.15723 ],
      [0.08615,   0.077009,  0.10591,   0.20589 ],
      [0.057246,  0.10591,   0.13482,   0.17699 ],
      [0.15723,   0.20589,   0.17699,   0.037484],
    ])

    #W2
    P_ep_expected = np.array([
      [0.47952,  0.16448,   0.1093,    0.27221 ],
      [0.16448,  0.16213,   0.22299,   0.3931  ],
      [0.1093,   0.22299,   0.28385,   0.33791 ],
      [0.27221,  0.3931,    0.33791,   0.064897],
    ])

    D_expected = np.array([
      [1.3158,   0.0,       0.0,     0.0   ],
      [0.0,      1.4510,    0.0,     0.0   ],
      [0.0,      0.0,       1.4510,  0.0   ],
      [0.0,      0.0,       0.0,     1.3158],
    ])

    # Run
    [P_ep_actual, D_actual] = DiffusionMapSolver.dm_cov(W)

    # Assert
    Utilities.assert_almost_equal(self, P_ep_actual, P_ep_expected, p = 1e-4)
    Utilities.assert_almost_equal(self, D_actual, D_expected, p = 1e-4)


  def test_diffusion_coordinate(self: unittest.TestCase):
     # Initialize
    S2 = np.array([
      [0.0,   16.0,  9.0,   25.0],
      [16.0,  0.0,   25.0,  9.0 ],
      [9.0,   25.0,  0.0,   16.0],
      [25.0,  9.0,   16.0,  0.0],
    ])
    N = np.array([
      [1,    3,    2,    4  ],
      [2,    4,    1,    3  ],
      [3,    1,    4,    2  ],
      [4,    2,    3,    1  ],
    ]) - 1
    Epsilon = 1.0
    k = 2
    validate = False
                                            
    Ps_expected = np.array([
      [-0.689225,  0.060209],
      [-0.689225,  0.235587],
      [-0.689225,  0.115495],
      [-0.689225,  -0.348909]
    ])
    Lambda_expected = np.array([
      1.0, -0.3242
    ])

    # Run
    [Ps_actual, Lambda_actual] = DiffusionMapSolver.diffusion_coordinate(S2, N, Epsilon, k, validate)

    # Assert
    Utilities.assert_almost_equal(self, Ps_actual, Ps_expected, p = 1e-4)
    Utilities.assert_almost_equal(self, Lambda_actual, Lambda_expected, p = 1e-4)


if __name__ == '__main__':
    unittest.main()
