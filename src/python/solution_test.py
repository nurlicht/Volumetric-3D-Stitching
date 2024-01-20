import unittest
from matplotlib.figure import Figure

import numpy as np


from projection import Projector
from solution import OrientationRecoverySolver


class SolutionTest(unittest.TestCase):
  def test_solve(self: unittest.TestCase):
    # Initialize
    parameters = OrientationRecoverySolver.default_parameters()
    parameters['plot'] = False
    k = parameters['k']
    nRot1D = parameters['nRot1D']
    N_loop = nRot1D ** 3
    Experiment = Projector.experiment_parameters(parameters['n_voxel_1d'])['Experiment']
    n_pixels_2d = Experiment['N_p'] ** 2

    # Run
    array_map: dict[str, np.array] = OrientationRecoverySolver.solve(parameters)

    # Assert
    Ps = array_map['Ps']
    self.assertIsNotNone(Ps)
    self.assertEqual(Ps.shape, (N_loop, 10))

    Lambda = array_map['Lambda']
    self.assertIsNotNone(Lambda)
    self.assertEqual(Lambda.shape, (10,))

    c0 = array_map['c0']
    self.assertIsNotNone(c0)
    self.assertEqual(c0.shape, (9, 9))

    c = array_map['c']
    self.assertIsNotNone(c)
    self.assertEqual(c.shape, (9, 9))

    S2 = array_map['S2']
    self.assertIsNotNone(S2)
    self.assertEqual(S2.shape, (N_loop, k))

    N = array_map['N']
    self.assertIsNotNone(N)
    self.assertEqual(N.shape, (N_loop, k))

    Images = array_map['Images']
    self.assertIsNotNone(Images)
    self.assertEqual(Images.shape, (N_loop, n_pixels_2d))

    R = array_map['R']
    self.assertIsNotNone(R)
    self.assertEqual(R.shape, (9, N_loop))

  def test_solve_and_plots(self: unittest.TestCase):
    # Initialize
    parameters = OrientationRecoverySolver.default_parameters()
    parameters['plot'] = False

    # Run
    plot_arrays: [[str, str, Figure]] = OrientationRecoverySolver.solve_and_plot(parameters)

    # Assert
    self.assertIsNotNone(plot_arrays)

    self.assertEqual(len(plot_arrays), 20)
    self.assertEqual(len(plot_arrays[0]), 1)
    [self.assertEqual(len(plot_arrays[i]), 2) for i in range(1, len(plot_arrays))]
  """
  """


if __name__ == '__main__':
    unittest.main()
