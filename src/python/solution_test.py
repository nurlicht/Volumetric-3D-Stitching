import unittest


from projection import Projector
from solution import OrientationRecoverySolver


class SolutionTest(unittest.TestCase):

  def test_solve(self: unittest.TestCase):
    # Initialize
    parameters = {}
    Experiment = Projector.experiment_parameters()['Experiment']
    n_pixels_2d = Experiment['N_p'] ** 2
    nRot1D = OrientationRecoverySolver.default_parameters()['nRot1D']
    N_loop = nRot1D ** 3

    # Run
    [[Ps, Lambda, c0, c, S2, N, Images, R], titles] = OrientationRecoverySolver.solve(parameters)

    # Assert
    self.assertIsNotNone(Ps)
    self.assertEqual(Ps.shape, (N_loop, 10))

    self.assertIsNotNone(Lambda)
    self.assertEqual(Lambda.shape, (10,))

    self.assertIsNotNone(c0)
    self.assertEqual(c0.shape, (9, 9))

    self.assertIsNotNone(c)
    self.assertEqual(c.shape, (9, 9))

    self.assertIsNotNone(S2)
    self.assertEqual(S2.shape, (N_loop, 150))

    self.assertIsNotNone(N)
    self.assertEqual(N.shape, (N_loop, 150))

    self.assertIsNotNone(Images)
    self.assertEqual(Images.shape, (N_loop, n_pixels_2d))

    self.assertIsNotNone(R)
    self.assertEqual(R.shape, (9, N_loop))


if __name__ == '__main__':
    unittest.main()
