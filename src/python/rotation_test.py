import math

import unittest

import numpy as np

from rotation import Rotation
from utils import Utilities

class RotationTest(unittest.TestCase):

  def test_hopf_quaternions(self: unittest.TestCase):
    # Initialize
    nRot1D = 1
    n_orient = round(math.pow(nRot1D, 3))
    
    # Run
    q = Rotation.uniform_so3_hopf(n_orient)

    # Assert
    self.assertIsNotNone(q)
    self.assertEqual(q.shape,       (4, n_orient))


  def test_quat_to_axes(self: unittest.TestCase):
    # Initialize
    nRot1D = 1
    n_orient = round(math.pow(nRot1D, 3))

    # Run
    q = Rotation.uniform_so3_hopf(n_orient)
    axes = Rotation.Quat2Axis(q)

    # Assert
    self.assertIsNotNone(q)
    self.assertIsNotNone(axes)
    self.assertEqual(q.shape,       (4, n_orient))
    self.assertEqual(axes.shape,    (3, n_orient))


  def test_axes_to_rot_mat(self: unittest.TestCase):
    # Initialize
    nRot1D = 1
    n_orient = round(math.pow(nRot1D, 3))
    
    # Run
    q = Rotation.uniform_so3_hopf(n_orient)
    axes = Rotation.Quat2Axis(q)
    rot_mat = Rotation.Axis2RotMatBatch(axes)

    # Assert
    self.assertIsNotNone(q)
    self.assertIsNotNone(axes)
    self.assertIsNotNone(rot_mat)
    self.assertEqual(q.shape,       (4, n_orient))
    self.assertEqual(axes.shape,    (3, n_orient))
    self.assertEqual(rot_mat.shape, (9, n_orient))


  def test_all_rot(self: unittest.TestCase):
    # Initialize
    nRot1D = 1
    n_orient = round(math.pow(nRot1D, 3))
    
    # Run
    rot_mat = Rotation.all_rot_variables(n_orient)[0]

    # Assert
    self.assertIsNotNone(rot_mat)


  def test_mean_geodesci_distance_degrees(self: unittest.TestCase):
    # Initialize
    noise_geodesic_degrees = [40, 10, 5, 1, 0.1, 0.01]
    n_orient = 20 ** 3

    q1 = Rotation.uniform_so3_hopf(n_orient)

    for i in range(len(noise_geodesic_degrees)):
      # Initialize
      noise_quaternion = 0.99 * (np.pi / 180) * (0.5 * noise_geodesic_degrees[i])
      q2 = q1 + noise_quaternion * np.random.randn(4, n_orient)

      # Run
      mean_geodesci_distance = Rotation.mean_geodesic_distance_degrees(q1, q2)

      # Assert
      self.assertLess(mean_geodesci_distance, noise_geodesic_degrees[i])


  def test_axis_to_rot_mat(self: unittest.TestCase):
    # Initialize
    Axis = (np.pi / 2) * np.array([0.0, 0.0, 1.0])
    rot_mat_expected= np.array([
      [0.0,  -1.0,  0.0],
      [1.0,   0.0,  0.0],
      [0.0,   0.0,  1.0],
    ])
    
    # Run
    rot_mat_actual = Rotation.Axis2RotMat(Axis)

    # Assert
    Utilities.assert_almost_equal(self, rot_mat_actual, rot_mat_expected, 1e-15)


  def test_rot_mat_to_axis(self: unittest.TestCase):
    # Initialize
    rot_mat= np.array([
      [0.0,  -1.0,  0.0],
      [1.0,   0.0,  0.0],
      [0.0,   0.0,  1.0],
    ])
    Axis_expected = (np.pi / 2) * np.array([0.0, 0.0, 1.0]).reshape((3, 1))
    
    # Run
    Aixs_actual = Rotation.rot_mat_to_axis(rot_mat)

    # Assert
    Utilities.assert_almost_equal(self, Aixs_actual, Axis_expected, 1e-15)


  def test_rot_mat_to_quat(self: unittest.TestCase):
    # Initialize
    rot_mat= np.array([
      [0.0,  -1.0,  0.0],
      [1.0,   0.0,  0.0],
      [0.0,   0.0,  1.0],
    ])
    quat_expected = math.sqrt(0.5) * np.array([1.0, 0.0, 0.0, 1.0]).reshape((4, 1))
    
    # Run
    quat_actual = Rotation.rot_mat_to_quat(rot_mat)

    # Assert
    Utilities.assert_almost_equal(self, quat_actual, quat_expected, 1e-15)


  def test_rot_mat_rotate_point(self: unittest.TestCase):
    # Initialize
    rot_mat= np.array([
      [0.0,  -1.0,  0.0],
      [1.0,   0.0,  0.0],
      [0.0,   0.0,  1.0],
    ])
    A = np.array([1.0, 1.0, 0.0]).reshape((3, 1))
    A_rotated_expected = np.array([-1.0, 1.0, 0.0]).reshape((3, 1))
    
    # Run
    A_rotated_actual = np.matmul(rot_mat, A)

    # Assert
    Utilities.assert_almost_equal(self, A_rotated_actual, A_rotated_expected, 1e-15)


if __name__ == '__main__':
    unittest.main()
