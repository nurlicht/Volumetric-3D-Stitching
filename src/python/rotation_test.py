import math

import unittest

from rotation import Rotation

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

if __name__ == '__main__':
    unittest.main()
