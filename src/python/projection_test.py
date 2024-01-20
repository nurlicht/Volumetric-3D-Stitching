import unittest

import numpy as np

from projection import Projector
from rotation import Rotation

class ProjectorTest(unittest.TestCase):

  def test_synsthesize_3d(self: unittest.TestCase):
    # Initialize
    n_voxel_1d = 31

    # Run
    result = Projector.synsthesize_3d(n_voxel_1d)['ED']

    # Assert
    self.assertIsNotNone(result)


  def test_camera_x_y(self: unittest.TestCase):
    # Initialize
    n_voxel_1d = 31
    Experiment = Projector.experiment_parameters(n_voxel_1d)['Experiment']

    # Run
    [Camera_x, Camera_y] = Projector.camera_x_y(Experiment)

    # Assert
    self.assertIsNotNone(Camera_x)
    self.assertIsNotNone(Camera_y)
    self.assertEqual(Camera_x.shape, Camera_y.shape)
    self.assertEqual(np.min(Camera_x), np.min(Camera_y))
    self.assertEqual(np.max(Camera_x), np.max(Camera_y))
    self.assertEqual(Camera_x.shape[0], Experiment['N_p'])

        
  def test_q_and_mask(self: unittest.TestCase):
    # Initialize
    n_voxel_1d = 31
    Experiment = Projector.experiment_parameters(n_voxel_1d)['Experiment']

    # Run
    [Q, Circle_Index] = Projector.q_and_mask(Experiment)

    # Assert
    self.assertIsNotNone(Q)
    self.assertIsNotNone(Circle_Index)


  def test_generate_single_image(self: unittest.TestCase):
    # Initialize
    n_voxel_1d = 1
    nRot1D = 1
    N_loop = nRot1D ** 3
    R = Rotation.all_rot_variables(N_loop)
    Experiment = Projector.experiment_parameters(n_voxel_1d)['Experiment']
    Protein = Projector.synsthesize_3d(Experiment['n_voxel_1d'])
    [Q, Circle_Index] = Projector.q_and_mask(Experiment)
    k = Projector.fourier_scaled_axes_from_grid(Protein['Grid_3D'])[2]

    # Run
    Image = Projector.generate_single_image(
        Protein['ED'],
        R[:][0].reshape((3, 3)),
        k,
        Q,
        Circle_Index
    )[0]

    # Assert
    self.assertIsNotNone(Image)
    self.assertEqual(Image.size, Circle_Index.size)


  def test_generate(self: unittest.TestCase):
    # Initialize
    n_voxel_1d = 31
    nRot1D = 2
    N_loop = nRot1D ** 3
    Experiment = Projector.experiment_parameters(n_voxel_1d)['Experiment']

    # Run
    [Images, R, Axis, Quat] = Projector.generate(nRot1D, n_voxel_1d)

    # Assert
    self.assertIsNotNone(Images)
    self.assertEqual(Images.shape, (N_loop, Experiment['N_p'] ** 2))


if __name__ == '__main__':
    unittest.main()
