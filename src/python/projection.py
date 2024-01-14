import numpy as np
from rotation import Rotation
from utils import Utilities


"""
(Fourier) Projector

  - Get (Fourier) projected images corresponding to rotation matrices
  
"""
class Projector:
    
  @staticmethod
  def generate(nRot1D: int) -> [np.array]:
    # Input parameters
    Experiment = Projector.experiment_parameters()['Experiment']
    Protein = Projector.synsthesize_3d(Experiment['n_voxel_1d'])
    N_loop = nRot1D ** 3   #28 Cube of an "even" integer

    # Camera coordinate (k-space)
    [Q, Circle_Index] = Projector.q_and_mask(Experiment)

    R = Rotation.all_rot_variables(N_loop)[0]
    k = Projector.fourier_scaled_axes_from_grid(Protein['Grid_3D'])[2]

    N_p2 = Experiment['N_p'] ** 2
    Images = np.random.randn(N_p2 , N_loop)

    # (Fourier-space) Projections
    for cntr in range(0, N_loop):
      Images[:, cntr] = Projector.generate_single_image(
        Protein['ED'],
        R[:,cntr].reshape((3, 3)),
        k,
        Q,
        Circle_Index
      )[0]

    return [Images.T, R]


  @staticmethod
  def generate_single_image(
    ED: np.array, R: np.array, k: np.array, Q: np.array, Circle_Index: np.array
  ) -> [np.array]:
    ED_rot = Rotation.rotate_structure_index(ED, R)
    ED_rot_f = np.fft.fftshift(abs(np.fft.fftn(ED_rot)))

    Camera_I = Utilities.interp3(
      k['x'], k['y'], k['z'],
      ED_rot_f,
      Q['x'], Q['y'], Q['z']
    )

    Camera_I[Circle_Index] = 0

    Image = Camera_I.reshape((Camera_I.size,))

    return [Image, ED_rot, ED_rot_f, Camera_I]


  @staticmethod
  def synsthesize_3d(n_voxel_1d) -> {}:
    N = n_voxel_1d
    U = (np.linspace(1, N, N) - (N + 1) / 2) / (N / 2)

    [z, x, y] = Utilities.meshgrid_3d_single(U)

    A = x / 0.47
    B = y / 0.37
    C = z / 0.29

    F = 1 - 0.4 * ((A - 0.15) ** 2 + (B + 0.2) ** 2 + (C - 0.1) * (A - 0.15) * (B + 0.2))

    F[np.cos(20 * np.pi * (x - z - 0.2) * abs(y + z + 0.1) * abs(z - 0.3)) < 0.2] = 0
    F[A ** 2 + B ** 2 + C ** 2 > 1] = 0
    F[F < 0] = 0

    return {'ED': F, 'Grid_3D': {'x': 2e-7 * x, 'y': 2e-7 * y, 'z': 2e-7 * z}}


  @staticmethod
  def experiment_parameters() -> {}:
     parameters = {
      'N_P_NoBin': 1024,
      'Experiment': {
        'n_voxel_1d': 31,
        'Pixel': 75e-6,
        'zD': 0.5,
        'Lambda': 2*1e-9,
        'N_p': np.nan, # number of pixels along each coordinate
        'SuperPixel': np.nan,
        'Width': np.nan
      }
     }

     parameters['Experiment']['N_p'] = 1 + 2 * parameters['Experiment']['n_voxel_1d']

     parameters['Experiment']['SuperPixel'] = parameters['Experiment']['Pixel'] * (
      parameters['N_P_NoBin'] / parameters['Experiment']['N_p'])

     parameters['Experiment']['Width'] = parameters['Experiment']['SuperPixel'] * (
      parameters['Experiment']['N_p'])
     
     return parameters


  @staticmethod
  def fourier_scaled_axis(Number: int, Length: float) -> [float]:
    d = Length / (Number - 1)
    Nyquist = 0.5 / d
    N1 = (Number - 1) / 2
    k = np.arange(-N1, N1 + 1) * (2 * Nyquist / Number)
    return [Nyquist, k]


  @staticmethod
  def fourier_scaled_axes(Number: [int], Length: [float]) -> [{}]:
    [Nyquist_x, k_x_1D] = Projector.fourier_scaled_axis(Number[0], Length['x'])
    [Nyquist_y, k_y_1D] = Projector.fourier_scaled_axis(Number[1], Length['y'])
    [Nyquist_z, k_z_1D] = Projector.fourier_scaled_axis(Number[2] ,Length['z'])
 
    [k_z, k_x, k_y] = Utilities.meshgrid_3d(k_z_1D, k_x_1D, k_y_1D)
 
    return [
      {'x': Nyquist_x, 'y': Nyquist_y, 'z': Nyquist_z},
      {'x': k_x, 'y': k_y, 'z': k_z},
      {'x': k_x_1D, 'y': k_y_1D, 'z': k_z_1D}
    ]

  @staticmethod
  def fourier_scaled_axes_from_grid(Grid_3D: np.array) -> [{}]:
    [Length, Number] = Projector.extract_coordinates(Grid_3D)
 
    return Projector.fourier_scaled_axes(Number, Length)

  @staticmethod
  def extract_coordinates(Grid_3D: np.array) -> [{}, np.shape]:
    Temp = Grid_3D['x'].flatten()
    Length_x = max(Temp) - min(Temp)
 
    Temp = Grid_3D['y'].flatten()
    Length_y = max(Temp) - min(Temp)
 
    Temp = Grid_3D['z'].flatten()
    Length_z = max(Temp) - min(Temp)
 
    return [{'x': Length_x, 'y': Length_y, 'z': Length_z}, Grid_3D['x'].shape]


  @staticmethod
  def q_and_mask(Experiment: {}) -> [{}, np.array]:
    Lambda = Experiment['Lambda']
    zD = Experiment['zD']
    Width = Experiment['Width']
 
    [Camera_x, Camera_y] = Projector.camera_x_y(Experiment)
 
    Temp = Lambda * np.sqrt(Camera_x ** 2 + Camera_y ** 2 + zD ** 2)
 
    Q_x = Camera_x / Temp
    Q_y = Camera_y / Temp
    Q_z = (zD /Temp -1 / Lambda)

    Circle_Index = (Camera_x ** 2 + Camera_y ** 2) > (Width / 2) ** 2
 
    return [{'x': Q_x, 'y': Q_y, 'z': Q_z}, Circle_Index]


  @staticmethod
  def camera_x_y(Experiment: {}) -> [np.array, np.array]:
    Width = Experiment['Width']
    N = Experiment['N_p']
    U = (Width / (N - 1)) * (np.linspace(1, N, N) - (N + 1) / 2)
 
    [Camera_x, Camera_y] = np.meshgrid(U, U, indexing = 'xy')
 
    return [Camera_x, Camera_y]


  @staticmethod
  def fft_shift(F: np.array) -> np.array:
    N = [F.shape[0], F.shape[1], F.shape[2]]
    Nh = [(N[0] - 1) / 2, (N[1] - 1) / 2, (N[2] - 1) / 2]
    Index = [None, None, None]

    for cntr in range(0, 3):
      a = np.arange(Nh[cntr] + 2, N[cntr] + 1)
      b = np.arange(1, Nh[cntr] + 2)
      c = [a.reshape(1, a.size), b.reshape(1, b.size)]
      Index[cntr] = np.hstack(c) - 1

    G = F.copy()
    for i in range(0, F.shape[0]):
      for j in range(0, F.shape[1]):
        for k in range(0, F.shape[2]):
          G[i][j][k] = F[(int) (Index[0][0][i])][(int) (Index[1][0][j])][(int) (Index[2][0][k])]

    return G
