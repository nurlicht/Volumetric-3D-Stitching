import math
import numpy as np

from compute import ComputeUtils
from utils import Utilities



"""
Rotation

  - Generate rotation matrices
  
"""
class Rotation:

  @staticmethod
  def all_rot_variables(N: int) -> [np.array]:
    Q = Rotation.uniform_so3_hopf(N)

    Axis = Rotation.Quat2Axis(Q)

    R = Rotation.Axis2RotMatBatch(Axis)

    return [R, Axis, Q]

  @staticmethod
  def uniform_so3_hopf(n_orient: int) -> np.array:
    n1 = round(math.pow(n_orient, (1/3)))
    if (n_orient != math.pow(n1, 3)):
      raise Exception('Input argument should be the cube of a positive integer.')

    Psi_ = (2 * np.pi) * np.linspace(0, 1, n1 + 1)[0:n1]
    Theta_ = np.arccos(np.linspace(1, -1, n1)[0:n1])
    Phi_ = (2 * np.pi) * np.linspace(0, 1, n1 + 1)[0:n1]

    Psi_ = Psi_- np.mean(Psi_)
    Theta_ = Theta_- np.mean(Theta_) + (np.pi / 2)
    Phi_ = Phi_- np.mean(Phi_)

    Phi, Psi, Theta = ComputeUtils.meshgrid_3d(Phi_, Psi_, Theta_)

    Psi = Psi.reshape((1, n_orient))
    Theta = Theta.reshape((1, n_orient))
    Phi = Phi.reshape((1, n_orient))

    Q = np.vstack([np.cos(Theta / 2) * np.cos(Psi / 2),
      np.sin(Theta / 2) * np.sin(Phi + Psi / 2),
      np.sin(Theta / 2) * np.cos(Phi + Psi / 2),
      np.cos(Theta / 2) * np.sin(Psi / 2)])

    Q = Rotation.unit_mag_pos(Q)

    Q = Utilities.to_real(Q)

    return Q


  @staticmethod
  def unit_mag_pos(q: np.array) -> np.array:
    nprm_q = np.zeros((1, q.shape[1])).reshape((1, q.shape[1]))

    for cntr in range(4):
      nprm_q = nprm_q + q[cntr, :].reshape((1, q.shape[1])) ** 2

    nprm_q = np.sqrt(nprm_q)
    factor = np.sign(q[0, :]).reshape((1, q.shape[1])) / nprm_q

    for cntr in range(4):
      q[cntr, :] = q[cntr, :].reshape((1, q.shape[1])) * factor

    return q

  @staticmethod
  def Axis2RotMat(Axis: np.array) -> np.array:
    Theta = np.linalg.norm(Axis)

    if (Theta == 0):
      return np.eye(3)

    Axis = np.clip(Axis / Theta, 0.0, 1.0)

    a = math.cos(Theta)
    la = 1 - a
    b = math.sin(Theta)
    m = Axis[0]
    n = Axis[1]
    p = Axis[2]

    R = np.array([
      [a + m ** 2 * la,    m * n * la - p * b, m * p *la + n * b ],
      [n * m * la + p * b, a + n ** 2 * la,    n * p * la - m * b],
      [p * m * la - n * b, p * n * la + m * b, a + p ** 2 * la   ]
    ])

    R = Utilities.to_real(R)

    return R


  @staticmethod
  def Axis2RotMatBatch(Axis: np.array) -> np.array:
    N = Axis.shape[1]
    R = np.zeros((9,N))

    for cntr in range(0, N):
      R[:, cntr] = np.array(Rotation.Axis2RotMat(Axis[:,cntr])).T.flatten()

    if np.sum(np.isnan(R.flatten())):
      print('Nan in rotation matrix batch')

    R = Utilities.to_real(R)

    return R


  @staticmethod
  def Quat2Axis(Q: np.array) -> np.array:
    N = Q.shape[1]
    Axis = np.zeros((3,N))

    for cntr in range(0, N):
      Axis[:, cntr] = Rotation.Quat2AxisSingle(Q[:, cntr])

    return Axis


  @staticmethod
  def Quat2AxisSingle(Q: np.array) -> np.array:
    Angle = np.real(2 * np.arccos(np.clip(np.abs(Q[0]), 0.0, 1.0)))
    Angle = Utilities.to_real(Angle)

    if Angle == 0:
      Axis = np.zeros((3,))
    else:
      Axis_norm = Q[1:4]/np.linalg.norm(Q[1:4])
      Axis = Angle*Axis_norm

    return Axis
  

  @staticmethod
  def rotate_structure_index(F: np.array, R: np.array):
    N = max(F.shape)

    U = (np.linspace(1, N, N) - (N + 1) / 2) / (N / 2)
    [z, x, y] = ComputeUtils.meshgrid_3d_single(U)

    x_size = x.size
    x_shape = x.shape

    Q = [x.reshape((1, x_size)), y.reshape((1, x_size)), z.reshape((1, x_size))]
    Q = np.vstack(Q)
    Q = np.matmul(R, Q)

    Qx = Q[0,:].reshape(x_shape)
    Qy = Q[1,:].reshape(x_shape)
    Qz = Q[2,:].reshape(x_shape)

    Protein = ComputeUtils.interp3(U, U, U, F, Qx, Qy, Qz)

    return Protein
  
  @staticmethod
  def rot_mat_to_axis(R: np.array) -> np.array:
    x = R[2, 1] - R[1, 2]
    y = R[0, 2] - R[2, 0]
    z = R[1, 0] - R[0, 1]

    r_2sin = np.linalg.norm(np.array([x,y,z]))

    if r_2sin != 0:
        Theta = np.arctan2(r_2sin, np.matrix.trace(R) - 1)
        Theta = Utilities.to_real(Theta)
        Axis = (Theta / r_2sin) * np.array([x, y, z]).reshape((3, 1))
    elif Utilities.are_almost_equal(R, np.eye(3), 1e-6):
        Axis = np.zeros((3, 1))
    else:
        Axis = np.zeros((3, 1))

    return Axis


  @staticmethod
  def axis_to_quat(Axis: np.array) -> np.array:
    if Axis.shape[0] != 3:
      print('Error in size of Axis.')
      return

    N = Axis.shape[1]
    Q = np.zeros((4, N))

    for cntr in range(N):
      Temp = Axis[:, cntr]
      Theta = np.linalg.norm(Temp)
      if Theta == 0:
        Q[:, cntr] = np.array([1, 0, 0, 0]).reshape((4,))
      else:
        Temp = ((math.sin(Theta / 2)) / Theta) * Temp
        Q[:, cntr] = np.array([math.cos(Theta/2), Temp[0], Temp[1], Temp[2]]).reshape((4,))

    return Q

  @staticmethod
  def rot_mat_to_quat(R: np.array) -> np.array:
    Axis = Rotation.rot_mat_to_axis(R)

    Q = Rotation.axis_to_quat(Axis)

    return Q


  @staticmethod
  def geodesic_distance_matrix(Q: np.array) -> np.array:
    if (len(Q.shape) != 2 or Q.shape[0] != 4):
      raise Exception('Quaternion shape was ' + str(Q.shape))
    
    Utilities.check(not np.iscomplex(Q).any(), 'Complex array was encountered.')

    return np.real(np.arccos(np.clip(np.matmul(Q.T, Q), 0, 1.0)))


  @staticmethod
  def mean_geodesic_distance_degrees(Q1: np.array, Q2: np.array) -> float:
    return (180 / math.pi) * Rotation.mean_geodesic_distance(Q1, Q2)


  @staticmethod
  def mean_geodesic_distance(Q1: np.array, Q2: np.array) -> float:
    Utilities.check(not np.iscomplex(Q1).any(), 'Complex array was encountered.')
    Utilities.check(not np.iscomplex(Q2).any(), 'Complex array was encountered.')
    Utilities.check(len(Q1.shape) == 2, 'Q1 shape')
    Utilities.check(len(Q2.shape) == 2, 'Q2 shape')
    Utilities.check(Q1.shape == Q2.shape, 'Q1 shape vs. Q2 shape')
    Utilities.check(Q2.shape[0] == 4, 'Q.shape[0]')
    N = Q1.shape[1]

    Q1_matrix = Rotation.geodesic_distance_matrix(Q1)
    Q2_matrix = Rotation.geodesic_distance_matrix(Q2)
    diff = Q1_matrix.flatten() - Q2_matrix.flatten()
    mean_q_distance = np.sum(np.abs(diff)) / (N * (N - 1))
    mean_geodesic_distance = 2 * mean_q_distance

    return mean_geodesic_distance


