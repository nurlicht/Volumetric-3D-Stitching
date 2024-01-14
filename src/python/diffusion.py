import math
from matplotlib.pylab import pinv
import numpy as np
from scipy.sparse import csr_array, linalg
from scipy.optimize import least_squares
import numpy.matlib as matlib

from rotation import Rotation
from utils import Utilities


"""
Diffusion Coordinate

  - Convert distance matrix to the diffusion-coordinate
  - Find the linear transform mapping the dissuion coordinate to rotation matrix
"""
class DiffusionMapSolver:
  
  @staticmethod
  def solve(S2: np.array, N: np.array, R: np.array, parameters: {}) -> [np.array, np.array, np.array, np.array]:
    k = parameters['k']
    Epsilon = parameters['Epsilon']
    aPrioriFlag = parameters['aPrioriFlag']

    c0 = np.nan * np.ones((9, 9))
    c  = np.nan * np.ones((9, 9))

    [Ps, Lambda] = DiffusionMapSolver.diffusion_coordinate(S2, N, Epsilon)
    if Utilities.has_nan(Lambda):
      print('Problem in convergence of eigenvalues with Epsilon=' + str(Epsilon) + ' and k=' + str(k))
    elif np.iscomplexobj(Ps):
      print('Complex eigenvector found with Epsilon=' + str(Epsilon) + ' and k=' + str(k))
    else:
      [c, c0] = DiffusionMapSolver.dm_fit_all(Ps, R, aPrioriFlag)

    return [Ps, Lambda, c0, c]


  @staticmethod
  def diffusion_coordinate(S2: np.array, N: np.array, Epsilon: float) -> [np.array, np.array]:
    W = DiffusionMapSolver.s2_to_w_matrix(S2, N, Epsilon)
    W = DiffusionMapSolver.anisotropic_norm(W)
    [P_ep, D] = DiffusionMapSolver.dm_cov(W)

    [Lambda, Ps] = linalg.eigs(P_ep.T, k = 10, which = 'LM')

    Index = np.flip(np.argsort(Lambda)).astype(int)
    Lambda = Lambda[Index]
    Ps = Ps[:, Index]
    Ps = np.matmul(Ps, np.diag(Lambda))
    Ps = np.matmul(D, Ps)

    if (Utilities.are_almost_equal(np.imag(Ps), np.zeros(Ps.shape), 1e-6)):
      Ps = np.real(Ps)

    return [Ps,Lambda]


  @staticmethod
  def anisotropic_norm(W: np.array) -> np.array:
    N_orient = W.shape[0]
    Min = 1 / N_orient

    Q_Alpha = np.sum(W, axis = 0)
    Q_Alpha[Q_Alpha < Min] = Min
    Q_Alpha = csr_array(
      (np.divide(1, Q_Alpha).flatten(),
      (np.arange(N_orient).flatten(), np.arange(N_orient).flatten())),
      shape = (N_orient, N_orient)
    ).toarray()

    W = np.matmul(Q_Alpha, W)
    W = np.matmul(W, Q_Alpha)
    W = (W + W.T) / 2

    return W


  @staticmethod
  def dm_fit_all(Ps: np.array, Rotation_R: np.array, aPrioriFlag: bool) -> [np.array, np.array]:
    Ps = Ps[:, 1::]
    Rotation_R = Rotation_R.T

    if aPrioriFlag:
      c0 = np.linalg.lstsq(Ps, Rotation_R)[0]
      c = c0
    else:
      c_scale = 5
      options = DiffusionMapSolver.least_square_options(c_scale)
      c0_1D = options['x0'] / c_scale

      c_1D = least_squares(
        DiffusionMapSolver.or_func,
        x0 = options['x0'],
        method = options['method'],
        max_nfev = options['max_nfev'],
        ftol = options['ftol']
      )

      c0 = pinv(c0_1D.reshape((9, 9)))
      c  = pinv(c_1D.reshape((9, 9)))

    Recon_R = np.matmul(Ps, c)

    N = Ps.shape[0]
    shape_3_3 = (3, 3)
    for cntr in range(N):
      Temp_00 = DiffusionMapSolver.polar_decompose(Recon_R[cntr,:].reshape(shape_3_3))
      Recon_R[cntr,:] = Temp_00.reshape((1, Temp_00.size))

    Q_1 = np.random.rand(4, N) #Initializing "Eestimated" Quaternions of rotations
    Q_2 = np.random.rand(4, N) #Initializing "Known" Quaternions of rotations
    for cntr in range(N):
      r0 = Recon_R[cntr, :].reshape(shape_3_3)
      Q_1[: , cntr] = Rotation.rot_mat_to_quat(r0).reshape((4,))
      r0 = Rotation_R[cntr,:].reshape(shape_3_3)
      Q_2[: , cntr] = Rotation.rot_mat_to_quat(r0).reshape((4,))

    TempA = DiffusionMapSolver.geodesic_distance_matrix(Q_1.T, Q_1) #Pairwise geodesic distances
    TempB = DiffusionMapSolver.geodesic_distance_matrix(Q_2.T, Q_2) #Pairwise geodesic distances
    Sigma_T = (180 / math.pi) * 2 * np.sum(np.abs(TempA.flatten() - TempB.flatten())) / ( N * (N-1))

    print('Measure of error in Relative Orientations of All Pairs')
    print('Sigma_All_Pairs: '+ str(Sigma_T) + ' degrees')

    return [c, c0]


  @staticmethod
  def geodesic_distance_matrix(x: np.array, y: np.array) -> np.array:
    if (x.shape != y.T.shape):
      return np.array([np.nan])

    return np.arccos(np.clip(np.abs(np.matmul(x, y)), 0, 1.0))


  @staticmethod
  def polar_decompose(R: np.array) -> np.array:
    U, X ,V = np.linalg.svd(R)
    R_orth = np.matmul(U, V.T)
    return R_orth


  # Setting the nonlinear optimization parameters
  @staticmethod
  def least_square_options(c_scale: float) -> {}:
    return {
      'c_scale': c_scale,
      'x0': c_scale * np.random.rand((81, 1)),
      'method': 'trf',
      'max_nfev': 250,
      'ftol': 1e-20 * c_scale,
      'max_nfev': 1e8,
      'DiffMaxChange': 1e5 * c_scale,
      'LB': [],
      'UB': [],
      'Display': 'final'
    }


  @staticmethod
  def dm_cov(W: np.array) -> [np.array, np.array]:
    N_orient = W.shape[0]
    Min = 1 / N_orient

    D = np.sum(W, axis = 0)
    D[D < Min] = Min
    D = csr_array(
      (np.divide(1, np.sqrt(D)).flatten(),
      (np.arange(N_orient).flatten(), np.arange(N_orient).flatten())),
      shape=(N_orient, N_orient)
    ).toarray()

    W = np.matmul(D, W)
    W = np.matmul(W, D)
    W = (W + W.T) / 2

    return [W, D]


  # Imposing the rotation matrix constraints to find {c}
  @staticmethod
  def or_func(c: np.array, PsMod: np.array) -> np.array:
    r = PsMod.shape[1]
    r_MP5 = 1 / math.sqrt(r)
    N_c = 9

    c_Temp = c.reshape((N_c, N_c))

    Temp = np.zeros((N_c, N_c))
    Temp2 = np.random.randn((N_c ** 2, 1))
    I = np.eye(3)

    G_Functional = np.zeros((r, 1))
    R_Big = np.matmul(c_Temp, PsMod)

    for cntr_l in range(r):
      R = R_Big[:, cntr_l].reshape((3, 3))
      Temp = np.matmul(R.T, R) - I
      Temp2 = Temp.reshape((Temp.size, 1))
      G_Functional[cntr_l] = math.sqrt(np.matmul(Temp2.T, Temp2) + math.abs(np.linalg.det(R) - 1)) #L2

    G_Functional = np.matmul(np.sqrt(G_Functional), r_MP5)

    return G_Functional


  @staticmethod
  def s2_to_w_matrix(S2: np.array, N: np.array, Epsilon: float) -> np.array:
    N_orient , d = S2.shape
    S2_Max = np.median(S2[: , 4])

    Index = 0
    Dist_Max = np.min(S2[: , d - 1])
    while (np.max(S2[: , Index]) < Dist_Max):
      Index = Index + 1

    Dist_Thr = np.max(S2[: , Index - 1])
    S2[S2 > Dist_Thr] = np.inf

    Epsilon = Epsilon * S2_Max

    W = np.arange(0, N_orient).reshape((N_orient, 1))
    W = matlib.repmat(W, 1, d)

    W = csr_array(
      (np.exp(- S2.astype(float).flatten() / Epsilon),
       (W.flatten(), N.astype(float).flatten())),
       shape=(N_orient, N_orient)
    ).toarray()

    return W
   
