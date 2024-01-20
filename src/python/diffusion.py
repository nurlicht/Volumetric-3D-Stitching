import math
from matplotlib.pylab import pinv
import numpy as np
from scipy.sparse import csr_array, linalg
from scipy.optimize import least_squares
import numpy.matlib as matlib
from compute import ComputeUtils
from plot import Plotter

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
    print('DiffusionMapSolver.solve started.')
    k = parameters['k']
    Epsilon = parameters['Epsilon']
    aPrioriFlag = parameters['aPrioriFlag']

    [Ps, Lambda] = DiffusionMapSolver.diffusion_coordinate(S2, N, Epsilon)
    if (parameters['plot']):
      Plotter.plot_eigen_values_vectors(Ps, Lambda, show = True)
    [c, c0, Sigma_T] = DiffusionMapSolver.dm_fit_all(Ps, R, aPrioriFlag)
    print('DiffusionMapSolver.solve ended.')
    return [Ps, Lambda, c0, c, Sigma_T]


  @staticmethod
  def diffusion_coordinate(
    S2: np.array,
    N: np.array,
    Epsilon: float,
    k: int = 10,
    validate: bool = True
  ) -> [np.array, np.array]:
    print('DiffusionMapSolver.diffusion_coordinate started.')

    W = DiffusionMapSolver.s2_to_w_matrix(S2, N, Epsilon)
    Utilities.check_nan_complex(W, 'W')
    W = DiffusionMapSolver.anisotropic_norm(W)
    Utilities.check_nan_complex(W, 'W')
    [P_ep, D] = DiffusionMapSolver.dm_cov(W)
    Utilities.check_nan_complex(P_ep, 'P_ep')
    Utilities.check_nan_complex(D, 'D')

    print('DiffusionMapSolver.diffusion_coordinate np.linalg.eigs started.')
    v0 = (1 - 5e-4) * np.arange(P_ep.shape[0])
    #P_ep(N_orient, N_orient), P_s(N_orient, 10), Lambda(10, 1) 
    #[Lambda, Ps] = linalg.eigs(P_ep, k = 10, which = 'LM', v0 = v0, maxiter = 1e6)
    if (k >= P_ep.shape[0] - 1):
      raise Exception('No spare-matrix for eigs-calculation')
    [Lambda, Ps] = linalg.eigs(P_ep, k)
    Ps = - Ps
    if validate:
      Utilities.check(Ps.shape[0] >= k, 'Too few orientations')
    print('DiffusionMapSolver.diffusion_coordinate np.linalg.eigs ended.')

    Ps = Utilities.to_real(Ps)
    Lambda = Utilities.to_real(Lambda)

    if validate:
      ComputeUtils.validate_eigen_values_vectors(P_ep, Lambda, Ps)
      Lambda = np.clip(Lambda, 0.0, 1.0)

    Index = np.flip(np.argsort(Lambda)).astype(int)
    Lambda = Lambda[Index]
    Ps = Ps[:, Index]

    for i in range(k):
      Ps[:, i] *= Lambda[i]

    Ps = np.matmul(D, Ps)

    print('DiffusionMapSolver.diffusion_coordinate ended.')

    return [Ps,Lambda]


  @staticmethod
  def anisotropic_norm(W: np.array) -> np.array:
    N_orient = W.shape[0]
    Min = 1 / N_orient

    Q_Alpha = np.sum(W, axis = 1)
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
    print('DiffusionMapSolver.dm_fit_all started.')

    Utilities.check(not np.iscomplex(Ps).any(), 'Ps is complex')
    Utilities.check(not np.iscomplex(Rotation_R).any(), 'Rotation_R is complex')

    Ps = Ps[:, 1::]
    Rotation_R = Rotation_R.T

    if aPrioriFlag:
      print('DiffusionMapSolver.dm_fit_all np.linalg.lstsq started.')
      #[c0, residuals, rank, s] = np.linalg.lstsq(Ps, Rotation_R, rcond = None)
      c0 = ComputeUtils.lsq_linear(Ps , Rotation_R)
      #ComputeUtils.validate_lsq_linear(Ps, Rotation_R, c0, 1e-14)
      #DiffusionMapSolver.validate_lstsq_results(c0, residuals, rank, s, k, Rotation_R, Ps)
      c = c0
      print('DiffusionMapSolver.dm_fit_all np.linalg.lstsq ended.')
    else:
      c_scale = 5
      options = DiffusionMapSolver.least_square_options(c_scale)
      c0_1D = options['x0'] / c_scale

      print('DiffusionMapSolver.dm_fit_all np.linalg.least_squares started.')
      c_1D = least_squares(
        DiffusionMapSolver.or_func,
        x0 = options['x0'],
        method = options['method'],
        max_nfev = options['max_nfev'],
        ftol = options['ftol']
      )
      print('DiffusionMapSolver.dm_fit_all np.linalg.least_squares ended.')

      c0 = pinv(c0_1D.reshape((9, 9)))
      c  = pinv(c_1D.reshape((9, 9)))

    Recon_R = np.matmul(Ps, c)
    if (np.iscomplex(Ps).any()):
      raise Exception('Ps is complex')
    if (np.iscomplex(c).any()):
      raise Exception('c is complex')
    if (np.iscomplex(Recon_R).any()):
      raise Exception('Recon_R is complex')

    N = Ps.shape[0]
    shape_3_3 = (3, 3)
    for cntr in range(N):
      Temp_00 = ComputeUtils.polar_decompose(Recon_R[cntr,:].reshape(shape_3_3))
      Recon_R[cntr,:] = Temp_00.reshape((1, Temp_00.size))

    if (np.iscomplex(Recon_R).any()):
      raise Exception('Recon_R is complex')

    Q_1 = np.random.rand(4, N) #Initializing "Eestimated" Quaternions of rotations
    Q_2 = np.random.rand(4, N) #Initializing "Known" Quaternions of rotations
    for cntr in range(N):
      r0 = Recon_R[cntr, :].reshape(shape_3_3)
      Q_1[: , cntr] = Rotation.rot_mat_to_quat(r0).reshape((4,))
      r0 = Rotation_R[cntr,:].reshape(shape_3_3)
      Q_2[: , cntr] = Rotation.rot_mat_to_quat(r0).reshape((4,))

    if (np.iscomplex(Q_1).any()):
      raise Exception('Q_1 is complex')
    if (np.iscomplex(Q_2).any()):
      raise Exception('Q_2 is complex')

    Sigma_T = Rotation.mean_geodesic_distance_degrees(Q_1, Q_2)
    if (Sigma_T > 60):
      raise Exception ('Sigma_T = ' + str(Sigma_T))

    print('Measure of error in Relative Orientations of All Pairs')
    print('Sigma_All_Pairs: '+ str(Sigma_T) + ' degrees')

    print('DiffusionMapSolver.dm_fit_all ended.')

    return [c, c0, Sigma_T]
  
  @staticmethod
  def validate_lstsq_results(
    c0: np.array,
    residuals: np.array,
    rank: int,
    s: np.array,
    k: int,
    Rotation_R: np.array,
    Ps: np.array
  ) -> None:
    Utilities.check(len(residuals) != 0, 'residuals = []')
    Utilities.check(rank == k, 'rank != ' + str(k))
    Utilities.check(c0.shape == (k, k), 'c0.shape != ' + str((k, k)))
    Utilities.check(not np.iscomplex(s).any(), 'Complex singular values were encountered: ' + str(s))
    Utilities.check(not np.iscomplex(c0).any(), 'Complex lsq-matrix was encountered: ' + str(s))
    Utilities.check(np.min(s) > 0, 'Non-positive singular values were encountered: ' + str(s))

    print('Singular values       = ' + str(s))

    print('Residuals             = ' + str(residuals))
    print('Residuals: max        = ' + str(max(residuals)) + ', median = ' + str(np.median(residuals)))

    b_rms = np.sum(Rotation_R ** 2, axis = 0)
    ax_rms = np.sum((np.matmul(Ps, c0)) ** 2, axis = 0)
    residualsNorm1 = np.divide(residuals, b_rms)
    if (np.max(residualsNorm1) > 1 + 1e-6):
      raise Exception ('residualsNorm1.max = ' + str(np.max(residualsNorm1)))

    residualsNorm2 = np.divide(residuals, ax_rms)
    residualsNorm3 = np.divide(residuals, 0.5 * (b_rms + ax_rms))

    print('ax_rms                = ' + str(ax_rms))
    print('b_rms                 = ' + str(b_rms))
    print('ResidualsNorm1        = ' + str(residualsNorm1))
    print('ResidualsNorm1: max   = ' + str(max(residualsNorm1)) + ', median = ' + str(np.median(residualsNorm1)))
    print('ResidualsNorm2        = ' + str(residualsNorm2))
    print('ResidualsNorm2: max   = ' + str(max(residualsNorm2)) + ', median = ' + str(np.median(residualsNorm2)))
    print('ResidualsNorm3        = ' + str(residualsNorm3))
    print('ResidualsNorm3: max   = ' + str(max(residualsNorm3)) + ', median = ' + str(np.median(residualsNorm3)))


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
    D = Utilities.to_real(csr_array(
      (np.divide(1, np.sqrt(D)).flatten(),
      (np.arange(N_orient).flatten(), np.arange(N_orient).flatten())),
      shape=(N_orient, N_orient)
    ).toarray())

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
  def s2_to_w_matrix(
    S2: np.array,
    N: np.array,
    Epsilon: float,
    s2_median_index: int = 4,
    index_offset: int = 1
  ) -> np.array:
    N_orient, d = S2.shape
    S2_Max = np.median(S2[min(s2_median_index, d - 1), :])
    if (S2_Max == 0):
      S2_Max = np.max(S2)
    Utilities.check(S2_Max != 0, 'S2_Max = 0')

    Index = 0
    Dist_Max = np.min(S2[d - 1, :])
    while (np.max(S2[Index, :]) < Dist_Max):
      Index = Index + 1

    #Dist_Thr = np.max(S2[: , Index - 1]) if Index > 0 else np.inf
    Dist_Thr = np.max(S2[Index - index_offset, :])
    S2[S2 > Dist_Thr] = np.inf

    Epsilon = Epsilon * S2_Max

    W = np.arange(0, N_orient, dtype = float).reshape((N_orient, 1))
    W = matlib.repmat(W, 1, d)
    W = csr_array(
      (np.exp(- S2.astype(float).flatten() / Epsilon),
       (W.flatten(), N.flatten())),
       shape=(N_orient, N_orient)
    ).toarray()

    W = Utilities.to_real(W)

    return W
   
