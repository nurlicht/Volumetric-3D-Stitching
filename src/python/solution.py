
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from diffusion import DiffusionMapSolver
from distance import DistanceMatrix
from plot import Plotter
from projection import Projector
from utils import Utilities


"""
Main Logic

  - Get and validate (and if needed, generate default) parameters
  - Generate rotation matrices
  - Get (Fourier) projected images corresponding to rotation matrices
  - Get Distances of projected images
  - Convert distance matrix to the diffusion-coordinate
  - Find the linear transform mapping the dissuion coordinate to rotation matrix
  
"""
class OrientationRecoverySolver:

  @staticmethod
  def solve_and_plot(parameters: {}) -> [[Figure, Axes]]:
    array_map = OrientationRecoverySolver.solve(parameters)
    show_plots: bool = (not 'plot' in parameters.keys()) or parameters['plot']
    figures_axes: [[Figure, Axes]] = Plotter.plot(array_map, show_plots)
    return figures_axes


  @staticmethod
  def solve(parameters: {}) -> dict[str, np.array]:
    print('OrientationRecoverySolver.solve started.')
    if (not OrientationRecoverySolver.validate_parameters(parameters)):
       parameters = OrientationRecoverySolver.default_parameters()
    if (not OrientationRecoverySolver.validate_parameters(parameters)):
      raise Exception('Default parameters are invalid.')

    [Images, R, Axis, Quat] = Projector.generate(parameters['nRot1D'], parameters['n_voxel_1d'])

    [S2, N] = DistanceMatrix.knn(Images, parameters['k'])

    [Ps, Lambda, c0, c, Sigma_T] = DiffusionMapSolver.solve(S2, N, R, parameters)

    print('OrientationRecoverySolver.solve ended.')

    return {
      'Ps': Utilities.to_real(Ps),
      'Lambda': Utilities.to_real(Lambda),
      'c0': Utilities.to_real(c0),
      'c': Utilities.to_real(c),
      'S2': Utilities.to_real(S2),
      'N': Utilities.to_real(N),
      'Images': Utilities.to_real(Images),
      'R': Utilities.to_real(R),
      'Sigma_T': np.array([Sigma_T]),
      'Axis': Utilities.to_real(Axis.T),
      'Quat': Utilities.to_real(Quat.T)      
    }


  @staticmethod
  def validate_parameters(parameters: {}) -> bool:
    return (
      isinstance(parameters, dict) and
      list(parameters.keys()) == ['n_voxel_1d', 'nRot1D', 'k', 'Epsilon', 'plot', 'aPrioriFlag']      and
      isinstance(parameters['n_voxel_1d'], int)   and parameters['n_voxel_1d'] > 0                    and
      isinstance(parameters['k'], int)            and parameters['k'] > 8                             and
      isinstance(parameters['nRot1D'], int)       and parameters['nRot1D'] ** 3 - 1 > parameters['k'] and
      isinstance(parameters['Epsilon'], float)    and parameters['Epsilon'] > 0                       and
      isinstance(parameters['plot'], bool)        and
      isinstance(parameters['aPrioriFlag'], bool)
    )


  @staticmethod
  def default_parameters() -> {}:
     return {
      'n_voxel_1d': 7, #31,15,7 
      'nRot1D': 8, #28,14,6
      'k': 24, #150,100,20
      'Epsilon': 0.7,
      'plot': True,
      'aPrioriFlag': True
    }
