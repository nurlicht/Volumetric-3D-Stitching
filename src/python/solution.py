
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from diffusion import DiffusionMapSolver
from distance import DistanceMatrix
from plot import Plotter
from projection import Projector


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
    [arrays, titles] = OrientationRecoverySolver.solve(parameters)
    figures_axes: [[Figure, Axes]] = Plotter.plot(arrays, titles)
    return figures_axes


  @staticmethod
  def solve(parameters: {}) -> [[np.array], [str]]:
    if (not OrientationRecoverySolver.validate_parameters(parameters)):
       parameters = OrientationRecoverySolver.default_parameters()
    if (not OrientationRecoverySolver.validate_parameters(parameters)):
      raise Exception('Default parameters are invalid.')

    [Images, R] = Projector.generate(parameters['nRot1D'])

    [S2, N] = DistanceMatrix.knn(Images, parameters['k'])

    [Ps, Lambda, c0, c] = DiffusionMapSolver.solve(S2, N, R, parameters)

    print('k=' + str(parameters['k']) + ', Epsilon=' + str(parameters['Epsilon']))

    return [
      [ Ps,   Lambda,   c0,   c,   S2,   N,   Images,   R],
      ['Ps', 'Lambda', 'c0', 'c', 'S2', 'N', 'Images', 'R']
    ]
  

  @staticmethod
  def validate_parameters(parameters: {}) -> bool:
    keys = list(parameters.keys())

    return (
      isinstance(parameters, dict) and
      list(parameters.keys()) == ['nRot1D', 'k', 'Epsilon', 'aPrioriFlag'] and
      isinstance(parameters['nRot1D'], int)       and parameters['nRot1D'] > 0  and
      isinstance(parameters['k'], int)            and parameters['k'] > 0       and
      isinstance(parameters['Epsilon'], float)    and parameters['Epsilon'] > 0 and
      isinstance(parameters['aPrioriFlag'], bool)
    )


  @staticmethod
  def default_parameters() -> {}:
     return {
      'nRot1D': 6,
      'k': 150,
      'Epsilon': 0.7,
      'aPrioriFlag': True
    }
