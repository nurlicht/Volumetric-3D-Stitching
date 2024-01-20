import numpy as np
import numpy.matlib as matlib

from utils import Utilities



"""
Distance matrix

  - Get Distances of projected images
  
"""
class DistanceMatrix:

  @staticmethod
  def knn(A: np.array, k: int) -> np.array:
    print('DistanceMatrix.knn started.')
    B = DistanceMatrix.convert_to_round_distance(A)
    B = DistanceMatrix.knn_from_round_distance(B, k)
    print('DistanceMatrix.knn ended.')
    return B


  @staticmethod
  def knn_from_round_distance(B: np.array, k: int) -> np.array:
    print('DistanceMatrix.knn_from_round_distance started.')

    if (len(B.shape) != 2 or B.shape[0] != B.shape[1]):
      raise Exception('Non-square round-distance matrix was encountered.')

    N_orient = B.shape[0]
    if (k > N_orient):
      raise Exception('Number of nearest neighbors to preserve is too high.')

    S2 = np.random.randn(N_orient, k)
    N = np.random.randn(N_orient, k)

    for cntr in range(N_orient):
      II = np.argsort(B[cntr,:].reshape(1, N_orient), axis = 1).astype(int)[0, range(k)]
      N[cntr,:] = II
      S2[cntr,:] = B[cntr, II]
      Utilities.log_progress('DistanceMatrix.knn_from_round_distance', cntr, N_orient)

    print('')
    print('DistanceMatrix.knn_from_round_distance ended.')

    return [S2, N]

  @staticmethod
  def convert_to_round_distance(A: np.array) -> np.array:
    print('DistanceMatrix.convert_to_round_distance started.')

    A = DistanceMatrix.remove_offset(A)
    A = DistanceMatrix.set_norm_to_one(A)
    A = np.arccos(np.clip(np.matmul(A, A.T), -1, 1))

    A[A != A] = 0
    A = (A + A.T) / 2

    print('DistanceMatrix.convert_to_round_distance ended.')

    return A

  @staticmethod
  def set_norm_to_one(A: np.array) -> np.array:
    return np.divide(A, matlib.repmat(np.sqrt(np.sum(A ** 2 , axis = 1)), A.shape[1], 1).T)

  @staticmethod
  def remove_offset(A: np.array) -> np.array:
    return A - matlib.repmat(np.mean(A, axis = 0), A.shape[0], 1)
