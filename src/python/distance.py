import numpy as np
import numpy.matlib as matlib



"""
Distance matrix

  - Get Distances of projected images
  
"""
class DistanceMatrix:

  @staticmethod
  def knn(A: np.array, k: int) -> np.array:
    B = DistanceMatrix.convert_to_round_distance(A)

    return DistanceMatrix.knn_from_round_distance(B, k)

  @staticmethod
  def knn_from_round_distance(B: np.array, k: int) -> np.array:
    if (len(B.shape) != 2 or B.shape[0] != B.shape[1]):
      raise Exception('Non-square round-distance matrix was encountered.')

    N_orient = B.shape[0]
    S2 = np.random.randn(N_orient, k)
    N = np.random.randn(N_orient, k)

    for cntr in range(N_orient):
      II = np.argsort(B[cntr,:].reshape(1, N_orient), axis = 1).astype(int)[0, range(k)]
      N[cntr,:] = II
      S2[cntr,:] = B[cntr, II]

    return [S2, N]

  @staticmethod
  def convert_to_round_distance(A: np.array) -> np.array:
    A = DistanceMatrix.remove_offset(A)
    A = DistanceMatrix.set_norm_to_one(A)
    A = np.matmul(A, A.T)
    A = np.arccos(A)

    A[A != A] = 0
    A = (A + A.T) / 2

    return A

  @staticmethod
  def set_norm_to_one(A: np.array) -> np.array:
    return np.divide(A, matlib.repmat(np.sqrt(np.sum(A ** 2 , axis = 1)), A.shape[1], 1).T)

  @staticmethod
  def remove_offset(A: np.array) -> np.array:
    return A - matlib.repmat(np.mean(A, axis = 0), A.shape[0], 1)
