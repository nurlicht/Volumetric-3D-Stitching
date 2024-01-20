
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Plotter:
    
  @staticmethod
  def plot(array_map: dict[str, np.array], show: bool) -> [[str, str, Figure]]:
    print('Plotter.plot started.')

    figures = [
      [Plotter.plot_eigen_values_vectors(array_map['Ps'], array_map['Lambda'])],

      Plotter.plot_vectors_2d_3d(array_map['Ps'], 'Ps', array_map['Axis'][:, 0], 'Axis_0', 1),
      Plotter.plot_vectors_2d_3d(array_map['Ps'], 'Ps', array_map['Axis'][:, 1], 'Axis_1', 1),
      Plotter.plot_vectors_2d_3d(array_map['Ps'], 'Ps', array_map['Axis'][:, 2], 'Axis_2', 1),

      Plotter.plot_vectors_2d_3d(array_map['Ps'], 'Ps', array_map['Quat'][:, 0], 'Quat_0', 1),
      Plotter.plot_vectors_2d_3d(array_map['Ps'], 'Ps', array_map['Quat'][:, 1], 'Quat_1', 1),
      Plotter.plot_vectors_2d_3d(array_map['Ps'], 'Ps', array_map['Quat'][:, 2], 'Quat_2', 1),
      Plotter.plot_vectors_2d_3d(array_map['Ps'], 'Ps', array_map['Quat'][:, 3], 'Quat_3', 1),

      Plotter.plot_vectors_2d_3d(array_map['Axis'], 'Axis', array_map['Ps'][:, 1], 'Ps_1', 0),
      Plotter.plot_vectors_2d_3d(array_map['Axis'], 'Axis', array_map['Ps'][:, 2], 'Ps_2', 0),
      Plotter.plot_vectors_2d_3d(array_map['Axis'], 'Axis', array_map['Ps'][:, 3], 'Ps_3', 0),
      Plotter.plot_vectors_2d_3d(array_map['Axis'], 'Axis', array_map['Ps'][:, 4], 'Ps_4', 0),

      Plotter.plot_vectors_2d_3d(array_map['Quat'], 'Quat', array_map['Ps'][:, 1], 'Ps_1', 0),
      Plotter.plot_vectors_2d_3d(array_map['Quat'], 'Quat', array_map['Ps'][:, 2], 'Ps_2', 0),
      Plotter.plot_vectors_2d_3d(array_map['Quat'], 'Quat', array_map['Ps'][:, 3], 'Ps_3', 0),
      Plotter.plot_vectors_2d_3d(array_map['Quat'], 'Quat', array_map['Ps'][:, 4], 'Ps_4', 0),

      Plotter.plot_vectors_2d_3d(array_map['Quat'], 'Quat', array_map['Ps'][:, 1], 'Ps_1', 1),
      Plotter.plot_vectors_2d_3d(array_map['Quat'], 'Quat', array_map['Ps'][:, 2], 'Ps_2', 1),
      Plotter.plot_vectors_2d_3d(array_map['Quat'], 'Quat', array_map['Ps'][:, 3], 'Ps_3', 1),
      Plotter.plot_vectors_2d_3d(array_map['Quat'], 'Quat', array_map['Ps'][:, 4], 'Ps_4', 1),
    ]

    if (show):
      plt.show()

    plt.close()

    print('Plotter.plot ended.')

    return figures


  @staticmethod
  def plot_eigen_values_vectors(Ps: np.array, Lambda: np.array, show: bool = False) -> [str, str, Figure]:
    fig, (ax_vector, ax_value) = plt.subplots(1, 2)

    plt.xlim(0, 10)
    pos_vector = ax_vector.imshow(Ps)
    ax_vector.set_box_aspect(0.10 * Ps.shape[0] / Ps.shape[1])
    fig.colorbar(pos_vector, ax = ax_vector, extend = 'both')
    ax_vector.set_title('Diffusion map eigenfunctions Ps')

    plt.xlim(0, 10)
    ax_value.plot(Lambda,'-*')
    ax_vector.set_title('Diffusion map eigenvalues')

    if (show):
      plt.show()

    return ['eigenvectors', 'eigenvalues', fig]


  @staticmethod
  def plot_vectors_2d_3d(
    x: np.array,
    x_label: str,
    color: np.array,
    color_label: str,
    x_index_offset: int = 0
  ) -> [[str, str, Figure]]:
    return [
      Plotter.plot_vectors_2d(x, x_label, color, color_label, x_index_offset),
      Plotter.plot_vectors_3d(x, x_label, color, color_label, x_index_offset)
    ]

  @staticmethod
  def plot_vectors_3d(
    x: np.array,
    x_label: str,
    color: np.array,
    color_label: str,
    x_index_offset: int = 0
  ) -> [str, str, Figure]:
    x_label_vector = [x_label + '_' + str(x_index_offset + i) for i in range(3)]
    
    fig = plt.figure()
    ax: Axes = fig.add_subplot(projection='3d')

    sc = ax.scatter(
      x[:, x_index_offset],
      x[:, x_index_offset + 1],
      x[:, x_index_offset + 2],
      c = color,
      cmap = 'hsv',
      alpha = 0.75
    )
    fig.colorbar(sc, ax = ax, extend = 'both')

    ax.set_title(
      x_label_vector[0] +
      ' vs. ' +
      x_label_vector[1] +
      ' vs. ' +
      x_label_vector[2] +
      ' color-coded with ' +
      color_label
    )
    ax.axis('equal')
    ax.set_xlabel(x_label_vector[0])
    ax.set_ylabel(x_label_vector[1])
    ax.set_zlabel(x_label_vector[2])

    return [x_label, color_label, fig]


  @staticmethod
  def plot_vectors_2d(
    x: np.array,
    x_label: str,
    color: np.array,
    color_label: str,
    x_index_offset: int = 0
  ) -> [str, str, Figure]:
    if (len(x.shape) != 2):
      raise Exception('2-dimensional array ' + x_label + ' was expected. shape = ' + str(x.shape))
    if (x.shape[1] < 3 + x_index_offset):
      raise Exception('Another shape of array ' + x_label + ' was expected. shape = ' + str(x.shape))
    if (len(color.shape) != 1):
      raise Exception('1-dimensional array ' + color_label + ' was expected. shape = ' + str(color.shape))
    if (color.shape[0] < x.shape[0]):
      raise Exception('Compatible shapes of arrays was expected. shapes = ' + str(x.shape) + ' and ' + str(color.shape))

    index_pairs = np.array([(0, 1), (1, 2), (0, 2)]) + x_index_offset

    fig, axs = plt.subplots(1, 3)

    n_plots = min(3, index_pairs.shape[0])
    for i in range(n_plots):
      index_pair = index_pairs[i, :]
      ax = axs[i]
      sc = ax.scatter(
        x[:, index_pair[0]],
        x[:, index_pair[1]],
        c = color,
        cmap = 'hsv',
        alpha = 0.75
      )
      fig.colorbar(sc, ax = ax, extend = 'both')
      ax.set_title(
        x_label + '_' + str(index_pair[0]) +
        ' vs. ' +
        x_label + '_' + str(index_pair[1]) +
        ' color-coded with ' + color_label
      )
      ax.set_xlabel(x_label + '_' + str(index_pair[0]))
      ax.set_ylabel(x_label + '_' + str(index_pair[1]))
      ax.axis('equal')

    return [x_label, color_label, fig]


  @staticmethod
  def demo(show: bool = True) -> None:
    n_samples = 1000
    dim = 10
    Ps = 2 + 5 * np.random.randn(n_samples, dim)
    Lambda = np.arange(dim)
    X = np.random.randn(n_samples, dim) * 2 + 3
    Ps[:, 0] = 0
    Plotter.plot_eigen_values_vectors(Ps, Lambda)
    Plotter.plot_vectors_2d_3d(Ps, 'Ps', X[:, 1].reshape(n_samples,), 'X', 1)
    Plotter.plot_vectors_2d_3d(X, 'X', Ps[:, 2].reshape(n_samples,), 'Ps', 1)
    if (show):
      plt.show()


if __name__ == '__main__':
  Plotter.demo()

