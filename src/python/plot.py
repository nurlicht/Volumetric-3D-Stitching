
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np


class Plotter:
    
  @staticmethod
  def plot(arrays: [np.array], titles: [str]) -> [[Figure, Axes]]:
    if (len(arrays) != len(titles)):
      raise Exception('Unpaired arrays and titles were encountered.')
    
    figures = []
    for i in range(len(arrays)):
      arr = arrays[i]
      title = titles[i]

      fig = plt.figure(title)

      ax = fig.gca()
      ax.set_title(title)

      figures.append([fig, ax])
      
      if (len(arr.shape) < 2 or (arr.shape[0] == 1 or arr.shape[1] == 1)):
        ax.scatter(range(len(arr)), arr)
      elif (len(arr.shape) == 2 and (arr.shape[0] > 1 and arr.shape[1] > 1)):
        ax.imshow(arr)
      else:
        raise Exception('Unknown arry-type was encountered.')

    plt.show()

    return figures
