import unittest

from plot import Plotter


class PlotterTest(unittest.TestCase):

  def test_demo(self: unittest.TestCase) -> None:
    Plotter.demo(show = False)


if __name__ == '__main__':
  unittest.main()
