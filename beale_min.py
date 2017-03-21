from scipy.optimize import minimize
from numpy.random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def beale( m, x ):
  f1 = 1.5   - x[0] * ( 1.0 - x[1]      )
  f2 = 2.25  - x[0] * ( 1.0 - x[1] ** 2 )
  f3 = 2.625 - x[0] * ( 1.0 - x[1] ** 3 )
  f = f1 ** 2 + f2 ** 2 + f3 ** 2
  return f

# Plot the function
fig = plt.figure()
ax = Axes3D(fig, azim = -128, elev = 43)
s = .1
X = np.arange(-5, 5.+s, s)
Y = np.arange(-5, 5.+s, s)
X, Y = np.meshgrid(X, Y)
Z = beale(2, [X, Y])
#ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = cm.jet, linewidth=0, edgecolor='none')
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm=LogNorm(), cmap = cm.jet, linewidth=0, edgecolor='none')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Beale's")

plt.savefig(beale.png)
#########################################
