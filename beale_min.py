from scipy.optimize import minimize
from numpy.random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def beale(x):
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
x0s = []
for i in range(0, 30):
    x0 = (random(2)-1)*20
    x0s.append(x0)


iters = []
feval = []
sol = []
objective = []
for i in range(0, 30):
    output = minimize(beale, x0s[i], method=_minimize_dfp, options= {'disp': True})
    iters.append(output.nit)
    feval.append(output.nfev)
    sol.append(output.x)
    objective.append(output.fun) 

#####################################
delta = 0.05
s = 0.05
X = np.arange(-3, 3, delta)
Y = np.arange(-3, 3, delta)
X, Y = np.meshgrid(X, Y)
Z = beale([X, Y])
levels = np.arange(10, 300, 10)
plt.contour(X, Y, Z, levels=levels, norm=LogNorm())
# plt.contour(X, Y, Z, levels=[0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50 , 100])

