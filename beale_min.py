from scipy.optimize import minimize
from numpy.random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import pyplot
import timeit

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
times= []
for i in range(0, 30):
    start_time = timeit.default_timer()
    output = minimize(beale, x0s[i], method='L-BFGS-B', options= {'disp': True})
    times.append(timeit.default_timer() - start_time)
    iters.append(output.nit)
    feval.append(output.nfev)
    sol.append(output.x)
    objective.append(output.fun) 

#####################################
delta = 0.05
s = 0.05
X = np.arange(-3, 5, delta)
Y = np.arange(-3, 3, delta)
X, Y = np.meshgrid(X, Y)
Z = beale([X, Y])
levels = np.arange(10, 300, 10)
#plt.contour(X, Y, Z, levels=levels, norm=LogNorm())
plt.contour(X, Y, Z, levels=[0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 60, 63, 66, 70, 75, 80, 100])
plt.title('Isolines')
plt.xlabel('X1')
plt.ylabel('X2')


xs = []
ys = []

def bstop(xk):
     xs.append(np.copy(xk))
     ys.append(beale(xk))

xs = [np.array([-1, -1])]
ys = [beale(xs[0])]
minimize(beale, [-1, -1], method='BFGS', callback=bstop, options= {'disp': True})

linex = [-1]
liney = [-1]
for i in xs:
    linex.append(i[0])
    liney.append(i[1])

bfgs_y = list(ys)
bfgs, = plt.plot(linex, liney, '-o', label='BFGS')

xs = [np.array([-1, -1])]
ys = [beale(xs[0])]
minimize(beale, [-1, -1], method='L-BFGS-B', callback=bstop, options= {'disp': True})
linex = [-1]
liney = [-1]
for i in xs:
    linex.append(i[0])
    liney.append(i[1])

lbfgsb_y = list(ys)
lbfgsb, = plt.plot(linex, liney, '-s', label='L-BFGS-B')


xs = [
np.array([-1, -1]),
np.array([0, -2.076923e-01]),
np.array([1.101268e+00, -9.677930e-01]),
np.array([8.970397e-01, -5.260371e-01]),
np.array([1.085339e+00, -5.058077e-01]),
np.array([1.832440e+00, -2.907016e-01]),
np.array([2.198566e+00, -5.155961e-02]),
np.array([2.692337e+00, 3.684094e-01]),
np.array([2.789503e+00, 4.511403e-01]),
np.array([2.795133e+00, 4.487888e-01]),
np.array([2.818547e+00, 4.483392e-01]),
np.array([2.840796e+00, 4.519267e-01]),
np.array([2.885289e+00, 4.612113e-01]),
np.array([2.923265e+00, 4.707860e-01]),
np.array([2.980495e+00, 4.865466e-01]),
np.array([3.024381e+00, 4.997452e-01]),
np.array([3.043476e+00, 5.064746e-01]),
np.array([3.047318e+00, 5.090894e-01]),
np.array([3.042225e+00, 5.097113e-01]),
np.array([3.030713e+00, 5.080590e-01]),
np.array([3.016008e+00, 5.050824e-01]),
np.array([3.006359e+00, 5.026518e-01]),
np.array([2.999553e+00, 5.005949e-01]),
np.array([2.997714e+00, 4.997436e-01]),
np.array([2.998416e+00, 4.996591e-01]),
np.array([2.999443e+00, 4.998514e-01]),
np.array([2.999928e+00, 4.999741e-01]),
np.array([3.000001e+00, 4.999987e-01])
]



ys = [
3.870312e+01,
1.420312e+01,
5.474402e+00,
5.132615e+00,
4.056161e+00,
1.634935e+00,
8.440893e-01,
5.062609e-02,
1.015695e-02,
8.785395e-03,
6.671388e-03,
5.511229e-03,
3.959797e-03,
2.900633e-03,
1.691332e-03,
1.011259e-03,
6.995383e-04,
4.831696e-04,
2.805545e-04,
1.529509e-04,
7.094062e-05,
3.357103e-05,
1.152309e-05,
3.063215e-06,
4.650093e-07,
5.222919e-08,
2.294078e-09,
4.511352e-11,
1.837179e-13
]


linex = []
liney = []
for i in xs:
    linex.append(i[0])
    liney.append(i[1])


powell_y = list(ys)
powell, = plt.plot(linex, liney, '-^', label='DFP')

plt.legend(handles=[bfgs, lbfgsb, powell])

plt.title('Isolines')
plt.xlabel('x1')
plt.ylabel('x2')
plt.figure()
b, = plt.plot(bfgs_y, '-o', label='BFGS')
l, = plt.plot(lbfgsb_y, '-s', label='L-BFGS-B')
p, = plt.plot(powell_y, '-^', label='DFP')
pyplot.yscale('log')
plt.grid(True)
plt.title('Objective')
plt.legend(handles=[b, l, p])
plt.xlabel('Number of Iterations')
plt.ylabel('Objective')
