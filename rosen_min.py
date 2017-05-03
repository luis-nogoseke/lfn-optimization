from scipy.optimize import minimize, rosen, rosen_der
from numpy.random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import pyplot
import numpy as np
from  numpy import array  as array
import timeit



# Plot the function
fig = plt.figure()
ax = Axes3D(fig, azim = -128, elev = 43)
s = .05
X = np.arange(-2, 2.+s, s)
Y = np.arange(-1, 3.+s, s)
X, Y = np.meshgrid(X, Y)
Z = rosen([X, Y])
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = cm.jet, linewidth=0, edgecolor='none')
ax.set_xlim([-2,2.0])                                                       
ax.set_ylim([-1,3.0])                                                       
ax.set_zlim([0, 2500]) 
plt.xlabel("x")
plt.ylabel("y")
plt.title('Rosenbrock 2D')
plt.figure()
# plt.show()

##################################################################

##################################################################
# Get 30 solutions

xs =  []
ys = []
it = 0

def bstop(xk):
     xs.append(np.copy(xk))
     ys.append(rosen(xk))
     global it
     it = it + 1


iters = []
feval = []
sol = []
objective = []
times= []
#x0s = []
for i in range(0, 30):
#    x0 = (random(2)-1)*20
#    x0s.append(x0)
    global it
    it = 0
    start_time = timeit.default_timer()
    output = minimize(rosen, x0s[i], method='L-BFGS-B', callback=bstop, options= {'disp': True})
    times.append(timeit.default_timer() - start_time)
    iters.append(it)
    feval.append(output.nfev)
    sol.append(output.x)
    objective.append(output.fun) 

##################################################################
# Plot solution on isolines

# Isolines
delta = 0.05
s = 0.05
X = np.arange(-1.5, 1.5, delta)
Y = np.arange(-1, 3, delta)
X, Y = np.meshgrid(X, Y)
Z = rosen([X, Y])
levels = np.arange(10, 300, 10)
plt.contour(X, Y, Z, levels=levels, norm=LogNorm())

xs = [np.array([-1, -0.5])]
ys = [rosen(xs[0])]
minimize(rosen, [-1, -0.5], method='BFGS', callback=bstop, options= {'disp': True})

linex = [-1]
liney = [-0.5]
for i in xs:
    linex.append(i[0])
    liney.append(i[1])

bfgs_y = list(ys)
bfgs, = plt.plot(linex, liney, '-o', label='BFGS')

xs = [np.array([-1, -0.5])]
ys = [rosen(xs[0])]
minimize(rosen, [-1, -0.5], method='L-BFGS-B', callback=bstop, options= {'disp': True})
linex = [-1]
liney = [-0.5]
for i in xs:
    linex.append(i[0])
    liney.append(i[1])

lbfgsb_y = list(ys)
lbfgsb, = plt.plot(linex, liney, '-s', label='L-BFGS-B')

#xs = [np.array([-1, -0.5])]

xs = [array([-1, -5.000000e-01]), 
array([0, -3.311268e-03]), 
array([1.355493e-03, -5.959506e-03]), 
array([3.809721e-02, 3.027428e-03]), 
array([2.079182e-01, 4.341413e-02]), 
array([2.970837e-01, 6.464193e-02]), 
array([2.855097e-01, 6.251982e-02]), 
array([2.888685e-01, 7.041021e-02]), 
array([3.002800e-01, 9.371150e-02]), 
array([3.301636e-01, 1.239246e-01]), 
array([3.431667e-01, 1.362663e-01]), 
array([3.790113e-01, 1.696302e-01]), 
array([6.666666e-01, 4.364363e-01]), 
array([6.673740e-01, 4.370968e-01]), 
array([6.827426e-01, 4.586153e-01]), 
array([8.033587e-01, 6.278525e-01]), 
array([7.713845e-01, 5.861505e-01]), 
array([7.816157e-01, 6.030214e-01]), 
array([8.603122e-01, 7.330873e-01]), 
array([8.943182e-01, 7.953762e-01]), 
array([9.339127e-01, 8.700019e-01]), 
array([9.673623e-01, 9.336677e-01]), 
array([9.848576e-01, 9.714503e-01]), 
array([1.000155e+00, 9.998819e-01]), 
array([9.986836e-01, 9.973498e-01]), 
array([9.995547e-01, 9.990956e-01])]


#ys = [rosen(xs[0])]
ys = [229,
1.001096e+00,
1.000845e+00,
9.255054e-01,
6.273970e-01,
5.498666e-01,
5.465811e-01,
5.226986e-01,
4.908637e-01,
4.709314e-01,
4.656657e-01,
4.531264e-01,
1.175240e-01,
1.175146e-01,
1.063106e-01,
6.940715e-02,
6.015688e-02,
5.393538e-02,
2.448267e-02,
1.313008e-02,
4.847591e-03,
1.515564e-03,
4.560516e-04,
1.832000e-05,
1.769561e-06,
2.180442e-07,
1.568248e-10]


#minimize(rosen, [-1, -0.5], method=_minimize_dfp, callback=bstop, options= {'disp': True})
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
plt.show()

##################################################################


iters = []
feval = []
sol = []
objective = []
x0s = []
it = 0
for i in range(0, 30):
    x0 = (random(30)-1)*10
    x0s.append(x0)
    global it
    it = 0
    output = minimize(rosen, x0, method='BFGS', callback=bstop, options= {'disp': True})
    iters.append(it)
    feval.append(output.nfev)
    sol.append(output.x)
    objective.append(output.fun) 
