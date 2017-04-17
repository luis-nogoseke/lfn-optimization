from scipy.optimize import minimize, rosen, rosen_der
from numpy.random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import pyplot
import numpy as np




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
     xs.append(xk)
     ys.append(rosen(xk))
     global it
     it = it + 1


iters = []
feval = []
sol = []
objective = []
#x0s = []
for i in range(0, 30):
#    x0 = (random(2)-1)*20
#    x0s.append(x0)
    global it
    it = 0
    output = minimize(rosen, x0s[i], method=_minimize_dfp, callback=bstop, options= {'disp': True})
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

xs = [np.array([-1, -0.5])]
ys = [rosen(xs[0])]
minimize(rosen, [-1, -0.5], method=_minimize_dfp, callback=bstop, options= {'disp': True})
linex = [-1]
liney = [-0.5]
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
