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
x1 = [19.16244142,  -8.84373827,  -9.61465608, -14.16049707, -10.31816955,  -2.06711925,  -9.76682906, -10.2196214 ,-19.47103279, -15.60733267, -14.74033397, -12.19729402,-13.25763513, -10.6918214 ,  -4.65098476, -10.44269597,-16.24941483,  -1.93662675, -19.5968836 , -16.4426469 ,-15.96335692,  -5.91464348, -15.44063562, -13.92548209, -5.29251259, -11.0406572 , -14.06732047,  -9.8154472 , -0.81449062, -16.7515287]

ys = [rosen(x1)]
minimize(rosen, x1, method='BFGS', callback=bstop, options= {'disp': True})

bfgs_y = list(ys)

ys = [rosen(x1)]
minimize(rosen, x1, method='L-BFGS-B', callback=bstop, options= {'disp': True})

lbfgsb_y = list(ys)


ys = [1.171021e+08,
1.011677e+08,
9.507324e+07,
9.345037e+07,
9.375648e+07,
9.329303e+07,
9.325070e+07,
9.322355e+07,
9.320441e+07,
9.317711e+07,
9.315965e+07,
9.314863e+07,
9.314421e+07,
9.313755e+07,
9.312846e+07,
9.311031e+07,
9.308026e+07,
9.307318e+07,
9.305779e+07,
9.300462e+07,
9.247609e+07,
9.201532e+07,
9.103527e+07,
8.865215e+07,
7.894637e+07,
7.736897e+07,
7.719423e+07,
7.714015e+07,
7.681186e+07,
7.414562e+07,
6.164531e+07,
2.207991e+07,
1.067651e+07,
7.346318e+06,
2.298724e+06,
1.592257e+06,
1.034926e+06,
3.828778e+05,
1.266231e+05,
4.915495e+04,
2.931377e+04,
1.883113e+04,
1.250849e+04,
9.721541e+03,
8.546081e+03,
7.711669e+03,
6.995326e+03,
6.379238e+03,
5.821292e+03,
5.304984e+03,
4.803033e+03,
4.305541e+03,
3.845520e+03,
3.606976e+03,
3.481324e+03,
3.124648e+03,
3.010547e+03,
2.790071e+03,
2.708211e+03,
2.584882e+03,
2.512085e+03,
2.413673e+03,
2.344604e+03,
2.271408e+03,
2.219367e+03,
2.163770e+03,
2.120493e+03,
2.067341e+03,
2.030008e+03,
1.981795e+03,
1.950461e+03,
1.910126e+03,
1.882765e+03,
1.848159e+03,
1.822273e+03,
1.791659e+03,
1.766163e+03,
1.738860e+03,
1.714898e+03,
1.692443e+03,
1.673860e+03,
1.657606e+03,
1.642809e+03,
1.628442e+03,
1.614727e+03,
1.601363e+03,
1.588920e+03,
1.577252e+03,
1.566761e+03,
1.557080e+03,
1.548269e+03,
1.539881e+03,
1.531983e+03,
1.524239e+03,
1.516806e+03,
1.509489e+03,
1.502546e+03,
1.495978e+03,
1.490058e+03,
1.484656e+03,
1.479687e+03,
1.474906e+03,
1.470334e+03,
1.465814e+03]

powell_y = list(ys)

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


