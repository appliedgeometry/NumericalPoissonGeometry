import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

print('Start')
npg = NumPoissonGeometry(3, 'x')
P = {(1, 2): '-x3', (1, 3): '-x2', (2,3): 'x1'}
W = {(1,): 'x1 * x3 * exp(-1/(x1**2 + x2**2 - x3**2)**2) / (x1**2 + x2**2)',
     (2,): 'x2 * x3 * exp(-1/(x1**2 + x2**2 - x3**2)**2) / (x1**2 + x2**2)',
     (3,): 'exp(-1 / (x1**2 + x2**2 - x3**2)**2)'}

num_coboundary_operator_res = dict()
for i in [2, 3, 4, 5, 6]:
    Qmesh_10_3 = np.random.rand(10**i, 3)
    tiempos = dict()
    for k in range(25):
        A = datetime.datetime.now()
        npg.num_coboundary_operator(P, W, Qmesh_10_3, pt_output=True)
        B = datetime.datetime.now()
        tiempos[k] = (B - A).total_seconds()
    num_coboundary_operator_res[f'10**{i}'] = tiempos

print(num_coboundary_operator_res)
print('Finish')
#print(f'tiempos: {tiempos}')
#print(f'Promedio = {stat.mean(tiempos.values())}\nDS = {stat.pstdev(tiempos.values())}\n')
