import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

npg = NumPoissonGeometry(3, 'x')
P = {(1, 2): '1/4*x3*(x1**4 + x2**4 + x3**4)', (1, 3): '-1/4*x2*(x1**4 + x2**4 + x3**4)', (2, 3): '1/4*x1*(x1**4 + x2**4 + x3**4)'}
Qmesh_10_3 = np.random.rand(10**3, 3)

tiempos = dict()
for k in range(25):
    A = datetime.datetime.now()
    npg.num_modular_vf(P, 1, Qmesh_10_3 , pt_output=True)
    B = datetime.datetime.now()
    tiempos[k] = (B - A).total_seconds()

print(f'tiempos: {tiempos}')
print(f'Promedio = {stat.mean(tiempos.values())}\nDS = {stat.pstdev(tiempos.values())}\n')
