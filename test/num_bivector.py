import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

npg = NumPoissonGeometry(3, 'x')
P_so3 = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
Qmesh_10_3 = np.random.rand(10**3, 3)

tiempos = dict()

for k in range(25):
    A = datetime.datetime.now()
    npg.num_bivector(P_so3, Qmesh_10_3, pt_output=True)
    B = datetime.datetime.now()
    tiempos[k] = (B - A).total_seconds()

print(f'tiempos: {tiempos}')
print(f'Promedio = {stat.mean(tiempos.values())}\nDS = {stat.pstdev(tiempos.values())}\n')
