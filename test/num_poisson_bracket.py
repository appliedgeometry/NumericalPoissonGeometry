import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

npg = NumPoissonGeometry(6, 'x')
P = {(1,2): 0, (1,3): 0, (1,4): 1, (1,5): 0, (1,6): 0,
     (2,3): 0, (2,4): 0, (2,5): 1, (2,6): 0,
     (3,4): 0, (3,5): 0, (3,6): 1,
     (4,5): 0, (4,6): 0,
     (5,6): 'x2**2'}

f, g = 'x6', 'x5'
# Change from Qmesh_10_7 to Qmesh_10_3
Qmesh_10_3 = np.random.rand(10**3, 6)

tiempos = dict()

for k in range(25):
    A = datetime.datetime.now()
    npg.num_poisson_bracket(P, f, g, Qmesh_10_3, pt_output=True)
    B = datetime.datetime.now()
    tiempos[k] = (B - A).total_seconds()

print(f'tiempos: {tiempos}')
print(f'Promedio = {stat.mean(tiempos.values())}\nDS = {stat.pstdev(tiempos.values())}\n')