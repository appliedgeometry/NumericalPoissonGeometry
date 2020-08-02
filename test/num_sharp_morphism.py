import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

print('Start')
npg = NumPoissonGeometry(3, 'x')
P_so3 = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
dK = {(1,): 'x1', (2,): 'x2', (3,): 'x3'}

num_sharp_morphism_res = dict()
for i in [2, 3, 4, 5, 6]:
    print(f'step {i}')
    Qmesh_10_3 = np.random.rand(10**i, 3)
    tiempos = dict()
    for k in range(25):
        A = datetime.datetime.now()
        npg.num_sharp_morphism(P_so3, dK, Qmesh_10_3, pt_output=True)
        B = datetime.datetime.now()
        tiempos[k] = (B - A).total_seconds()
    num_sharp_morphism_res[f'10**{i}'] = tiempos

print(num_sharp_morphism_res)
print('Finish')

# print(f'tiempos: {tiempos}')
# print(f'Promedio = {stat.mean(tiempos.values())}\nDS = {stat.pstdev(tiempos.values())}\n')
