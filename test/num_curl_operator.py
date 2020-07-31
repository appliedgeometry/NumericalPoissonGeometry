import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

print('Start')
npg = NumPoissonGeometry(4, 'x')
P ={(1, 3): '2*x4', (1, 4): '2*x3', (2, 3): '-2*x4', (2, 4): '2*x3', (3, 4): 'x1 - x2'}

num_curl_operator_res = dict()
for i in [2, 3, 4, 5, 6]:
    Qmesh_10_3 = np.random.rand(10**i, 4)
    tiempos = dict()
    for k in range(25):
        A = datetime.datetime.now()
        npg.num_curl_operator(P, 1, Qmesh_10_3, pt_output=True)
        B = datetime.datetime.now()
        tiempos[k] = (B - A).total_seconds()
    num_curl_operator_res[f'10**{i}'] = tiempos

print(num_curl_operator_res)
print('Finish')
# print(f'tiempos: {tiempos}')
# print(f'Promedio = {stat.mean(tiempos.values())}\nDS = {stat.pstdev(tiempos.values())}\n')
