import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

print('Start')
npg = NumPoissonGeometry(4, 'x')
functions = ['1/2*x4', '-x1**2 + x2**2 + x3**2']

num_flaschka_ratiu_bivector_res = dict()
for i in [2, 3, 4, 5, 6]:
    print(f'step {i}')
    Qmesh_10_4 = np.random.rand(10**i, 4)
    tiempos = dict()
    for k in range(25):
        A = datetime.datetime.now()
        npg.num_flaschka_ratiu_bivector(functions, Qmesh_10_4, pt_output=True)
        B = datetime.datetime.now()
        tiempos[k] = (B - A).total_seconds()
    num_flaschka_ratiu_bivector_res[f'10**{i}'] = tiempos

print(num_flaschka_ratiu_bivector_res)
print('Finish')

# print(f'tiempos: {tiempos}')
# print(f'Promedio = {stat.mean(tiempos.values())}\nDS = {stat.pstdev(tiempos.values())}\n')
