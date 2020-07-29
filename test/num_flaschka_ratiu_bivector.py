import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

npg = NumPoissonGeometry(4, 'x')
functions = ['1/2*x4', '-x1**2 + x2**2 + x3**2']
Qmesh_10_4 = np.random.rand(10**4, 4)

tiempos = dict()
for k in range(25):
    A = datetime.datetime.now()
    npg.num_flaschka_ratiu_bivector(functions, Qmesh_10_4, pt_output=True)
    B = datetime.datetime.now()
    tiempos[k] = (B - A).total_seconds()

print(f'tiempos: {tiempos}')
print(f'Promedio = {stat.mean(tiempos.values())}\nDS = {stat.pstdev(tiempos.values())}\n')
