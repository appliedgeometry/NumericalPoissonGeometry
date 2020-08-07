import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

print('Start')
npg = NumPoissonGeometry(4, 'x')
functions = ['1/2*x4', '-x1**2 + x2**2 + x3**2']

num_flaschka_ratiu_bivector_res = dict()
j = 2
for mesh_path in ['4Qmesh_10_2.npy', '4Qmesh_10_3.npy', '4Qmesh_10_4.npy', '4Qmesh_10_5.npy', '4Qmesh_10_6.npy', '4Qmesh_10_7.npy']:
    print(f'step {j}')
    tiempos = dict()
    with open(mesh_path, 'rb') as f:
        mesh = np.load(f)
    for k in range(25):
        print(f'Step: {k}')
        A = datetime.datetime.now()
        npg.num_flaschka_ratiu_bivector(functions, mesh, pt_output=True)
        B = datetime.datetime.now()
        tiempos[k] = (B - A).total_seconds()
    promedio = stat.mean(tiempos.values())
    desviacion = stat.pstdev(tiempos.values())
    tiempos['promedios'] = promedio
    tiempos['desviacion'] = desviacion
    num_flaschka_ratiu_bivector_res[f'10**{j}'] = tiempos
    j = j + 1

print(num_flaschka_ratiu_bivector_res)
print('Finish')

# print(f'tiempos: {tiempos}')
# print(f'Promedio = {stat.mean(tiempos.values())}\nDS = {stat.pstdev(tiempos.values())}\n')
