import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

print('Start')
npg = NumPoissonGeometry(3, 'x')
P_so3 = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
Lambda = {(1, 2): 'x1 - x2', (1, 3): 'x1 - x3', (2, 3): 'x2 - x3'}

num_gauge_transformation_res = dict()
j = 2
for mesh_path in ['3Qmesh_10_2.npy', '3Qmesh_10_3.npy', '3Qmesh_10_4.npy', '3Qmesh_10_5.npy', '3Qmesh_10_6.npy', '3Qmesh_10_7.npy']:
    print(f'step {j}')
    tiempos = dict()
    with open(mesh_path, 'rb') as f:
        mesh = np.load(f)
    for k in range(25):
        A = datetime.datetime.now()
        npg.num_gauge_transformation(P_so3, Lambda, mesh, pt_output=True)
        B = datetime.datetime.now()
        tiempos[k] = (B - A).total_seconds()
    promedio = stat.mean(tiempos.values())
    desviacion = stat.pstdev(tiempos.values())
    tiempos['promedios'] = promedio
    tiempos['desviacion'] = desviacion
    num_gauge_transformation_res[f'10**{j}'] = tiempos
    j = j + 1

print(num_gauge_transformation_res)
print('Finish')
