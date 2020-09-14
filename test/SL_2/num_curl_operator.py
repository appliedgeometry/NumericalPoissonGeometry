import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

print('Start')
npg = NumPoissonGeometry(3, 'x')
P = {(1, 2): '-x3', (1, 3): '-x2', (2, 3): 'x1'}

num_curl_operator_res = dict()
j = 2
for mesh_path in ['3Qmesh_10_2.npy', '3Qmesh_10_3.npy', '3Qmesh_10_4.npy', '3Qmesh_10_5.npy', '3Qmesh_10_6.npy', '3Qmesh_10_7.npy', '3Qmesh_10_8.npy']:
    print(f'step {j}')
    tiempos = dict()
    with open(mesh_path, 'rb') as f:
        mesh = np.load(f)
    for k in range(25):
        A = datetime.datetime.now()
        npg.num_curl_operator(P, 1, mesh, pt_output=True)
        B = datetime.datetime.now()
        tiempos[k] = (B - A).total_seconds()
    promedio = stat.mean(tiempos.values())
    desviacion = stat.pstdev(tiempos.values())
    tiempos['promedios'] = promedio
    tiempos['desviacion'] = desviacion
    num_curl_operator_res[f'10**{j}'] = tiempos
    print(tiempos)
    j = j + 1

print(num_curl_operator_res)
print('Finish')
