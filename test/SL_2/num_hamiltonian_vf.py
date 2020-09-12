import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

npg = NumPoissonGeometry(3, 'x')
P_sl2 = {(1, 2): '-x3', (1, 3): '-x2', (2, 3): 'x1'}
h = '(x1**2)/2 + (x2**2)/2 + (x3**2)/2'

num_hamiltonian_vf_res = dict()
j = 2
for mesh_path in ['3Qmesh_10_2.npy', '3Qmesh_10_3.npy', '3Qmesh_10_4.npy', '3Qmesh_10_5.npy', '3Qmesh_10_6.npy', '3Qmesh_10_7.npy', '3Qmesh_10_8.npy']:
    print(f'step {j}')
    tiempos = dict()
    with open(mesh_path, 'rb') as f:
        mesh = np.load(f)
    for k in range(25):
        A = datetime.datetime.now()
        npg.num_hamiltonian_vf(P_sl2, h, mesh, pt_output=True)
        B = datetime.datetime.now()
        tiempos[k] = (B - A).total_seconds()
    promedio = stat.mean(tiempos.values())
    desviacion = stat.pstdev(tiempos.values())
    tiempos['promedios'] = promedio
    tiempos['desviacion'] = desviacion
    num_hamiltonian_vf_res[f'10**{j}'] = tiempos
    j = j + 1

print(num_hamiltonian_vf_res)
print('Finish')
