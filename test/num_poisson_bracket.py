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
f = 'x6'
g = 'x5'

num_poisson_bracket_res = dict()
j = 2
for mesh_path in ['6Qmesh_10_2.npy', '6Qmesh_10_3.npy' , '6Qmesh_10_4.npy', '6Qmesh_10_5.npy', '6Qmesh_10_6.npy', '6Qmesh_10_7.npy']:
    print(f'step {j}')
    tiempos = dict()
    with open(mesh_path, 'rb') as file:
        mesh = np.load(file)
    for k in range(25):
        A = datetime.datetime.now()
        npg.num_poisson_bracket(P, f, g, mesh, pt_output=True)
        B = datetime.datetime.now()
        tiempos[k] = (B - A).total_seconds()
    promedio = stat.mean(tiempos.values())
    desviacion = stat.pstdev(tiempos.values())
    tiempos['promedios'] = promedio
    tiempos['desviacion'] = desviacion
    num_poisson_bracket_res[f'10**{j}'] = tiempos
    j = j + 1

print(num_poisson_bracket_res)
print('Finish')
