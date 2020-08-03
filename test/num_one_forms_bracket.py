import datetime
import time
import numpy as np
import statistics as stat
from numpoisson.numpoisson import NumPoissonGeometry

npg = NumPoissonGeometry(6, 'x')
P = {(1, 4): 1, (2, 5): 1, (3, 6): 1, (5, 6): 'x2**2'}
alpha = {(5,): 1}
beta = {(6,): 1}

num_one_forms_bracket_res = dict()
for i in [2, 3, 4, 5, 6]:
    print(f'step {i}')
    Qmesh_10_3 = np.random.rand(10**i, 6)
    tiempos = dict()
    for k in range(25):
        A = datetime.datetime.now()
        npg.num_one_form_bracket(P, alpha, beta, Qmesh_10_3 , pt_output=True)
        B = datetime.datetime.now()
        tiempos[k] = (B - A).total_seconds()
    num_one_forms_bracket_res[f'10**{i}'] = tiempos

print(num_one_forms_bracket_res)
print('Finish')

# print(f'tiempos: {tiempos}')
# print(f'Promedio = {stat.mean(tiempos.values())}\nDS = {stat.pstdev(tiempos.values())}\n')
