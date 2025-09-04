import pennylane as qml
from pennylane import numpy as np
import random
from scipy.optimize import minimize
import math
from Potential_problem import *
from Pauli_algebra_v4 import *

n = 11
dim = 2**n

gate_set, coefficient_set, A_norm, b_norm = Potential_problem(n)

gate_set
fname = f'{n}x{n}sites_potential.txt'

with open(fname, 'w') as f:
    for idx, gate in enumerate(gate_set):
        if coefficient_set[idx].real != 0.0 or coefficient_set[idx].imag != 0.0:
            f.write(gate + ' ' + str(coefficient_set[idx].real) + ' ' + str(coefficient_set[idx].imag)) 
            f.write('\n')
    
np.save(f'{n}x{n}_bnorm', np.array(b_norm))