import pennylane as qml
from pennylane import numpy as np

def Random_QLSP_v7(n, kappa, algorithm, J, k_loc):
    qubit_wires = range(n)
    dim = 2**n

    paulis = ['X', 'Y', 'Z']

    gate_set = []
    coefficient_set = []

    for j in range(int(J)):
        cond = False
        while cond == False:
            sigmas = np.random.choice(paulis, k_loc)
            pos = np.sort(np.random.choice(qubit_wires,  k_loc, replace=False))


            int_string = ''
            counter = 0
            for i in qubit_wires:
                if i in pos:
                    int_string += str(sigmas[counter])
                    counter += 1
                else:
                    int_string += 'I'
            
            if int_string in gate_set:   
                pass
            else:
                final_string = int_string             
                gate_set.append(final_string)            
                a = 2*(np.random.rand()-0.5)
                coefficient_set.append(a)
                cond = True

    A = 0

    if algorithm=='shadows':
        return gate_set, coefficient_set, A