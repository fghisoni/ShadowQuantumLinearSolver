import pennylane as qml
from pennylane import numpy as np
import random
from scipy.optimize import minimize
import math
from Random_QLSP_v8 import *
from Pauli_algebra_v3 import *

task = 1

seed = 10
np.random.seed(seed)

n = 4
n_shots = 900
qubit_wires = range(1,n+1)
dim = 2**n
layers = 10
kappa = 10

error = 0.1

cost_function_values = []
fidelity_values = []

params = 0.01*np.random.randn(3*n*layers)

gate_set, coefficient_set, A = Random_QLSP_v8(n, kappa=kappa, algorithm='shadows')
b = [1/np.sqrt(dim) for i in range(dim)]

list_of_string_observables, local_calc, norm_calc = create_operator_list(A_strings=gate_set, A_coeffs=coefficient_set)

#single layer
def layer(qubit_wires, parameters):
    n = len(qubit_wires)
    qml.broadcast(qml.RY, wires=qubit_wires, parameters=parameters[0:n], pattern='single')
    qml.broadcast(qml.RX, wires=qubit_wires, parameters=parameters[n:2*n], pattern='single')
    qml.broadcast(qml.RZ, wires=qubit_wires, parameters=parameters[2*n:3**n], pattern='single')
    qml.broadcast(qml.CNOT,wires=qubit_wires, pattern='ring')

#Fixed anstaz
def apply_fixed_ansatz(qubit_wires, parameters, layers):  
    n = len(qubit_wires)      
    for l in range(layers):
        layer(qubit_wires, parameters[3*n*l:3*n*(l+1)])

dev = qml.device('lightning.qubit', wires=range(n+1), shots = n_shots)

@qml.qnode(dev)
def circuit_phiphi(gate1, parameters, imag=False):
    qml.Hadamard(wires=0)
    
    if imag==True:
        qml.adjoint(qml.S(wires=0))

    apply_fixed_ansatz(qubit_wires, parameters, layers)
    for i in range(len(gate1)):
        if gate1[i] == 'I':
            pass
        if gate1[i] == 'X':
            qml.CNOT([0,i+1])
        if gate1[i] == 'Y':
            qml.CY([0,i+1])
        if gate1[i] == 'Z':
            qml.CZ([0,i+1])

    qml.Hadamard(wires=0)
    return qml.probs(wires=0)

dev2 = qml.device('lightning.qubit', wires=qubit_wires)

@qml.qnode(dev2)
def circuit_fielity(parameters):
    apply_fixed_ansatz(qubit_wires, parameters, layers)
    return qml.state()

def calc_fidelity_old(state_vec):
    Aprodx = np.matmul(A, state_vec)
    Aprodx_norm = Aprodx/np.linalg.norm(Aprodx)  
    fidelity = np.linalg.norm(np.dot(np.array(b), Aprodx_norm))
    return fidelity

def calc_fidelity(state_vec):
    Ainv = np.linalg.inv(A)
    Ainvprodb = np.matmul(Ainv,b)
    Ainvprodb_norm = Ainvprodb/np.linalg.norm(Ainvprodb)  
    fidelity = np.linalg.norm(np.dot(np.array(Ainvprodb_norm), state_vec))
    return fidelity

def calculate_cost_function(parameters):
    estimates = []
    
    for i in range(len(list_of_string_observables)):
        circ_real = circuit_phiphi(list_of_string_observables[i], parameters, imag=False)
        circ_imag = circuit_phiphi(list_of_string_observables[i], parameters, imag=True)

        real = circ_real[0] - circ_real[1]
        imag = circ_imag[0] - circ_imag[1]
        res = real + imag*1j
        estimates.append(res)

    overall_sum_1 = sum([coeff**2 for coeff in coefficient_set])
    
    for i in enumerate(list_of_string_observables):
        try:
            overall_sum_1 += norm_calc[i[1]]*estimates[i[0]]
        except:
            pass

    try:
        overall_sum_2 = local_calc['I'*n]
    except:
	    overall_sum_2 = 0

    for i in enumerate(list_of_string_observables):
        try:
            overall_sum_2+=local_calc[i[1]]*estimates[i[0]]
        except:
            pass

    cost = 0.5-(1/(2*n))*(np.linalg.norm(overall_sum_2)/np.linalg.norm(overall_sum_1))
    res = circuit_fielity(parameters)
    fidelity = calc_fidelity(res)
    fidelity_values.append(fidelity._value.item())
    cost_function_values.append(cost._value.item())
    return cost

condition = False
opt = qml.AdamOptimizer(stepsize=0.01)
params_f = np.copy(params)

while condition==False:
    params_f, cost = opt.step_and_cost(calculate_cost_function, params_f)

    if any(val > 1-error for val in fidelity_values)==True:
        condition = True
    

    evals = int(len(cost_function_values))
    if evals%1 == 0:

        fname = f'4qubit/Random_{n}qubits_{kappa}kappa_{n_shots}shots_{seed}seed_costs_Adam.txt'
        with open(fname, 'w') as f:
            for i in range(len(cost_function_values)):
                f.write(str(cost_function_values[i]))
                f.write('\n')

        fname = f'4qubit/Random_{n}qubits_{kappa}kappa_{n_shots}shots_{seed}seed_fidelity_Adam.txt'
        with open(fname, 'w') as f:
            for i in range(len(fidelity_values)):
                f.write(str(fidelity_values[i]))
                f.write('\n')

        fname = f'4qubit/Random_{n}qubits_{kappa}kappa_{n_shots}shots_{seed}seed_params_Adam.txt'
        with open(fname, 'w') as f:
            for i in range(len(params)):
                f.write(str(params[i]))
                f.write('\n')
    
    if evals > 20000:
        break

fname = f'4qubit/Random_{n}qubits_{kappa}kappa_{n_shots}shots_{seed}seed_costs_Adam.txt'

with open(fname, 'w') as f:
    for i in range(len(cost_function_values)):
        f.write(str(cost_function_values[i]))
        f.write('\n')

fname = f'4qubit/Random_{n}qubits_{kappa}kappa_{n_shots}shots_{seed}seed_fidelity_Adam.txt'

with open(fname, 'w') as f:
    for i in range(len(fidelity_values)):
        f.write(str(fidelity_values[i]))
        f.write('\n')

fname = f'4qubit/Random_{n}qubits_{kappa}kappa_{n_shots}shots_{seed}seed_params_Adam.txt'

with open(fname, 'w') as f:
    for i in range(len(params)):
        f.write(str(params[i]))
        f.write('\n')
