import pennylane as qml
from pennylane import numpy as np
import random
import math
from Pauli_algebra_v3 import * 
from Ising_Inspired_QLSP import *
from scipy.optimize import minimize

task_number = 1

seed = 1
np.random.seed(seed)

n = 4
shadow_error = 0.01
qubit_wires = range(n)
dim = 2**n
layers = 4
kappa = 60

error = 0.01

cost_function_values = []
fidelity_values = []

params = 0.01*np.random.randn(n+2*(n-1)*layers) if n%2==0 else 0.01*np.random.randn(n+n*2*layers)

gate_set, coefficient_set, A = Ising_Inspired_QLSP(n, kappa=kappa, algorithm='shadows')
b = [1/np.sqrt(dim) for i in range(dim)]

list_of_string_observables, local_calc, norm_calc = create_operator_list(A_strings=gate_set, A_coeffs=coefficient_set)
print('Problem created')
#single layer
def layer(qubit_wires, parameters):
    n = len(qubit_wires)
    if n%2==0:
        qml.broadcast(qml.CNOT,wires=qubit_wires, pattern='double')
        qml.broadcast(qml.RY, wires=qubit_wires, parameters=parameters[0:n], pattern='single')
        qml.broadcast(qml.CNOT,wires=qubit_wires[1:], pattern='double')
        qml.broadcast(qml.RY, wires=qubit_wires[1:-1], parameters=parameters[n:2*(n-1)], pattern='single')
    else:
        qml.broadcast(qml.CNOT,wires=qubit_wires, pattern='double')
        qml.CNOT(wires=[qubit_wires[0], qubit_wires[-1]])
        qml.broadcast(qml.RY, wires=qubit_wires, parameters=parameters[0:n], pattern='single')
        qml.broadcast(qml.CNOT,wires=qubit_wires[1:], pattern='double')
        qml.CNOT(wires=[qubit_wires[0], qubit_wires[-1]])
        qml.broadcast(qml.RY, wires=qubit_wires, parameters=parameters[n:2*n], pattern='single')

#Fixed anstaz
def apply_fixed_ansatz(qubit_wires, parameters, layers):  
    n = len(qubit_wires)  
    qml.broadcast(qml.RY, wires=qubit_wires, parameters=parameters[0:n], pattern='single')
    
    if n%2==0:
        for l in range(layers):
            layer(qubit_wires, parameters[n+2*(n-1)*l:n+2*(n-1)*l+2*(n-1)])
    else:
        for l in range(layers):
            layer(qubit_wires, parameters[n+2*n*l:n+2*n*l+2*n])

list_of_observables_local = []
list_of_observables_norm = []

list_of_observale_coeffs_local = []
list_of_observale_coeffs_norm = []

for i in enumerate(list_of_string_observables):
    if i[1] in list(local_calc.keys()):
        list_of_observale_coeffs_local.append(local_calc[i[1]])
        list_of_observables_local.append(qml.pauli.string_to_pauli_word(i[1]))
    if i[1] in list(norm_calc.keys()):
        list_of_observale_coeffs_norm.append(norm_calc[i[1]])
        list_of_observables_norm.append(qml.pauli.string_to_pauli_word(i[1]))

H_local = qml.Hamiltonian(list_of_observale_coeffs_local, list_of_observables_local)
H_norm = qml.Hamiltonian(list_of_observale_coeffs_norm, list_of_observables_norm)

shadow_size_bound, k = shadow_bound(error=shadow_error, observables= list_of_string_observables, type='Pauli')
print('Shadow Size defined')

dev = qml.device("default.qubit", wires=n, shots=shadow_size_bound)

@qml.qnode(dev)
def circuit_norm(parameters):
    apply_fixed_ansatz(qubit_wires, parameters, layers)
    return qml.shadow_expval(H_norm)

@qml.qnode(dev)
def circuit_local(parameters):
    apply_fixed_ansatz(qubit_wires, parameters, layers)
    return qml.shadow_expval(H_local)

dev2 = qml.device('lightning.qubit', wires=qubit_wires)

@qml.qnode(dev2)
def circuit_fielity(parameters):
    apply_fixed_ansatz(qubit_wires, parameters, layers)
    return qml.state()

def calc_fidelity(state_vec):
    Aprodx = np.matmul(A, state_vec)
    Aprodx_norm = Aprodx/np.linalg.norm(Aprodx)  
    fidelity = np.linalg.norm(np.dot(np.array(b), Aprodx_norm))
    return fidelity

def calculate_cost_function(parameters):
    overall_sum_1 = sum([coeff**2 for coeff in coefficient_set])
    overall_sum_1 += circuit_norm(parameters)
    overall_sum_2 = circuit_local(parameters) 
    if 'I'*n in local_calc.keys():
        overall_sum_2+= local_calc['I'*n]
    cost = 0.5-(1/(2*n))*(np.linalg.norm(overall_sum_2)/np.linalg.norm(overall_sum_1))
    res = circuit_fielity(parameters)
    fidelity = calc_fidelity(res)
    fidelity_values.append(fidelity._value.item())
    cost_function_values.append(cost._value.item())
    print(cost._value.item())
    return cost

condition = False
opt = qml.AdamOptimizer(stepsize=0.1)
params_f = np.copy(params)

while condition==False:
    params_f, cost = opt.step_and_cost(calculate_cost_function, params_f)

    if any(val > 1-error for val in fidelity_values)==True:
        condition = True
    
    if cost < 0.1:
        opt = qml.AdamOptimizer(stepsize=0.01)
    elif cost < 0.01:
        opt = qml.AdamOptimizer(stepsize=0.001)
    elif cost < 0.001:
        opt = qml.AdamOptimizer(stepsize=0.0001)
    elif cost < 0.0001:
        opt = qml.AdamOptimizer(stepsize=0.00001)

    evals = int(len(cost_function_values))
    if evals%10 == 0:
        fname = f'Random_{n}qubits_{shadow_error}serr_costs_{seed}seed_Powell'
        np.save(fname, np.array(cost_function_values))

        fname = f'Random_{n}qubits_{shadow_error}serr_fidelity_{seed}seed_Powell'
        np.save(fname, np.array(fidelity_values))

        fname = f'Random_{n}qubits_{shadow_error}serr_params_{seed}seed_Powell'
        np.save(fname, np.array(params_f))

                
fname = f'Ising_{n}qubits_{shadow_error}serr_costs_{seed}seed_Powell'
np.save(fname, np.array(cost_function_values))

fname = f'Ising_{n}qubits_{shadow_error}serr_fidelity_{seed}seed_Powell'
np.save(fname, np.array(fidelity_values))

fname = f'Ising_{n}qubits_{shadow_error}serr_params_{seed}seed_Powell'
np.save(fname, np.array(params_f))
