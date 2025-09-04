import pennylane as qml
from pennylane import numpy as np
import random
import math
from Pauli_algebra_v4 import * 

task_number = 1

seed = 1
np.random.seed(seed)

n = 4
shadow_error = 0.01
qubit_wires = range(n)
dim = 2**n
layers = 4
kappa = 10

error = 0.01

cost_function_values = []
fidelity_values = []

params = 0.01*np.random.randn(n+2*(n-1)*layers) if n%2==0 else 0.01*np.random.randn(n+n*2*layers)

def create_A(gate_set, coefficient_set):
    A = np.zeros((dim,dim), dtype = complex)
    for i in range(len(gate_set)):
        An = np.zeros((2,2))
        for j in range(n):
            if j == 0:
                if gate_set[i][j] == 'I': 
                    An = np.eye(2)
                elif gate_set[i][j] == 'X':
                    An = qml.PauliX.compute_matrix()
                elif gate_set[i][j] == 'Y':
                    An = qml.PauliY.compute_matrix()
                elif gate_set[i][j] == 'Z':
                    An = qml.PauliZ.compute_matrix()
            else:
                if gate_set[i][j] == 'I': 
                    An = np.kron(An, np.eye(2))
                elif gate_set[i][j] == 'X':
                    An = np.kron(An, qml.PauliX.compute_matrix())
                elif gate_set[i][j] == 'Y':
                    An = np.kron(An, qml.PauliY.compute_matrix())
                elif gate_set[i][j] == 'Z':
                    An = np.kron(An, qml.PauliZ.compute_matrix())
        
        A = np.add(A, coefficient_set[i]*An)
    return A

gate_set, coefficient_set =  [], []

fname = f'{n}x{n}sites_potential.txt'
with open(fname, 'r') as f:
    for line in f.readlines():
        gate, real, imag = line.split()
        gate_set.append(gate)
        coefficient_set.append(complex(float(real), float(imag)))
        
A = create_A(gate_set, coefficient_set)
b = np.load(f'{n}x{n}_bnorm.npy')
list_of_string_observables, local_calc, norm_calc = create_operator_list(A_strings=gate_set, A_coeffs=coefficient_set)

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
    return cost

condition = False
opt = qml.AdamOptimizer(stepsize=0.1)
params_f = np.copy(params)

term_cond = (1/n) * (error/kappa)**2
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

    if int(len(cost_function_values))%10==0:
        fname = f'Potential_{n}qubits_{shadow_error}serr_costs_{task_number}_Adam'
        np.save(fname, np.array(cost_function_values))
        with open(fname+'.txt', 'w') as f:
            for i in range(len(cost_function_values)):
                f.write(str(cost_function_values[i]))
                f.write('\n')

        fname = f'Potential_{n}qubits_{shadow_error}serr_fidelity_{task_number}_Adam'
        np.save(fname, np.array(fidelity_values))
        with open(fname+'.txt', 'w') as f:
            for i in range(len(fidelity_values)):
                f.write(str(fidelity_values[i]))
                f.write('\n')

        fname = f'Potential_{n}qubits_{shadow_error}serr_params_{task_number}_Adam'
        np.save(fname, np.array(params_f))
        with open(fname+'.txt', 'w') as f:
            for i in range(len(params_f)):
                f.write(str(params_f[i]))
                f.write('\n')
                
fname = f'Potential_{n}qubits_{shadow_error}serr_costs_{task_number}_Adam'
np.save(fname, np.array(cost_function_values))
with open(fname+'.txt', 'w') as f:
    for i in range(len(cost_function_values)):
        f.write(str(cost_function_values[i]))
        f.write('\n')

fname = f'Potential_{n}qubits_{shadow_error}serr_fidelity_{task_number}_Adam'
np.save(fname, np.array(fidelity_values))
with open(fname+'.txt', 'w') as f:
    for i in range(len(fidelity_values)):
        f.write(str(fidelity_values[i]))
        f.write('\n')

fname = f'Potential_{n}qubits_{shadow_error}serr_params_{task_number}_Adam'
np.save(fname, np.array(params_f))
with open(fname+'.txt', 'w') as f:
    for i in range(len(params_f)):
        f.write(str(params_f[i]))
        f.write('\n')