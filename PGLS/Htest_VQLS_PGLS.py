import pennylane as qml
from pennylane import numpy as np
import random
from scipy.optimize import minimize
import math
from Pauli_algebra_v4 import *

task_number = 1

seed = 11
np.random.seed(seed)

n = 4
n_shots = 900
qubit_wires = range(1,n+1)
dim = 2**n
layers = 4
kappa = 10

error = 0.01
print(f'task number: {task_number}')
print(f'number of shots: {n_shots}')

cost_function_values = []
fidelity_values = []

params = 0.01*np.random.randn(n+2*(n-1)*layers) if n%2==0 else 0.01*np.random.randn(n+n*2*layers)
print(params)

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
# local_calc, norm_calc

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


def Ui_phi(qubit_wires, parameters, layers, unitary_matrix):
    #fixed anstaz
    apply_fixed_ansatz(qubit_wires, parameters,layers)

    #An
    for i in range(len(unitary_matrix)):
        if unitary_matrix[i] == 'I':
            pass
        if unitary_matrix[i] == 'X':
            qml.CNOT([0,i+1])
        if unitary_matrix[i] == 'Y':
            qml.CY([0,i+1])
        if unitary_matrix[i] == 'Z':
            qml.CZ([0,i+1])

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

dev = qml.device('lightning.qubit', wires=qubit_wires)

@qml.qnode(dev)
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
    fidelity = np.dot(np.array(Ainvprodb_norm), state_vec)
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

term_cond = (1/n) * (error/kappa)**2
print(term_cond)
while condition==False:
    params_f, cost = opt.step_and_cost(calculate_cost_function, params_f)

    if any(val.real > 1-error for val in fidelity_values)==True:
        condition = True

    if int(len(cost_function_values))%10==0:
        fname = f'4qubit/Potential_{n}qubits_{n_shots}shots_costs_{seed}seed_Adam.txt'
        with open(fname, 'w') as f:
            for i in range(len(cost_function_values)):
                f.write(str(cost_function_values[i]))
                f.write('\n')

        fname = f'4qubit/Potential_{n}qubits_{n_shots}shots_fidelity_{seed}seed_Adam.txt'
        with open(fname, 'w') as f:
            for i in range(len(fidelity_values)):
                f.write(str(fidelity_values[i]))
                f.write('\n')

        fname = f'4qubit/Potential_{n}qubits_{n_shots}shots_params_{seed}seed_Adam.txt'
        with open(fname, 'w') as f:
            for i in range(len(params)):
                f.write(str(params[i]))
                f.write('\n')


fname = f'4qubit/Potential_{n}qubits_{n_shots}shots_costs_{seed}seed_Adam.txt'
with open(fname, 'w') as f:
    for i in range(len(cost_function_values)):
        f.write(str(cost_function_values[i]))
        f.write('\n')

fname = f'4qubit/Potential_{n}qubits_{n_shots}shots_fidelity_{seed}seed_Adam.txt'
with open(fname, 'w') as f:
    for i in range(len(fidelity_values)):
        f.write(str(fidelity_values[i]))
        f.write('\n')

fname = f'4qubit/Potential_{n}qubits_{n_shots}shots_params_{seed}seed_Adam.txt'
with open(fname, 'w') as f:
    for i in range(len(params)):
        f.write(str(params[i]))
        f.write('\n')
