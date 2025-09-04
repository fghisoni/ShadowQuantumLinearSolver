import pennylane as qml
from pennylane import numpy as np

def Ising_Inspired_QLSP(n, kappa, algorithm, J=0.1):
    qubit_wires = range(n)
    dim = 2**n
    no_gates = 2*n

    gate_set = []
    coefficient_set = []
    for i in qubit_wires:
        X_gate = 'I'*i + 'X' + (n-i-1)*'I'
        gate_set.append(X_gate)
        coefficient_set.append(1)
        if i<n-1:
            Z_gate = 'I'*i + 'ZZ' + (n-i-2)*'I'
            gate_set.append(Z_gate)
            coefficient_set.append(J)
    
    gate_set_perceptron = []
    A = np.zeros((dim,dim))

    for i in range(no_gates - 1):
        An = np.zeros((2,2))
        for j in range(n):
            if j == 0:
                if gate_set[i][j] == 'I': 
                    An = np.eye(2)
                elif gate_set[i][j] == 'X':
                    An = qml.PauliX.compute_matrix()
                elif gate_set[i][j] == 'Z':
                    An = qml.PauliZ.compute_matrix()
            else:
                if gate_set[i][j] == 'I': 
                    An = np.kron(An, np.eye(2))
                elif gate_set[i][j] == 'X':
                    An = np.kron(An, qml.PauliX.compute_matrix())
                elif gate_set[i][j] == 'Z':
                    An = np.kron(An, qml.PauliZ.compute_matrix())
        
        gate_set_perceptron.append(An)
        A += coefficient_set[i]*An
    
    def rescale_matrix_eigenvalues(matrix, x, y=1):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        min_eigenvalue = min(eigenvalues)
        max_eigenvalue = max(eigenvalues)

        eta = (max_eigenvalue*x - min_eigenvalue*y)/(1-x)
        zeta = (max_eigenvalue - min_eigenvalue)/(1-x)

        rescaled_matrix = (matrix + eta*np.eye(dim))/zeta
        return rescaled_matrix, eta, zeta

    rescaled_matrix, eta, zeta = rescale_matrix_eigenvalues(A, 1/kappa)

    gate_set.append('I'*n)
    gate_set_perceptron.append(np.eye(dim))

    coefficient_set.append(eta)
    coefficient_set = coefficient_set/zeta

    if algorithm=='shadows':
        return gate_set, coefficient_set, rescaled_matrix
    elif algorithm=='perceptron':
        return gate_set_perceptron, coefficient_set, rescaled_matrix
    
def matrix_condition_number(matrix):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    condition_number = max(singular_values) / min(singular_values)
    return condition_number

def calc_klocality(pstr):
    return pstr.count('X') + pstr.count('Y') + pstr.count('Z')

def shadow_bound(error, observables, type='1', failure_rate=0.01):
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)

    k_loc = 0
    for ob in observables:
        k_int = calc_klocality(ob)
        if k_int > k_loc:
            k_loc = k_int
    return int(np.ceil(np.log(M)*(3**k_loc)/error**2)), int(K)
