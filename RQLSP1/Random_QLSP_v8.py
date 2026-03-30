import pennylane as qml
from pennylane import numpy as np

def Random_QLSP_v8(n, kappa, algorithm, J=0.1):
    qubit_wires = range(n)
    dim = 2**n

    seed = 3
    # seed = 13
    np.random.seed(seed)

    paulis = ['X','Y','Z']

    gate_set = []
    coefficient_set = []

    for j in range(int(n)):
        cond = False
        while cond == False:
            sigma_j = np.random.choice(paulis)
            sigma_k = np.random.choice(paulis)

            pos_j = np.random.choice(range(n-1))
            pos_k = np.random.choice(range(pos_j,n))
            if pos_j == pos_k:
                pass
            else:
                int_string = 'I'*pos_j + sigma_j + 'I'*(pos_k-pos_j-1) + sigma_k + 'I'*(n-pos_k-1)
                if int_string in gate_set:   
                    pass
                else:
                    final_string = int_string             
                    gate_set.append(final_string)            
                    a = 2*(np.random.rand()-0.5)
                    coefficient_set.append(a)
                    cond = True
        
    gate_set_perceptron = []
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
        
        gate_set_perceptron.append(An)
        A = np.add(A, coefficient_set[i]*An)
    
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

def shadow_bound(error, observables, type='Pauli', failure_rate=0.01):
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)

    k_loc = 0
    for ob in observables:
        k_int = calc_klocality(ob)
        if k_int > k_loc:
            k_loc = k_int

    return int(np.ceil(np.log(M)*(3**k_loc)/error**2)), K