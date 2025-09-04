import pennylane as qml
from pennylane import numpy as np
from scipy.linalg import sqrtm
from Unitary_Decomposition_Pauli_strings import *

def setup_system(n):
    N = 2**n  

    A = np.zeros((N, N), dtype = complex)
    b = np.zeros(N)
    
    phi_top = 1
    phi_bottom = 0
    phi_left = 0
    phi_right = 0
    for i in range(int(np.sqrt(N))):
        for j in range(int(np.sqrt(N))):
            k = i * n + j
            # Fill in the matrix A
            A[k, k] = 4  # Main diagonal element
            if i > 0:
                A[k, k - n] = -1  # Above element
            else:
                b[k] += phi_top  # Top boundary condition
            if i < n - 1:
                A[k, k + n] = -1  # Below element
            else:
                b[k] += phi_bottom  # Bottom boundary condition
            if j > 0:
                A[k, k - 1] = -1  # Left element
            else:
                b[k] += phi_left  # Left boundary condition
            if j < n - 1:
                A[k, k + 1] = -1  # Right element
            else:
                b[k] += phi_right  # Right boundary condition
    return A, b

def decompose_unitaries(matrix):
    n  = matrix.shape[0]

    B = (1/2)*(matrix+np.conjugate(np.transpose(matrix)))
    C = (1/(2*complex(0,1)))*(matrix-np.conjugate(np.transpose(matrix)))

    Ub = B + complex(0,1)*sqrtm(np.eye(n) - np.matmul(B,B))
    Vb = B - complex(0,1)*sqrtm(np.eye(n) - np.matmul(B,B))

    Uc = C + complex(0,1)*sqrtm(np.eye(n) - np.matmul(C,C))
    Vc = C - complex(0,1)*sqrtm(np.eye(n) - np.matmul(C,C))

    Ub[abs(Ub)<1e-15] = 0
    Vb[abs(Vb)<1e-15] = 0
    Uc[abs(Uc)<1e-15] = 0
    Vc[abs(Vc)<1e-15] = 0
    
    return B, C, Ub, Vb, Uc, Vc

def Potential_problem(n):
    A, b = setup_system(n)
    A_norm = A/np.linalg.norm(A)
    b_norm  = b/np.linalg.norm(b)

    B_d, C_sparse, Ub_sparse, Vb_sparse, Uc_sparse, Vc_sparse = decompose_unitaries(A_norm)

    string_list = []
    coeff_list = []

    Ub_decomp = PauliDecomposition(Ub_sparse)

    for idx,coeff in enumerate(Ub_decomp[1]):
        if abs(coeff)>1e-15:
            string_list.append(Ub_decomp[0][idx])
            coeff_list.append(0.5*Ub_decomp[1][idx])

    Uc_decomp = PauliDecomposition(Uc_sparse)

    for idx,coeff in enumerate(Uc_decomp[1]):
        if abs(coeff)>1e-15:
            if Uc_decomp[0][idx] in string_list:
                coeff_list[string_list.index(Uc_decomp[0][idx])] += complex(0,0.5)*coeff
            else:
                string_list.append(Uc_decomp[0][idx])
                coeff_list.append(complex(0,0.5)*Uc_decomp[1][idx])

    Vb_decomp = PauliDecomposition(Vb_sparse)

    for idx,coeff in enumerate(Vb_decomp[1]):
        if abs(coeff)>1e-15:
            if Vb_decomp[0][idx] in string_list:
                coeff_list[string_list.index(Vb_decomp[0][idx])] += 0.5*coeff
            else:
                string_list.append(Vb_decomp[0][idx])
                coeff_list.append(0.5*Vb_decomp[1][idx])

    Vc_decomp = PauliDecomposition(Vc_sparse)

    for idx,coeff in enumerate(Vc_decomp[1]):
        if abs(coeff)>1e-15:
            if Vc_decomp[0][idx] in string_list:
                coeff_list[string_list.index(Vc_decomp[0][idx])] += complex(0,0.5)*coeff
            else:
                string_list.append(Vc_decomp[0][idx])
                coeff_list.append(complex(0,0.5)*Vc_decomp[1][idx])

    return string_list, np.array(coeff_list), A_norm, b_norm
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