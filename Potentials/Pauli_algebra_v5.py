import math
import random
import pennylane as qml
from pennylane import numpy as np
from collections import Counter

def simplify_pauli_string(pauli_string):
    """
    Simplify a Pauli string expression.

    Parameters:
    pauli_string (str): Input Pauli string expression.

    Returns:
    str: Simplified Pauli string expression.
    """
    pauli_operators = {'I', 'X', 'Y', 'Z'}
    
    simplified_string = ''
    current_operator = ''
    current_multiplier = 1

    for char in pauli_string:
        if char in pauli_operators:
            if current_operator == '':
                current_operator = char
            else:
                if current_operator == 'I':
                    current_operator = char
                elif current_operator == char:
                    current_operator = 'I'
                    current_multiplier *= 1  
                else:
                    intermediate_str = current_operator + char
                    if 'X' in intermediate_str and 'Z' in intermediate_str:
                        current_multiplier *= complex(0,-1) if  current_operator=='X' else complex(0,1)
                        current_operator = 'Y'
                    elif 'X' in intermediate_str and 'Y' in intermediate_str:
                        current_multiplier *= complex(0,1) if  current_operator=='X' else complex(0,-1)
                        current_operator = 'Z'
                    elif 'Z' in intermediate_str and 'Y' in intermediate_str:
                        current_multiplier *= complex(0,-1) if  current_operator=='Z' else complex(0,1)
                        current_operator = 'X'

        elif char.isdigit():
            current_multiplier *= int(char)
    
    if current_multiplier == 1:
        simplified_string += current_operator
    elif current_multiplier == -1:
        simplified_string += '-' + current_operator
    elif current_multiplier==complex(0,1):
        simplified_string += 'i'+current_operator
    elif current_multiplier==complex(0,-1):
        simplified_string += '-i'+current_operator

    return simplified_string

def create_operator_list(A_strings, A_coeffs):
    ham = ['X']
    n = len(A_strings[0])

    normalization_strings = []
    normalization_coeffs = []

    for i in range(len(A_strings)):
        for j in range(0, len(A_strings)):
            coeff = A_coeffs[i]*np.conjugate(A_coeffs[j])
            if i!=j:
                intermediate_norm_str = ''
                intermediate_norm_coeff = 1
                for k in range(len(A_strings[0])):
                    pauli_str = A_strings[i][k] + A_strings[j][k]
                    simplified_result = simplify_pauli_string(pauli_str)
                    if len(simplified_result)==1:
                        intermediate_norm_str += simplified_result
                        intermediate_norm_coeff *= complex(1,0)
                    elif simplified_result[0] == '-' and len(simplified_result)==2:
                        intermediate_norm_str += simplified_result[-1]
                        intermediate_norm_coeff *= complex(-1,0)
                    elif simplified_result[0] == 'i' and len(simplified_result)==2:
                        intermediate_norm_str += simplified_result[-1]
                        intermediate_norm_coeff *= complex(0,1)
                    else:
                        intermediate_norm_str += simplified_result[-1]
                        intermediate_norm_coeff *= complex(0,-1)
                normalization_strings.append(intermediate_norm_str)
                normalization_coeffs.append(intermediate_norm_coeff*coeff)
    
    local_cost_strings = []
    local_cost_coeffs = []

    for i in range(len(A_strings)):
        for j in range(len(A_strings)):
            coeff = A_coeffs[i]*np.conjugate(A_coeffs[j])
            for m in range(len(ham)):
                for l in range(len(A_strings[0])):
                    intermediate_local_str = ''
                    intermediate_local_coeff = 1
                    for k in range(len(A_strings[0])):
                        if l==k:
                            if l==5 or l==6 or l==7:
                                pauli_str = A_strings[i][k] + 'X' + A_strings[j][k]
                            else:
                                pauli_str = A_strings[i][k] + 'Z' + A_strings[j][k]
                        else:
                            pauli_str = A_strings[i][k] + A_strings[j][k]
                        simplified_result = simplify_pauli_string(pauli_str)
                        if len(simplified_result)==1:
                            intermediate_local_str += simplified_result
                            intermediate_local_coeff *= complex(1,0)
                        elif simplified_result[0] == '-' and len(simplified_result)==2:
                            intermediate_local_str += simplified_result[-1]
                            intermediate_local_coeff *= complex(-1,0)
                        elif simplified_result[0] == 'i' and len(simplified_result)==2:
                            intermediate_local_str += simplified_result[-1]
                            intermediate_local_coeff *= complex(0,1)
                        else:
                            intermediate_local_str += simplified_result[-1]
                            intermediate_local_coeff *= complex(0,-1)
                    local_cost_strings.append(intermediate_local_str)
                    local_cost_coeffs.append(intermediate_local_coeff*coeff)
        
    counter_object_local_cost = Counter(local_cost_strings)
    keys_local_cost = counter_object_local_cost.keys()

    counter_object_norm = Counter(normalization_strings)
    keys_norm = counter_object_norm.keys()

    local_cost_calc = {}
    for key in keys_local_cost:
        counter = 0
        for i in range(len(local_cost_strings)):
            if local_cost_strings[i]==key:
                counter += local_cost_coeffs[i]
        if counter.real != 0 or counter.imag != 0:
            local_cost_calc[key] = counter

    norm_calc = {}
    for key in keys_norm:
        counter = 0
        for i in range(len(normalization_strings)):
            if normalization_strings[i]==key:
                counter += normalization_coeffs[i]
        if counter.real != 0 or counter.imag != 0:
            norm_calc[key] = counter

    observables = list(set(local_cost_calc.keys()) | set(norm_calc.keys()))
    if 'I'*len(A_strings[0]) in observables:
        observables.remove('I'*len(A_strings[0]))
    return observables, local_cost_calc, norm_calc

def calc_klocality(pstr):
    return pstr.count('X') + pstr.count('Y') + pstr.count('Z')

def shadow_bound(error, observables, type='1', failure_rate=0.01):
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)

    if type == '1':
        shadow_norm = (
                lambda op: np.linalg.norm(
                    op - np.trace(op) / 2 ** int(np.log2(op.shape[0])), ord=np.inf
                )
                ** 2
            )
        list_of_observables = ([qml.pauli.string_to_pauli_word(pstrs) for pstrs in observables])
        list_of_matrices = [qml.matrix(o) for o in list_of_observables]

        return int(np.ceil(np.log(M)*(3**(max(shadow_norm(o) for o in list_of_matrices)))/error**2)), int(K)
    else: 
        k_loc = 0
        for ob in observables:
            k_int = calc_klocality(ob)
            if k_int > k_loc:
                k_loc = k_int
        return int(np.ceil(np.log(M)*(3**k_loc)/error**2)), int(K)
