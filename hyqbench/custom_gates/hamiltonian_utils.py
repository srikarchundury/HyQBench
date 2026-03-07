"""
Hamiltonian utilities for quantum optimization problems.

This module provides functions for constructing and manipulating Hamiltonians,
including binary-to-Pauli mappings for optimization problems like the knapsack problem.
"""

import numpy as np
import sympy as sp
import qutip as qt
from qutip import Qobj, qeye, sigmax, sigmay, sigmaz, tensor


# =============================================================================
# Qubit Projectors
# =============================================================================

def qproj00() -> Qobj:
    """Return the |0><0| projector for a qubit."""
    return qt.basis(2, 0).proj()


def qproj11() -> Qobj:
    """Return the |1><1| projector for a qubit."""
    return qt.basis(2, 1).proj()


def qproj01() -> Qobj:
    """Return the |0><1| off-diagonal projector for a qubit."""
    op = np.array([[0, 1], [0, 0]])
    return qt.Qobj(op)


def qproj10() -> Qobj:
    """Return the |1><0| off-diagonal projector for a qubit."""
    op = np.array([[0, 0], [1, 0]])
    return qt.Qobj(op)


# =============================================================================
# Binary Hamiltonian Construction
# =============================================================================

def binary_knapsack_ham(l_val: float, values: list, weights: list,
                        max_weight: int, include_id: bool = False):
    """
    Generate the binary Hamiltonian for the knapsack problem.

    The knapsack problem aims to maximize value while staying within weight capacity.
    This function constructs a QUBO (Quadratic Unconstrained Binary Optimization)
    Hamiltonian for the problem.

    Args:
        l_val: Lambda penalty parameter for constraint violation.
        values: List of item values.
        weights: List of item weights.
        max_weight: Total weight capacity.
        include_id: If True, use symbolic identity; if False, use value 1.

    Returns:
        tuple: (H_total, symbol_list) where H_total is the binary Hamiltonian
               as a SymPy expression and symbol_list contains all binary variables.
    """
    # Number of primary variables
    N_qb = len(weights)

    # Symbols
    symbol_list = list(sp.symbols(f'x_:{N_qb}'))
    Ident = sp.symbols(r'\mathbb{I}') if include_id else 1.0

    # E = sum(i) v(i) x(i)
    H_prob = sum(values[ii] * symbol_list[ii] for ii in range(N_qb))

    # Calculate scaling factor
    max_weight_bin_str = bin(max_weight).lstrip('0b')
    max_val = 2 ** len(max_weight_bin_str) - 1
    scaling_factor = max_val / max_weight

    # Apply scaling to weights
    scaled_weights = [weight * scaling_factor for weight in weights]

    # W0 = sum(i) w(i) x(i)
    H_constraints = sum(scaled_weights[ii] * symbol_list[ii] for ii in range(N_qb))

    # Bitstring representation converted to list
    bin_weight = list(bin(max_weight).lstrip('0b'))[::-1]

    # Auxiliary symbol indices start after the primary variables
    aux_symbols = sp.symbols(f'x_{N_qb}:{N_qb + len(bin_weight)}')

    # A = sum(i) 2^i y(i)
    H_constraints_aux = sum(aux_symbols[ii] * 2**ii for ii in range(len(bin_weight)))

    # Full Hamiltonian
    H_total = -H_prob + l_val * (max_weight - H_constraints - H_constraints_aux)**2

    # Construct full list of symbols in expression
    symbol_list.extend(list(aux_symbols))

    # Create a list of x_j**2
    sq_syms = [temp_sym**2 for temp_sym in symbol_list]

    # Maps x_j^{2} to x_j (binary constraint)
    conv_dict = dict(zip(sq_syms, list(symbol_list)))

    # Final binary Hamiltonian
    H_total = H_total.subs(conv_dict)

    return H_total, symbol_list


# =============================================================================
# Binary to Spin/Pauli Conversion
# =============================================================================

def binary_to_qubit_ham(H_bin, symbol_list: list, include_id: bool = False):
    """
    Map a symbolic binary Hamiltonian to a spin Hamiltonian.

    Performs the substitution x_i -> (1/2)(I - Z_i) to convert binary
    variables to Pauli Z operators.

    Args:
        H_bin: The binary Hamiltonian as a SymPy expression.
        symbol_list: SymPy symbols defining the Hamiltonian variables.
        include_id: If True, use symbolic identity; if False, use value 1.

    Returns:
        SymPy expression representing the spin Hamiltonian.
    """
    # Initialize spin variables (z0, z1, ..., zn)
    z_symbols = sp.symbols(f'z:{len(symbol_list)}')

    # Define the identity operator
    Ident = sp.symbols(r'\mathbb{I}') if include_id else 1.0

    # Create mapping: x_i -> (1/2)(I - Z_i)
    bin2spin_dict = {
        symbol: (1/2) * (Ident - z) for symbol, z in zip(symbol_list, z_symbols)
    }

    # Convert binary Hamiltonian to spin Hamiltonian
    spin_ham = H_bin.subs(bin2spin_dict).expand()

    # Z^2 = I
    sq_z = [z**2 for z in z_symbols]
    sq_values = [Ident] * len(z_symbols)
    spin_squared_map = dict(zip(sq_z, sq_values))

    # Substitute squared terms
    red_spin_ham = spin_ham.subs(spin_squared_map)

    return red_spin_ham


def _check_spinz(input_list: list, spinz) -> list:
    """
    Helper function to construct Pauli string from Z indices.

    Args:
        input_list: List of spin variable strings (e.g., ['z0', 'z2']).
        spinz: Set of all spin symbols in the expression.

    Returns:
        List of Pauli characters ('I' or 'Z') for each qubit.
    """
    out_val = ['I'] * len(spinz)
    for ll in range(len(input_list)):
        out_val[int(input_list[ll].strip('z'))] = 'Z'
    return out_val


def sympy_to_pauli_dict(smpy_exp) -> dict:
    """
    Convert a SymPy spin Hamiltonian to a dictionary of Pauli terms.

    Args:
        smpy_exp: SymPy expression representing a spin Hamiltonian.

    Returns:
        Dictionary mapping Pauli word strings to coefficient strings.
    """
    # Determine the number of qubits
    spinz = smpy_exp.free_symbols

    # Split at spaces to get individual terms/coefficients
    split_expr = str(smpy_exp).split()

    # First term
    matrix_dict = {}
    split_term = split_expr[0].split('*')
    tmp_coeff = split_term[0]
    tmp_paulis = split_term[1:]
    pauli_word = ''.join(_check_spinz(tmp_paulis, spinz))
    matrix_dict[pauli_word] = tmp_coeff

    # Iterate through remaining terms
    for ii in range(1, len(split_expr), 2):
        tmp_sign = split_expr[ii]
        split_term = split_expr[ii + 1].split('*')
        tmp_coeff = split_term[0]
        tmp_paulis = split_term[1:]
        pauli_word = ''.join(_check_spinz(tmp_paulis, spinz))
        matrix_dict[pauli_word] = tmp_sign + tmp_coeff

    return matrix_dict


def binary_to_pauli_list(H_total, symbol_list: list) -> list:
    """
    Map a binary Hamiltonian to a list of Pauli terms with coefficients.

    Args:
        H_total: The binary Hamiltonian as a SymPy expression.
        symbol_list: Symbols defining the Hamiltonian variables.

    Returns:
        List of [pauli_string, coefficient] pairs.
    """
    spin_ham = binary_to_qubit_ham(H_total, symbol_list)
    op_dict = sympy_to_pauli_dict(spin_ham)
    return [[key, float(value)] for key, value in op_dict.items()]


# =============================================================================
# Pauli Operator Construction
# =============================================================================

def generate_tensor_product(pauli_string: str) -> Qobj:
    """
    Construct a QuTiP tensor product operator from a Pauli word string.

    Args:
        pauli_string: String of Pauli characters (e.g., 'IXYZ').

    Returns:
        QuTiP Qobj representing the tensor product of Pauli operators.
    """
    operator_map = {
        'I': qeye(2),
        'X': sigmax(),
        'Y': sigmay(),
        'Z': sigmaz()
    }

    operators = [operator_map[char] for char in pauli_string]
    U = tensor(*operators).full()
    return Qobj(U)


def qubit_op_to_ham(pterms: list) -> Qobj:
    """
    Construct a QuTiP Hamiltonian from Pauli terms and coefficients.

    Args:
        pterms: List of [pauli_string, coefficient] pairs.

    Returns:
        QuTiP Qobj representing the summed Hamiltonian.
    """
    terms = [coeff * generate_tensor_product(pauli_str) for pauli_str, coeff in pterms]
    return sum(terms)
