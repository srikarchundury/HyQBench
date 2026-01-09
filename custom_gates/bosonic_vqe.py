"""
Bosonic Variational Quantum Eigensolver (VQE) with ECD ansatz.

This module implements a circuit-based VQE using the Echoed Conditional
Displacement (ECD) ansatz for hybrid qubit-qumode systems.
"""

import numpy as np
import qutip as qt
import qiskit
import c2qa
from qiskit import QuantumRegister
from qiskit.circuit.library import UnitaryGate
from functools import partial
from scipy.optimize import minimize

from .hamiltonian_utils import (
    qproj00, qproj11, qproj01, qproj10,
    binary_knapsack_ham, binary_to_pauli_list,
    generate_tensor_product, qubit_op_to_ham
)
from .vqe_utils import (
    pack_variables, unpack_variables, get_cvec_np,
    qubit_rot, ecd_op, generate_triples
)


# =============================================================================
# ECD Circuit Operations
# =============================================================================

def ecd_rot_op(beta: np.ndarray, theta: np.ndarray, phi: np.ndarray,
               nfocks: list, circuit: c2qa.CVCircuit,
               qmr: c2qa.QumodeRegister, qmr1: c2qa.QumodeRegister,
               qbr: QuantumRegister) -> c2qa.CVCircuit:
    """
    Apply an ECD-rotation block to the circuit.

    The block consists of: R1 -> ECD1 -> R2 -> ECD2, where R are qubit
    rotations and ECD are echoed conditional displacements.

    Args:
        beta: Array of two complex displacement parameters.
        theta: Array of two rotation angles.
        phi: Array of two rotation phases.
        nfocks: Fock cutoffs [n1, n2] for the two qumodes.
        circuit: The CVCircuit to append gates to.
        qmr: First qumode register.
        qmr1: Second qumode register.
        qbr: Qubit register.

    Returns:
        The modified circuit with the ECD-rotation block appended.
    """
    # Qubit rotations
    R1 = qubit_rot(theta[0], phi[0])
    R2 = qubit_rot(theta[1], phi[1])

    # ECD operators
    E1 = ecd_op(beta[0], theta[0], phi[0], 0, nfocks)
    E2 = ecd_op(beta[1], theta[1], phi[1], 1, nfocks)

    # Append gates to circuit
    r1_gate = UnitaryGate(R1.full(), label='R1')
    circuit.append(r1_gate, qbr[:])

    e1_gate = UnitaryGate(E1.full(), label='ECD1')
    circuit.append(e1_gate, qmr1[:] + qmr[:] + qbr[:])

    r2_gate = UnitaryGate(R2.full(), label='R2')
    circuit.append(r2_gate, qbr[:])

    e2_gate = UnitaryGate(E2.full(), label='ECD2')
    circuit.append(e2_gate, qmr1[:] + qmr[:] + qbr[:])

    return circuit


def ecd_rot_ansatz(bmag_mat: np.ndarray, barg_mat: np.ndarray,
                   theta_mat: np.ndarray, phi_mat: np.ndarray,
                   nfocks: list, circuit: c2qa.CVCircuit,
                   qmr: c2qa.QumodeRegister, qmr1: c2qa.QumodeRegister,
                   qbr: QuantumRegister) -> c2qa.CVCircuit:
    """
    Apply the full ECD-rotation ansatz to the circuit.

    Args:
        bmag_mat: ECD displacement magnitudes, shape (ndepth, 2).
        barg_mat: ECD displacement phases, shape (ndepth, 2).
        theta_mat: Qubit rotation angles, shape (ndepth, 2).
        phi_mat: Qubit rotation phases, shape (ndepth, 2).
        nfocks: Fock cutoffs [n1, n2] for the two qumodes.
        circuit: The CVCircuit to append gates to.
        qmr: First qumode register.
        qmr1: Second qumode register.
        qbr: Qubit register.

    Returns:
        The modified circuit with the full ansatz appended.

    Raises:
        ValueError: If parameter matrix dimensions don't match.
    """
    # Validate dimensions
    if bmag_mat.shape != barg_mat.shape:
        raise ValueError("Dimensions of bmag_mat and barg_mat do not match.")

    beta_mat = get_cvec_np(bmag_mat, barg_mat)

    if beta_mat.shape != theta_mat.shape:
        raise ValueError("Dimensions of beta_mat and theta_mat do not match.")
    if beta_mat.shape != phi_mat.shape:
        raise ValueError("Dimensions of beta_mat and phi_mat do not match.")

    ndepth = beta_mat.shape[0]

    # Apply ECD-rotation blocks
    for i in range(ndepth):
        circuit = ecd_rot_op(
            beta_mat[i, :], theta_mat[i, :], phi_mat[i, :],
            nfocks, circuit, qmr, qmr1, qbr
        )

    return circuit


# =============================================================================
# VQE Core Functions
# =============================================================================

def _create_vqe_circuit(nfocks: list) -> tuple:
    """
    Create the circuit and registers for VQE.

    Args:
        nfocks: Fock cutoffs [n1, n2] for the two qumodes.

    Returns:
        Tuple of (circuit, qmr, qmr1, qbr).
    """
    qmr = c2qa.QumodeRegister(
        num_qumodes=1,
        num_qubits_per_qumode=int(np.ceil(np.log2(nfocks[0]))),
        name='qmr'
    )
    qmr1 = c2qa.QumodeRegister(
        num_qumodes=1,
        num_qubits_per_qumode=int(np.ceil(np.log2(nfocks[1]))),
        name='qmr1'
    )
    qbr = QuantumRegister(1, name='qbit')
    circuit = c2qa.CVCircuit(qmr1, qmr, qbr)

    return circuit, qmr, qmr1, qbr


def energy_val(Xvec: np.ndarray, ndepth: int, nfocks: list, H: qt.Qobj) -> float:
    """
    Compute the expectation value <psi|H|psi> for the VQE ansatz.

    Args:
        Xvec: Flattened parameter array for the ansatz.
        ndepth: Circuit depth (number of ECD layers).
        nfocks: Fock cutoffs [n1, n2] for the two qumodes.
        H: QuTiP Hamiltonian operator.

    Returns:
        Energy expectation value.
    """
    # Create circuit
    circuit, qmr, qmr1, qbr = _create_vqe_circuit(nfocks)

    # Unpack parameters and build ansatz
    beta_mag, beta_arg, theta, phi = unpack_variables(Xvec, ndepth)
    circuit = ecd_rot_ansatz(
        beta_mag, beta_arg, theta, phi,
        nfocks, circuit, qmr, qmr1, qbr
    )

    # Simulate
    state, _, _ = c2qa.util.simulate(circuit)

    # Compute expectation value
    ham = qt.Qobj(H.full())
    energy = qt.expect(ham, qt.Qobj(state))

    return energy


def ecd_opt_vqe(H: qt.Qobj, ndepth: int, nfocks: list,
                maxiter: int = 100, method: str = 'COBYLA',
                verb: int = 0, threshold: float = 1e-08,
                print_freq: int = 10, Xvec: np.ndarray = None) -> tuple:
    """
    Optimize the ECD ansatz parameters to minimize the Hamiltonian energy.

    Args:
        H: QuTiP Hamiltonian operator.
        ndepth: Circuit depth (number of ECD layers).
        nfocks: Fock cutoffs [n1, n2] for the two qumodes.
        maxiter: Maximum number of optimization iterations.
        method: Scipy optimization method (e.g., 'COBYLA', 'BFGS').
        verb: Verbosity level (0=silent, 1=verbose).
        threshold: Convergence tolerance.
        print_freq: Frequency of printing intermediate results.
        Xvec: Optional initial parameter guess.

    Returns:
        Tuple of (final_energy, optimal_params, intermediate_results).
    """
    # Parameter bounds
    size = ndepth * 2
    bounds = (
        [(0.0, 10.0)] * size +           # beta_mag
        [(0.0, 2 * np.pi)] * size +      # beta_arg
        [(0.0, np.pi)] * size +          # theta
        [(0.0, 2 * np.pi)] * size        # phi
    )

    # Initialize parameters if not provided
    if Xvec is None or len(Xvec) == 0:
        beta_mag = np.random.uniform(0, 3, size=(ndepth, 2))
        beta_arg = np.random.uniform(0, np.pi, size=(ndepth, 2))
        theta = np.random.uniform(0, np.pi, size=(ndepth, 2))
        phi = np.random.uniform(0, np.pi, size=(ndepth, 2))
        Xvec = pack_variables(beta_mag, beta_arg, theta, phi)

    # Objective function
    obj_fun = partial(energy_val, ndepth=ndepth, nfocks=nfocks, H=H)

    # Tracking
    iteration_step = 0
    intermediate_results = []

    def callback(xk):
        nonlocal iteration_step
        iteration_step += 1
        loss_value = obj_fun(xk)

        if verb == 1 and (iteration_step % print_freq == 0):
            print("-------------------")
            print(f"iter: {iteration_step}")
            print(f"fval: {loss_value}")

        if iteration_step % print_freq == 0:
            intermediate_results.append((loss_value, xk.copy()))

    # Optimize
    options = {'disp': True, 'maxiter': maxiter}
    result = minimize(
        obj_fun, Xvec, method=method, bounds=bounds,
        tol=threshold, options=options, callback=callback
    )

    return result.fun, result.x, intermediate_results


# =============================================================================
# State Analysis Functions
# =============================================================================

def num_prob_basis(Xvec: np.ndarray, nvec: np.ndarray,
                   ndepth: int, nfocks: list) -> float:
    """
    Compute |<psi|q,n,m>|^2 for a specific basis state.

    Args:
        Xvec: Ansatz parameters.
        nvec: Basis state indices [qubit, fock_n, fock_m].
        ndepth: Circuit depth.
        nfocks: Fock cutoffs [n1, n2].

    Returns:
        Probability of measuring the specified basis state.
    """
    # Create and run circuit
    circuit, qmr, qmr1, qbr = _create_vqe_circuit(nfocks)

    beta_mag, beta_arg, theta, phi = unpack_variables(Xvec, ndepth)
    circuit = ecd_rot_ansatz(
        beta_mag, beta_arg, theta, phi,
        nfocks, circuit, qmr, qmr1, qbr
    )

    psi, _, _ = c2qa.util.simulate(circuit)

    # Construct basis state |q, n, m>
    basis_state = qt.tensor(
        qt.basis(2, int(nvec[0])),
        qt.basis(nfocks[0], int(nvec[1])),
        qt.basis(nfocks[1], int(nvec[2]))
    )

    # Compute overlap
    overlap = qt.Qobj(psi).overlap(basis_state)
    return np.abs(overlap) ** 2


def num_prob_all(Xvec: np.ndarray, ndepth: int, nfocks: list) -> np.ndarray:
    """
    Compute |<psi|q,n,m>|^2 for all basis states.

    Args:
        Xvec: Ansatz parameters.
        ndepth: Circuit depth.
        nfocks: Fock cutoffs [n1, n2].

    Returns:
        Array of probabilities for all basis states.
    """
    triples = generate_triples(nfocks)
    probabilities = [
        num_prob_basis(Xvec, triples[i, :], ndepth, nfocks)
        for i in range(triples.shape[0])
    ]
    return np.array(probabilities)


# =============================================================================
# Circuit Builder (for benchmarks)
# =============================================================================

def build_vqe_circuit(Xvec: np.ndarray, ndepth: int,
                      nfocks: list) -> c2qa.CVCircuit:
    """
    Build a VQE circuit with given parameters.

    Args:
        Xvec: Ansatz parameters.
        ndepth: Circuit depth.
        nfocks: Fock cutoffs [n1, n2].

    Returns:
        The constructed CVCircuit.
    """
    circuit, qmr, qmr1, qbr = _create_vqe_circuit(nfocks)

    beta_mag, beta_arg, theta, phi = unpack_variables(Xvec, ndepth)
    circuit = ecd_rot_ansatz(
        beta_mag, beta_arg, theta, phi,
        nfocks, circuit, qmr, qmr1, qbr
    )

    return circuit
