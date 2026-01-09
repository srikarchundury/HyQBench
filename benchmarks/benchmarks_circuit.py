"""
Benchmark circuit builders for hybrid CV-DV quantum systems.

This module provides functions to construct benchmark circuits for various
quantum algorithms including state preparation, state transfer, VQE, QAOA,
QFT, JCH simulation, and Shor's algorithm.
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate

# Add parent directory for custom_gates imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PARENT_DIR)

import c2qa
from custom_gates import state_transfer, jch_sim, bosonic_vqe, shors, bosonic_qaoa


# =============================================================================
# State Preparation Circuits
# =============================================================================

def cat_state_circuit(cutoff: int, circuit: c2qa.CVCircuit,
                      qbr: QuantumRegister, qmr: c2qa.QumodeRegister,
                      alpha: float) -> c2qa.CVCircuit:
    """
    Generate a cat state (superposition of coherent states).

    Creates |cat> = |alpha> + |-alpha> using conditional displacements.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: CVCircuit to append gates to.
        qbr: Qubit register.
        qmr: Qumode register.
        alpha: Coherent state amplitude.

    Returns:
        Modified circuit with cat state preparation.
    """
    circuit.h(qbr[0])
    circuit.cv_c_d(alpha / np.sqrt(2), qmr[0], qbr[0])
    circuit.h(qbr[0])

    circuit.sdg(qbr[0])
    circuit.h(qbr[0])
    circuit.cv_c_d(1j * np.pi / (8 * alpha * np.sqrt(2)), qmr[0], qbr[0])
    circuit.h(qbr[0])
    circuit.s(qbr[0])

    return circuit


def gkp_state_circuit(cutoff: int, circuit: c2qa.CVCircuit,
                      qbr: QuantumRegister, qmr: c2qa.QumodeRegister,
                      N_rounds: int = 9, r: float = 0.222,
                      qumode_idx: int = 0) -> c2qa.CVCircuit:
    """
    Generate a GKP (Gottesman-Kitaev-Preskill) state.

    Creates an approximate GKP state using iterative conditional displacements.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: CVCircuit to append gates to.
        qbr: Qubit register.
        qmr: Qumode register.
        N_rounds: Number of preparation rounds.
        r: Squeezing parameter.
        qumode_idx: Index of qumode to prepare (default 0).

    Returns:
        Modified circuit with GKP state preparation.
    """
    alpha = np.sqrt(np.pi)
    circuit.cv_sq(r, qmr[qumode_idx])

    for _ in range(1, N_rounds):
        circuit.h(qbr[0])
        circuit.cv_c_d(alpha / np.sqrt(2), qmr[qumode_idx], qbr[0])
        circuit.h(qbr[0])

        circuit.sdg(qbr[0])
        circuit.h(qbr[0])
        circuit.cv_c_d(1j * np.pi / (8 * alpha * np.sqrt(2)), qmr[qumode_idx], qbr[0])
        circuit.h(qbr[0])
        circuit.s(qbr[0])

    return circuit


# =============================================================================
# Basis Transformation
# =============================================================================

def apply_basis_transformation(circuit: c2qa.CVCircuit,
                               qbr: QuantumRegister) -> None:
    """
    Apply basis transformation for CV-DV state transfer.

    Transforms computational basis for measurement in the appropriate basis.

    Args:
        circuit: CVCircuit to append gates to.
        qbr: Qubit register to transform.
    """
    num_qubits = len(qbr)
    for i in range(num_qubits):
        circuit.h(qbr[i])
        if i == num_qubits - 1:  # MSB
            circuit.x(qbr[i])
            circuit.z(qbr[i])
        elif i == 0:  # LSB
            circuit.z(qbr[i])
        else:  # Middle qubits
            circuit.x(qbr[i])


def apply_basis_transformation_reverse(circuit: c2qa.CVCircuit,
                                        qbr: QuantumRegister) -> None:
    """
    Apply reverse basis transformation for CV-DV state transfer.

    Inverse of apply_basis_transformation.

    Args:
        circuit: CVCircuit to append gates to.
        qbr: Qubit register to transform.
    """
    num_qubits = len(qbr)
    for i in range(num_qubits):
        if i == num_qubits - 1:  # MSB
            circuit.z(qbr[i])
            circuit.x(qbr[i])
            circuit.h(qbr[i])
        elif i == 0:  # LSB
            circuit.z(qbr[i])
            circuit.h(qbr[i])
        else:  # Middle qubits
            circuit.x(qbr[i])
            circuit.h(qbr[i])


# =============================================================================
# State Transfer Circuits
# =============================================================================

def state_transfer_CVtoDV(cutoff: int, circuit: c2qa.CVCircuit,
                          qmr: c2qa.QumodeRegister, qbr: QuantumRegister,
                          cr: ClassicalRegister, n: int,
                          lmbda: float = 0.29,
                          apply_basis: bool = True,
                          measure: bool = True) -> c2qa.CVCircuit:
    """
    Transfer quantum state from continuous variable to discrete variable.

    Applies the non-abelian state transfer protocol using V and W gates.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: CVCircuit to append gates to.
        qmr: Qumode register.
        qbr: Qubit register.
        cr: Classical register for measurements.
        n: Number of qubits.
        lmbda: Coupling strength parameter (default 0.29).
        apply_basis: Whether to apply basis transformation (default True).
        measure: Whether to add measurement gates (default True).

    Returns:
        Modified circuit with state transfer.
    """
    for j in range(1, n + 1):
        V_j = state_transfer.Vj(lmbda, j, n, cutoff)
        gate = UnitaryGate(V_j.full(), label=f'V{j}')
        circuit.append(gate, qmr[:] + qbr[:])

        W_j = state_transfer.Wj(lmbda, j, n, cutoff)
        gate = UnitaryGate(W_j.full(), label=f'W{j}')
        circuit.append(gate, qmr[:] + qbr[:])

    if apply_basis:
        apply_basis_transformation(circuit, qbr)

    if measure:
        for i in range(n):
            circuit.measure(qbr[i], cr[-(i + 1)])

    return circuit


def state_transfer_DVtoCV(cutoff: int, circuit: c2qa.CVCircuit,
                          qmr: c2qa.QumodeRegister, qbr: QuantumRegister,
                          cr: ClassicalRegister, n: int,
                          lmbda: float = 0.29,
                          apply_basis: bool = True,
                          measure: bool = True) -> c2qa.CVCircuit:
    """
    Transfer quantum state from discrete variable to continuous variable.

    Inverse of state_transfer_CVtoDV.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: CVCircuit to append gates to.
        qmr: Qumode register.
        qbr: Qubit register.
        cr: Classical register for measurements.
        n: Number of qubits.
        lmbda: Coupling strength parameter (default 0.29).
        apply_basis: Whether to apply basis transformation (default True).
        measure: Whether to add measurement gates (default True).

    Returns:
        Modified circuit with state transfer.
    """
    if apply_basis:
        apply_basis_transformation_reverse(circuit, qbr)

    for j in range(n, 0, -1):
        W_j = state_transfer.Wj(lmbda, j, n, cutoff)
        gate = UnitaryGate(W_j.full(), label=f'W{j}')
        circuit.append(gate, qmr[:] + qbr[:])

        V_j = state_transfer.Vj(lmbda, j, n, cutoff)
        gate = UnitaryGate(V_j.full(), label=f'V{j}')
        circuit.append(gate, qmr[:] + qbr[:])

    if measure:
        for i in range(n):
            circuit.measure(qbr[i], cr[-(i + 1)])

    return circuit


# =============================================================================
# JCH Simulation Circuits
# =============================================================================

def JCH_simulation_circuit(Nsites: int, Nqubits: int, cutoff: int,
                           J: float, omega_r: float, omega_q: float,
                           g: float, tau: float):
    """
    Create a single timestep JCH simulation circuit as a gate.

    The circuit implements one Trotter step of the JCH Hamiltonian.

    Args:
        Nsites: Number of CV modes (cavity sites).
        Nqubits: Number of qubits.
        cutoff: Fock space cutoff.
        J: Hopping strength.
        omega_r: Resonator frequency.
        omega_q: Qubit frequency.
        g: Coupling strength.
        tau: Time step.

    Returns:
        Qiskit Gate for one timestep evolution.
    """
    return jch_sim.createCircuit(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau)


def JCH_simulation_circuit_unitary(Nsites: int, Nqubits: int, cutoff: int,
                                   J: float, omega_r: float, omega_q: float,
                                   g: float, tau: float):
    """
    Get the unitary gate for one JCH simulation timestep.

    Args:
        Nsites: Number of CV modes (cavity sites).
        Nqubits: Number of qubits.
        cutoff: Fock space cutoff.
        J: Hopping strength.
        omega_r: Resonator frequency.
        omega_q: Qubit frequency.
        g: Coupling strength.
        tau: Time step.

    Returns:
        Qiskit Gate for one timestep evolution.
    """
    return jch_sim.createCircuit(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau)


def JCH_simulation_circuit_display(Nsites: int, Nqubits: int, cutoff: int,
                                   J: float, omega_r: float, omega_q: float,
                                   g: float, tau: float,
                                   timesteps: int) -> c2qa.CVCircuit:
    """
    Get the full JCH simulation circuit for multiple timesteps.

    Args:
        Nsites: Number of CV modes (cavity sites).
        Nqubits: Number of qubits.
        cutoff: Fock space cutoff.
        J: Hopping strength.
        omega_r: Resonator frequency.
        omega_q: Qubit frequency.
        g: Coupling strength.
        tau: Time step.
        timesteps: Number of timesteps.

    Returns:
        CVCircuit with unrolled timesteps.
    """
    return jch_sim.circuit_display(Nsites, Nqubits, cutoff, J, omega_r, omega_q, g, tau, timesteps)


# =============================================================================
# VQE Circuits
# =============================================================================

def binary_knapsack_vqe(H, ndepth: int, nfocks: list,
                        maxiter: int = 100, method: str = 'BFGS',
                        verb: int = 1, threshold: float = 1e-9,
                        print_freq: int = 10, Xvec: np.ndarray = None) -> tuple:
    """
    Run VQE optimization for binary knapsack problem.

    Args:
        H: QuTiP Hamiltonian operator.
        ndepth: Circuit depth.
        nfocks: Fock cutoffs [n1, n2].
        maxiter: Maximum iterations.
        method: Optimization method.
        verb: Verbosity level.
        threshold: Convergence threshold.
        print_freq: Print frequency.
        Xvec: Initial parameters.

    Returns:
        Tuple of (final_energy, optimal_params, intermediate_results).
    """
    return bosonic_vqe.ecd_opt_vqe(
        H, ndepth, nfocks, maxiter=maxiter, method=method,
        verb=verb, threshold=threshold, print_freq=print_freq, Xvec=Xvec
    )


def binary_knapsack_vqe_circuit(H, ndepth: int, nfocks: list,
                                Xvec: np.ndarray = None) -> c2qa.CVCircuit:
    """
    Build VQE circuit with given or random parameters.

    Args:
        H: QuTiP Hamiltonian (unused, kept for API compatibility).
        ndepth: Circuit depth.
        nfocks: Fock cutoffs [n1, n2].
        Xvec: Parameter array (random if None).

    Returns:
        CVCircuit with VQE ansatz.
    """
    # Initialize parameters if not provided
    if Xvec is None or len(Xvec) == 0:
        beta_mag = np.random.uniform(0, 3, size=(ndepth, 2))
        beta_arg = np.random.uniform(0, np.pi, size=(ndepth, 2))
        theta = np.random.uniform(0, np.pi, size=(ndepth, 2))
        phi = np.random.uniform(0, np.pi, size=(ndepth, 2))
        Xvec = bosonic_vqe.pack_variables(beta_mag, beta_arg, theta, phi)

    return bosonic_vqe.build_vqe_circuit(Xvec, ndepth, nfocks)


# =============================================================================
# QAOA Circuits
# =============================================================================

def cv_qaoa(cutoff: int, s: float, a: float, p: int, n: int,
            maxiter: int = 100, method: str = 'SLSQP'):
    """
    Run CV-QAOA optimization.

    Args:
        cutoff: Fock space cutoff.
        s: Squeezing parameter.
        a: Target position value.
        p: Number of QAOA layers.
        n: Cost function power.
        maxiter: Maximum iterations.
        method: Optimization method.

    Returns:
        Scipy OptimizeResult.
    """
    costval = []
    estval = []

    def cost_function(params):
        return bosonic_qaoa.cvQAOA(params, cutoff, p, s, n, a, costval, estval)

    initial_params = np.random.uniform(0, 2 * np.pi, size=2 * p)

    result = minimize(
        cost_function,
        initial_params,
        method=method,
        tol=1e-6,
        options={'maxiter': maxiter}
    )

    return result


def cv_qaoa_circuit(params: np.ndarray, cutoff: int, s: float,
                    a: float, p: int, n: int) -> c2qa.CVCircuit:
    """
    Build CV-QAOA circuit with given parameters.

    Args:
        params: Parameter array [gamma_0, ..., gamma_{p-1}, eta_0, ..., eta_{p-1}].
        cutoff: Fock space cutoff.
        s: Squeezing parameter.
        a: Target position value.
        p: Number of QAOA layers.
        n: Cost function power.

    Returns:
        CVCircuit with QAOA layers.
    """
    return bosonic_qaoa.build_qaoa_circuit(params, cutoff, p, s, n, a)


# =============================================================================
# Shor's Algorithm Circuit
# =============================================================================

def shors_circuit(N: int, m: int, R: int, a: int,
                  delta: float, cutoff: int) -> c2qa.CVCircuit:
    """
    Build Shor's factoring algorithm circuit.

    Args:
        N: Number to factor.
        m: Modular exponentiation parameter.
        R: Reference parameter.
        a: Base for exponentiation.
        delta: GKP squeezing parameter.
        cutoff: Fock space cutoff.

    Returns:
        CVCircuit for Shor's algorithm.
    """
    qmr = c2qa.QumodeRegister(
        num_qumodes=3,
        num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))),
        name='qumode'
    )
    qbr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr, qbr, cr)

    # GKP state preparation on qumodes 0 and 1
    circuit = gkp_state_circuit(cutoff, circuit, qbr, qmr, qumode_idx=0)
    circuit = gkp_state_circuit(cutoff, circuit, qbr, qmr, qumode_idx=1)

    # Squeezing on qumode 2
    circuit.cv_sq(-np.log(delta), qmr[2])

    # Modular arithmetic
    circuit = shors.translation_R(cutoff, R, circuit, qmr, 0)
    circuit = shors.multiplication(cutoff, N, circuit, qmr, 1)
    circuit = shors.U_aNm(cutoff, circuit, qmr, qbr, a, N, m)

    return circuit


# =============================================================================
# QFT Circuit
# =============================================================================

def qft_circuit(cutoff: int, delta: float, n: int, a: int,
                append: int) -> c2qa.CVCircuit:
    """
    Build Quantum Fourier Transform circuit using CV-DV hybrid.

    Args:
        cutoff: Fock space cutoff.
        delta: Displacement parameter.
        n: Number of main qubits.
        a: Number of ancilla qubits.
        append: Number of append qubits.

    Returns:
        CVCircuit for QFT.
    """
    total = n + a + append

    qmr = c2qa.QumodeRegister(
        1, num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))), name='qumode'
    )
    qbr1 = QuantumRegister(n, name='qbits')
    append_reg = QuantumRegister(append, name='append')
    ancilla_reg = QuantumRegister(a, name='ancilla')
    creg = ClassicalRegister(n, name='creg')

    circuit = c2qa.CVCircuit(qmr, ancilla_reg, qbr1, append_reg, creg)

    # Initialize ancillas
    for q in ancilla_reg:
        circuit.h(q)

    total_reg = ancilla_reg[:] + qbr1[:] + append_reg[:]
    reversed_reg = list(reversed(total_reg))

    # Basis preparation
    circuit.z(reversed_reg[0])
    circuit.z(reversed_reg[-1])
    circuit.h(reversed_reg[-1])

    for q in reversed_reg[:-1]:
        circuit.x(q)
        circuit.h(q)

    # DV to CV transfer (without basis transformation or measurement)
    state_transfer_DVtoCV(cutoff, circuit, qmr, total_reg, creg, total,
                          apply_basis=False, measure=False)

    # QFT operations in CV space
    delta_prime = (2 * np.pi) / (2**total * delta)
    circuit.cv_d(delta / 2, qmr[0])
    circuit.cv_r(np.pi / 2, qmr[0])
    circuit.cv_d(-delta_prime / 2, qmr[0])

    # CV to DV transfer (without basis transformation or measurement)
    state_transfer_CVtoDV(cutoff, circuit, qmr, total_reg, creg, total,
                          apply_basis=False, measure=False)

    # Reverse basis preparation
    for q in reversed_reg[:-1]:
        circuit.h(q)
        circuit.x(q)
    circuit.h(reversed_reg[-1])
    circuit.z(reversed_reg[0])
    circuit.z(reversed_reg[-1])

    # Measure main qubits
    start_index = a
    for i in range(n):
        circuit.measure(total_reg[start_index + i], creg[i])

    return circuit


