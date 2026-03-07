"""
Shor's Algorithm for quantum factorization.

This module implements a circuit-based version of Shor's algorithm using
hybrid qubit-qumode (CV-DV) systems. The implementation encodes integers
in continuous variable modes and performs modular exponentiation.
"""

import numpy as np
# import c2qa
import bosonic_qiskit as c2qa
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import partial_trace
from qutip import qeye, position, momentum, num, Qobj, tensor
from scipy.stats.contingency import margins

from .hamiltonian_utils import qproj00, qproj11


# =============================================================================
# Basic Quantum Gates
# =============================================================================

def hadamard() -> Qobj:
    """Return the Hadamard gate as a QuTiP operator."""
    op = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return Qobj(op)


# =============================================================================
# CV Displacement Operations
# =============================================================================

def Q_displacement_plus1(cutoff: int) -> Qobj:
    """
    Momentum displacement by +1 in position basis.

    Implements exp(-i * p) which translates position by +1.

    Args:
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP unitary operator.
    """
    return (-1j * momentum(cutoff)).expm()


def Q_displacement_minus1(cutoff: int) -> Qobj:
    """
    Momentum displacement by -1 in position basis.

    Implements exp(i * p) which translates position by -1.

    Args:
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP unitary operator.
    """
    return (1j * momentum(cutoff)).expm()


def single_mode_squeeze(cutoff: int, squeeze_param: float) -> Qobj:
    """
    Single-mode squeezing gate.

    Implements S(r) = exp(i * r * (x*p + p*x) / 2).

    Args:
        cutoff: Fock space cutoff dimension.
        squeeze_param: Squeezing parameter r.

    Returns:
        QuTiP unitary operator.
    """
    xp_anticomm = position(cutoff) * momentum(cutoff) + momentum(cutoff) * position(cutoff)
    return (1j * squeeze_param * xp_anticomm / 2).expm()


# =============================================================================
# Controlled CV Operations
# =============================================================================

def Q_control_plus1(cutoff: int) -> Qobj:
    """
    Qubit-controlled position displacement by +1.

    |0><0| x I + |1><1| x exp(-i*p)

    Args:
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP controlled unitary operator.
    """
    return tensor(qproj00(), qeye(cutoff)) + tensor(qproj11(), Q_displacement_plus1(cutoff))


def Q_control_minus1(cutoff: int) -> Qobj:
    """
    Qubit-controlled position displacement by -1.

    |0><0| x I + |1><1| x exp(i*p)

    Args:
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP controlled unitary operator.
    """
    return tensor(qproj00(), qeye(cutoff)) + tensor(qproj11(), Q_displacement_minus1(cutoff))


def rotation_control(cutoff: int, sign: int) -> Qobj:
    """
    Qubit-controlled phase rotation based on photon number.

    |0><0| x I + |1><1| x exp(i * pi/2 * sign * n)

    Args:
        cutoff: Fock space cutoff dimension.
        sign: +1 or -1 for rotation direction.

    Returns:
        QuTiP controlled unitary operator.
    """
    return tensor(qproj00(), qeye(cutoff)) + tensor(qproj11(), (1j * np.pi / 2 * sign * num(cutoff)).expm())


# =============================================================================
# Core Shor's Algorithm Operations
# =============================================================================

def multiplication(cutoff: int, alpha: float, circuit: c2qa.CVCircuit,
                   qumode_register: c2qa.QumodeRegister, qumode_idx: int) -> c2qa.CVCircuit:
    """
    Multiply the CV mode value by a scalar alpha using squeezing.

    Implements multiplication by decomposing into a sequence of squeezing operations.

    Args:
        cutoff: Fock space cutoff dimension.
        alpha: Multiplication factor.
        circuit: The CVCircuit to append gates to.
        qumode_register: Qumode register.
        qumode_idx: Index of the qumode to apply multiplication.

    Returns:
        The modified circuit.
    """
    if alpha == 1:
        return circuit

    log_alpha = np.log(alpha)
    l = int(np.ceil(abs(log_alpha)))

    if l == 0:
        return circuit

    small_r = -log_alpha / l

    for _ in range(l):
        circuit.cv_sq(small_r, qumode_register[qumode_idx])

    return circuit


def extractLSB(cutoff: int, circuit: c2qa.CVCircuit,
               qumode_register: c2qa.QumodeRegister,
               qubit_register, qumode_idx: int) -> c2qa.CVCircuit:
    """
    Extract the least significant bit from a CV mode into a qubit.

    Uses H -> controlled-phase -> H sequence.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: The CVCircuit to append gates to.
        qumode_register: Qumode register.
        qubit_register: Qubit register.
        qumode_idx: Index of the qumode.

    Returns:
        The modified circuit.
    """
    qumode_gate = tensor(qproj00(), qeye(cutoff)) + tensor(qproj11(), (1j * np.pi * position(cutoff)).expm())
    circuit.h(qubit_register[0])

    # gate = UnitaryGate(qumode_gate.full(), label='LSB')
    # circuit.append(gate, qumode_register[qumode_idx] + qubit_register[:])
    circuit.cv_gate_from_matrix(
        qumode_gate.full(),
        qumodes=[qumode_register[qumode_idx]],
        qubits=qubit_register[:],
        label='LSB',
    )

    circuit.h(qubit_register[0])

    return circuit


def extractLSB_dag(cutoff: int, circuit: c2qa.CVCircuit,
                   qumode_register: c2qa.QumodeRegister,
                   qubit_register, qumode_idx: int) -> c2qa.CVCircuit:
    """
    Hermitian conjugate of extractLSB operation.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: The CVCircuit to append gates to.
        qumode_register: Qumode register.
        qubit_register: Qubit register.
        qumode_idx: Index of the qumode.

    Returns:
        The modified circuit.
    """
    qumode_gate = tensor(qproj00(), qeye(cutoff)) + tensor(qproj11(), (-1j * np.pi * position(cutoff)).expm())

    circuit.h(qubit_register[0])

    # gate = UnitaryGate(qumode_gate.full(), label='LSB_dag')
    # circuit.append(gate, qumode_register[qumode_idx] + qubit_register[:])
    circuit.cv_gate_from_matrix(
        qumode_gate.full(),
        qumodes=[qumode_register[qumode_idx]],
        qubits=qubit_register[:],
        label='LSB_dag',
    )

    circuit.h(qubit_register[0])

    return circuit


def translation_R(cutoff: int, R: float, circuit: c2qa.CVCircuit,
                  qumode_register: c2qa.QumodeRegister, qumode_idx: int) -> c2qa.CVCircuit:
    """
    Translate a CV mode by displacement R.

    Args:
        cutoff: Fock space cutoff dimension (unused, kept for API consistency).
        R: Displacement amount.
        circuit: The CVCircuit to append gates to.
        qumode_register: Qumode register.
        qumode_idx: Index of the qumode.

    Returns:
        The modified circuit.
    """
    circuit.cv_d(R / np.sqrt(2), qumode_register[qumode_idx])
    return circuit


def control_multiplication(cutoff: int, alpha: float, circuit: c2qa.CVCircuit,
                           qumode_register: c2qa.QumodeRegister,
                           qubit_register, qumode_idx: int) -> c2qa.CVCircuit:
    """
    Qubit-controlled multiplication of a CV mode.

    Args:
        cutoff: Fock space cutoff dimension.
        alpha: Multiplication factor.
        circuit: The CVCircuit to append gates to.
        qumode_register: Qumode register.
        qubit_register: Qubit register.
        qumode_idx: Index of the qumode.

    Returns:
        The modified circuit.
    """
    rotation_plus = rotation_control(cutoff, 1)
    # gate1 = UnitaryGate(rotation_plus.full(), label='cR1')

    rotation_minus = rotation_control(cutoff, -1)
    # gate2 = UnitaryGate(rotation_minus.full(), label='cR2')

    circuit = multiplication(cutoff, np.sqrt(alpha), circuit, qumode_register, qumode_idx)
    # circuit.append(gate2, qumode_register[qumode_idx] + qubit_register[:])
    circuit.cv_gate_from_matrix(
        rotation_minus.full(),
        qumodes=[qumode_register[qumode_idx]],
        qubits=qubit_register[:],
        label='cR2',
    )

    circuit = multiplication(cutoff, 1 / np.sqrt(alpha), circuit, qumode_register, qumode_idx)

    # circuit.append(gate1, qumode_register[qumode_idx] + qubit_register[:])
    circuit.cv_gate_from_matrix(
        rotation_plus.full(),
        qumodes=[qumode_register[qumode_idx]],
        qubits=qubit_register[:],
        label='cR1',
    )

    return circuit


# =============================================================================
# V-Gate Operations (Core of Shor's Algorithm)
# =============================================================================

def V_alpha(cutoff: int, circuit: c2qa.CVCircuit,
            qumode_register: c2qa.QumodeRegister,
            qubit_register, alpha: float) -> c2qa.CVCircuit:
    """
    Apply the V_alpha gate for modular multiplication.

    This is a core building block of Shor's algorithm that implements
    a step of modular exponentiation.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: The CVCircuit to append gates to.
        qumode_register: Qumode register (requires 3 qumodes).
        qubit_register: Qubit register.
        alpha: Multiplication parameter.

    Returns:
        The modified circuit.
    """
    # Multiply qumode[2] by 2
    circuit = multiplication(cutoff, 2, circuit, qumode_register, 2)

    # Extract LSB from qumode[0]
    circuit = extractLSB(cutoff, circuit, qumode_register, qubit_register, 0)

    # Controlled subtraction on qumode[0]
    control_subtraction = Q_control_minus1(cutoff)
    # gate = UnitaryGate(control_subtraction.full(), label='Q-1')
    # circuit.append(gate, qumode_register[0] + qubit_register[:])
    circuit.cv_gate_from_matrix(
        control_subtraction.full(),
        qumodes=[qumode_register[0]],
        qubits=qubit_register[:],
        label='Q-1',
    )

    # Controlled multiplication by alpha on qumode[1]
    circuit = control_multiplication(cutoff, alpha, circuit, qumode_register, qubit_register, 1)

    # Controlled addition on qumode[2]
    control_addition = Q_control_plus1(cutoff)
    # gate = UnitaryGate(control_addition.full(), label='Q+1')
    # circuit.append(gate, qumode_register[2] + qubit_register[:])
    circuit.cv_gate_from_matrix(
        control_addition.full(),
        qumodes=[qumode_register[2]],
        qubits=qubit_register[:],
        label='Q+1',
    )

    # Extract LSB from qumode[2]
    circuit = extractLSB(cutoff, circuit, qumode_register, qubit_register, 2)

    # Divide qumode[0] by 2
    circuit = multiplication(cutoff, 0.5, circuit, qumode_register, 0)

    return circuit


def V_alpha_dag(cutoff: int, circuit: c2qa.CVCircuit,
                qumode_register: c2qa.QumodeRegister,
                qubit_register, alpha: float) -> c2qa.CVCircuit:
    """
    Hermitian conjugate of V_alpha gate.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: The CVCircuit to append gates to.
        qumode_register: Qumode register (requires 3 qumodes).
        qubit_register: Qubit register.
        alpha: Multiplication parameter.

    Returns:
        The modified circuit.
    """
    circuit = multiplication(cutoff, 2, circuit, qumode_register, 0)

    circuit = extractLSB_dag(cutoff, circuit, qumode_register, qubit_register, 2)

    control_subtraction = Q_control_minus1(cutoff)
    # gate = UnitaryGate(control_subtraction.full(), label='Q-1')
    # circuit.append(gate, qumode_register[2] + qubit_register[:])
    circuit.cv_gate_from_matrix(
        control_subtraction.full(),
        qumodes=[qumode_register[2]],
        qubits=qubit_register[:],
        label='Q-1',
    )

    circuit = control_multiplication(cutoff, 1 / alpha, circuit, qumode_register, qubit_register, 1)

    control_addition = Q_control_plus1(cutoff)
    # gate = UnitaryGate(control_addition.full(), label='Q+1')
    # circuit.append(gate, qumode_register[0] + qubit_register[:])
    circuit.cv_gate_from_matrix(
        control_addition.full(),
        qumodes=[qumode_register[0]],
        qubits=qubit_register[:],
        label='Q+1',
    )

    circuit = extractLSB_dag(cutoff, circuit, qumode_register, qubit_register, 0)

    circuit = multiplication(cutoff, 0.5, circuit, qumode_register, 2)

    return circuit


# =============================================================================
# Modular Exponentiation
# =============================================================================

def V_aNm(cutoff: int, circuit: c2qa.CVCircuit,
          qumode_register: c2qa.QumodeRegister,
          qubit_register, a: int, N: int, m: int) -> c2qa.CVCircuit:
    """
    Apply the V_{a,N,m} sequence for modular exponentiation.

    Computes a^(2^i) mod N for i = 0, ..., m-1.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: The CVCircuit to append gates to.
        qumode_register: Qumode register.
        qubit_register: Qubit register.
        a: Base for exponentiation.
        N: Modulus.
        m: Number of bits in the exponent.

    Returns:
        The modified circuit.
    """
    for i in range(m):
        alpha = pow(a, 2**i) % N
        circuit = V_alpha(cutoff, circuit, qumode_register, qubit_register, alpha)
        circuit.barrier()

    for i in range(m):
        circuit = V_alpha_dag(cutoff, circuit, qumode_register, qubit_register, 1)
        circuit.barrier()

    return circuit


def V_aNm_dagger(cutoff: int, circuit: c2qa.CVCircuit,
                 qumode_register: c2qa.QumodeRegister,
                 qubit_register, a: int, N: int, m: int) -> c2qa.CVCircuit:
    """
    Hermitian conjugate of V_{a,N,m} sequence.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: The CVCircuit to append gates to.
        qumode_register: Qumode register.
        qubit_register: Qubit register.
        a: Base for exponentiation.
        N: Modulus.
        m: Number of bits in the exponent.

    Returns:
        The modified circuit.
    """
    for _ in range(m):
        V_alpha(cutoff, circuit, qumode_register, qubit_register, 1)
        circuit.barrier()

    for i in reversed(range(m)):
        alpha = pow(a, 2**i, N)
        V_alpha_dag(cutoff, circuit, qumode_register, qubit_register, alpha)
        circuit.barrier()

    return circuit


def U_aNm(cutoff: int, circuit: c2qa.CVCircuit,
          qumode_register: c2qa.QumodeRegister,
          qubit_register, a: int, N: int, m: int) -> c2qa.CVCircuit:
    """
    Complete modular exponentiation unitary U_{a,N,m}.

    Implements the quantum operation: |x> -> |a*x mod N>

    This is the core operation in Shor's algorithm for period finding.

    Args:
        cutoff: Fock space cutoff dimension.
        circuit: The CVCircuit to append gates to.
        qumode_register: Qumode register (requires 3 qumodes).
        qubit_register: Qubit register.
        a: Base for exponentiation.
        N: Modulus (number to factor).
        m: Number of bits in the exponent.

    Returns:
        The modified circuit implementing modular exponentiation.
    """
    circuit = V_aNm_dagger(cutoff, circuit, qumode_register, qubit_register, a, N, m)
    circuit.barrier()

    Q_addition = Q_displacement_plus1(cutoff)
    # gate = UnitaryGate(Q_addition.full(), label='Q+1')
    # circuit.append(gate, qumode_register[1])
    circuit.cv_gate_from_matrix(
        Q_addition.full(),
        qumodes=[qumode_register[1]],
        qubits=[],
        label='Q+1',
    )

    circuit.barrier()

    circuit = V_aNm(cutoff, circuit, qumode_register, qubit_register, a, N, m)
    circuit.barrier()

    return circuit


# =============================================================================
# State Analysis and Visualization
# =============================================================================

def position_plotting(state, cutoff: int, ax_min: float = -6,
                      ax_max: float = 6, steps: int = 500) -> tuple:
    """
    Compute the position probability distribution from the Wigner function.

    Args:
        state: Quantum state (density matrix or state vector).
        cutoff: Fock space cutoff dimension (unused, kept for API consistency).
        ax_min: Minimum axis value.
        ax_max: Maximum axis value.
        steps: Number of discretization points.

    Returns:
        Tuple of (position_distribution, position_axis).
    """
    w = c2qa.wigner.wigner(state, axes_max=ax_max, axes_min=ax_min, axes_steps=steps)
    x_dist, _ = margins(w.T)

    x_dist *= (ax_max - ax_min) / steps
    xaxis = np.linspace(ax_min, ax_max, steps)

    return x_dist, xaxis


def momentum_plotting(state, cutoff: int, ax_min: float = -6,
                      ax_max: float = 6, steps: int = 500) -> tuple:
    """
    Compute the momentum probability distribution from the Wigner function.

    Args:
        state: Quantum state (density matrix or state vector).
        cutoff: Fock space cutoff dimension (unused, kept for API consistency).
        ax_min: Minimum axis value.
        ax_max: Maximum axis value.
        steps: Number of discretization points.

    Returns:
        Tuple of (momentum_distribution, momentum_axis).
    """
    w = c2qa.wigner.wigner(state, axes_max=ax_max, axes_min=ax_min, axes_steps=steps)
    _, p_dist = margins(w.T)

    p_dist *= (ax_max - ax_min) / steps
    paxis = np.linspace(ax_min, ax_max, steps)

    return p_dist, paxis


# =============================================================================
# Partial Trace Utilities
# =============================================================================

def trace_out_qumode_index(circuit: c2qa.CVCircuit, state,
                           qumode_register: c2qa.QumodeRegister,
                           qubit_register, qumode_index: str = '0'):
    """
    Trace out all subsystems except the specified qumode.

    Args:
        circuit: The CVCircuit (used for register information).
        state: The quantum state to trace.
        qumode_register: Qumode register.
        qubit_register: Qubit register.
        qumode_index: Index of qumode to keep ('0', '1', or '2').

    Returns:
        Reduced density matrix for the specified qumode.
    """
    if qumode_index == '0':
        trace = c2qa.util.cv_partial_trace(circuit, state, qubit_register[0])
        trace = c2qa.util.cv_partial_trace(circuit, trace, qumode_register[1] + qumode_register[2])
    elif qumode_index == '1':
        trace = c2qa.util.cv_partial_trace(circuit, state, qubit_register[0])
        trace = c2qa.util.cv_partial_trace(circuit, trace, qumode_register[0] + qumode_register[2])
    else:
        trace = c2qa.util.cv_partial_trace(circuit, state, qubit_register[0])
        trace = c2qa.util.cv_partial_trace(circuit, trace, qumode_register[0] + qumode_register[1])

    return trace


def get_reduced_qumode_density_matrix(stateop, qumode_index: int,
                                      num_qumodes: int, cutoff: int):
    """
    Get the reduced density matrix for a single qumode using Qiskit's partial_trace.

    Args:
        stateop: The full quantum state operator.
        qumode_index: Index of the qumode to keep (0, 1, or 2).
        num_qumodes: Total number of qumodes in the system.
        cutoff: Fock space cutoff dimension.

    Returns:
        Reduced density matrix for the specified qumode.
    """
    num_qubits_per_qumode = int(np.ceil(np.log2(cutoff)))
    total_qubits = num_qumodes * num_qubits_per_qumode + 1

    keep_indices = list(range(
        qumode_index * num_qubits_per_qumode,
        (qumode_index + 1) * num_qubits_per_qumode
    ))

    all_indices = list(range(total_qubits))
    trace_indices = [i for i in all_indices if i not in keep_indices]

    return partial_trace(stateop, trace_indices)
