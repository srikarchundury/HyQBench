"""
Bosonic Quantum Approximate Optimization Algorithm (QAOA).

This module implements QAOA using continuous variable (bosonic) modes
for optimization problems. The cost Hamiltonian targets position eigenstates
and uses a kinetic mixer based on momentum.
"""

import numpy as np
# import c2qa
import bosonic_qiskit as c2qa
from qiskit import ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qutip import position, momentum, qeye, expect, Qobj
from scipy.stats.contingency import margins


def cost(cutoff: int, a: float, n: int, eta: float) -> Qobj:
    """
    Cost Hamiltonian unitary for CV-QAOA.

    Implements U_c = exp(-i * eta * (x - a*I)^n) which encodes the
    optimization objective of finding x = a.

    Args:
        cutoff: Fock space cutoff dimension.
        a: Target position value.
        n: Power of the cost function (typically 2 for quadratic).
        eta: Evolution time / strength parameter.

    Returns:
        QuTiP unitary operator for the cost evolution.
    """
    x = position(cutoff)
    aI = a * qeye(cutoff)
    Hc = (x - aI) ** n
    return (-1j * eta * Hc).expm()


def kinetic_mixer(cutoff: int, gamma: float) -> Qobj:
    """
    Kinetic mixer Hamiltonian unitary for CV-QAOA.

    Implements U_m = exp(-i * gamma * p^2 / 2) which provides
    the mixing dynamics in momentum space.

    Args:
        cutoff: Fock space cutoff dimension.
        gamma: Evolution time / mixing strength.

    Returns:
        QuTiP unitary operator for the mixer evolution.
    """
    p = momentum(cutoff)
    Hm = 0.5 * (p ** 2)
    return (-1j * gamma * Hm).expm()


def cvQAOA(params: np.ndarray, cutoff: int, depth: int,
           s: float, n: int, a: float,
           costval: list, estval: list) -> float:
    """
    Execute CV-QAOA circuit and compute cost function.

    Args:
        params: Parameter array [gamma_0, ..., gamma_{p-1}, eta_0, ..., eta_{p-1}].
        cutoff: Fock space cutoff dimension.
        depth: Number of QAOA layers (p).
        s: Initial squeezing parameter.
        n: Power of the cost function.
        a: Target position value.
        costval: List to append cost values (modified in-place).
        estval: List to append position expectation values (modified in-place).

    Returns:
        Cost function value (E[x] - a)^n.
    """
    gamma_list = params[:depth]
    eta_list = params[depth:]

    # Initialize circuit
    qmr = c2qa.QumodeRegister(
        num_qumodes=1,
        num_qubits_per_qumode=int(np.ceil(np.log2(cutoff)))
    )
    cr = ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr, cr)

    # Initialize to squeezed vacuum
    circuit.cv_initialize(0, qmr[0])
    circuit.cv_sq(-s, qmr[0])

    # QAOA layers
    for i in range(depth):
        # Cost unitary
        cost_unitary = cost(cutoff, a, n, eta_list[i])
        # cost_gate = UnitaryGate(cost_unitary.full(), label=f'Uc_{i}')
        # circuit.append(cost_gate, qmr[0])
        circuit.cv_gate_from_matrix(
            cost_unitary.full(),
            qumodes=[qmr[0]],
            qubits=[],
            label=f"Uc_{i}",
        )

        # Mixer unitary
        mixer_unitary = kinetic_mixer(cutoff, gamma_list[i])
        # mixer_gate = UnitaryGate(mixer_unitary.full(), label=f'Um_{i}')
        # circuit.append(mixer_gate, qmr[0])
        circuit.cv_gate_from_matrix(
            mixer_unitary.full(),
            qumodes=[qmr[0]],
            qubits=[],
            label=f"Um_{i}",
        )

    # Simulate and compute expectation value
    state, _, _ = c2qa.util.simulate(circuit)
    x = position(cutoff)
    expval = expect(x, Qobj(state))

    # Cost function
    cost_value = (expval - a) ** n

    # Track progress
    estval.append(expval)
    costval.append(cost_value)

    return cost_value


def results_final(params: np.ndarray, cutoff: int, depth: int,
                  s: float, n: int, a: float) -> tuple:
    """
    Execute final CV-QAOA circuit and compute Wigner distribution.

    Args:
        params: Optimized parameter array.
        cutoff: Fock space cutoff dimension.
        depth: Number of QAOA layers.
        s: Initial squeezing parameter.
        n: Power of the cost function.
        a: Target position value.

    Returns:
        Tuple of (position_expectation, position_distribution, position_axis).
    """
    gamma_list = params[:depth]
    eta_list = params[depth:]

    # Initialize circuit
    qmr = c2qa.QumodeRegister(
        num_qumodes=1,
        num_qubits_per_qumode=int(np.ceil(np.log2(cutoff)))
    )
    cr = ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr, cr)

    # Initialize to squeezed vacuum
    circuit.cv_initialize(0, qmr[0])
    circuit.cv_sq(-s, qmr[0])

    # QAOA layers
    for i in range(depth):
        cost_unitary = cost(cutoff, a, n, eta_list[i])
        # cost_gate = UnitaryGate(cost_unitary.full(), label=f'Uc_{i}')
        # circuit.append(cost_gate, qmr[0])
        circuit.cv_gate_from_matrix(
            cost_unitary.full(),
            qumodes=[qmr[0]],
            qubits=[],
            label=f"Uc_{i}",
        )

        mixer_unitary = kinetic_mixer(cutoff, gamma_list[i])
        # mixer_gate = UnitaryGate(mixer_unitary.full(), label=f'Um_{i}')
        # circuit.append(mixer_gate, qmr[0])
        circuit.cv_gate_from_matrix(
            mixer_unitary.full(),
            qumodes=[qmr[0]],
            qubits=[],
            label=f"Um_{i}",
        )

    # Simulate
    state, _, _ = c2qa.util.simulate(circuit)

    # Position expectation
    x = position(cutoff)
    expval = expect(x, Qobj(state))

    # Wigner function marginal for position distribution
    ax_min, ax_max, steps = -6, 6, 200
    w = c2qa.wigner.wigner(state, axes_max=ax_max, axes_min=ax_min, axes_steps=steps)
    x_dist, _ = margins(w.T)

    # Normalize
    x_dist *= (ax_max - ax_min) / steps
    xaxis = np.linspace(ax_min, ax_max, steps)

    return expval, x_dist, xaxis


def build_qaoa_circuit(params: np.ndarray, cutoff: int, depth: int,
                       s: float, n: int, a: float) -> c2qa.CVCircuit:
    """
    Build a CV-QAOA circuit with given parameters.

    Args:
        params: Parameter array [gamma_0, ..., gamma_{p-1}, eta_0, ..., eta_{p-1}].
        cutoff: Fock space cutoff dimension.
        depth: Number of QAOA layers.
        s: Initial squeezing parameter.
        n: Power of the cost function.
        a: Target position value.

    Returns:
        The constructed CVCircuit.
    """
    gamma_list = params[:depth]
    eta_list = params[depth:]

    qmr = c2qa.QumodeRegister(
        num_qumodes=1,
        num_qubits_per_qumode=int(np.ceil(np.log2(cutoff)))
    )
    cr = ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr, cr)

    circuit.cv_initialize(0, qmr[0])
    circuit.cv_sq(-s, qmr[0])

    for i in range(depth):
        cost_unitary = cost(cutoff, a, n, eta_list[i])
        # cost_gate = UnitaryGate(cost_unitary.full(), label=f'Uc_{i}')
        # circuit.append(cost_gate, qmr[0])
        circuit.cv_gate_from_matrix(
            cost_unitary.full(),
            qumodes=[qmr[0]],
            qubits=[],
            label=f"Uc_{i}",
        )

        mixer_unitary = kinetic_mixer(cutoff, gamma_list[i])
        # mixer_gate = UnitaryGate(mixer_unitary.full(), label=f'Um_{i}')
        # circuit.append(mixer_gate, qmr[0])
        circuit.cv_gate_from_matrix(
            mixer_unitary.full(),
            qumodes=[qmr[0]],
            qubits=[],
            label=f"Um_{i}",
        )

    return circuit
