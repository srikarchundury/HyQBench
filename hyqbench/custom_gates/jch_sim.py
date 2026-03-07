"""
Jaynes-Cummings-Hubbard (JCH) simulation circuits.

This module implements quantum circuits for simulating the Jaynes-Cummings-Hubbard
model, which describes coupled cavity-qubit systems with photon hopping.
"""

import numpy as np
# import c2qa
import bosonic_qiskit as c2qa
from qiskit import QuantumRegister
from qiskit.converters import circuit_to_gate
from qutip import identity, sigmax, sigmay, tensor, destroy, create, Qobj


def sigmax_(beta: float, j: int, n: int, cutoff: int) -> Qobj:
    """
    Controlled Pauli-X interaction with CV position coupling.

    Implements exp(i * beta/2 * sigma_x_j tensor (a + a^dag)).

    Args:
        beta: Coupling strength.
        j: Qubit index (1-indexed).
        n: Total number of qubits.
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP unitary operator.
    """
    qubit_pauli = tensor(
        [identity(2)] * (j - 1) + [sigmax()] + [identity(2)] * (n - j)
    )
    qumode_matrix = tensor(
        [identity(cutoff)] * (j - 1) +
        [1j * (beta / 2) * (destroy(cutoff) + create(cutoff))] +
        [identity(cutoff)] * (n - j)
    )
    return tensor(qubit_pauli, qumode_matrix).expm()


def sigmay_(beta: float, j: int, n: int, cutoff: int) -> Qobj:
    """
    Controlled Pauli-Y interaction with CV momentum coupling.

    Implements exp(-beta/2 * sigma_y_j tensor (a^dag - a)).

    Args:
        beta: Coupling strength.
        j: Qubit index (1-indexed).
        n: Total number of qubits.
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP unitary operator.
    """
    qubit_pauli = tensor(
        [identity(2)] * (j - 1) + [sigmay()] + [identity(2)] * (n - j)
    )
    qumode_matrix = tensor(
        [identity(cutoff)] * (j - 1) +
        [1j * (-1j) * (beta / 2) * (create(cutoff) - destroy(cutoff))] +
        [identity(cutoff)] * (n - j)
    )
    return tensor(qubit_pauli, qumode_matrix).expm()


def coupling_term(beta: float, n: int, j: int, cutoff: int) -> Qobj:
    """
    Qubit-qumode coupling term combining sigma_x and sigma_y interactions.

    Args:
        beta: Coupling strength.
        n: Total number of qubits.
        j: Site index (0-indexed, converted to 1-indexed internally).
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP unitary operator for the coupling term.
    """
    return sigmax_(beta, j + 1, n, cutoff) * sigmay_(beta, j + 1, n, cutoff)


def createCircuit(Nsites: int, Nqubits: int, cutoff: int,
                  J: float, omega_r: float, omega_q: float,
                  g: float, tau: float,
                  display_circuit: bool = False):
    """
    Create a single timestep JCH simulation circuit as a gate.

    The circuit implements one Trotter step of the JCH Hamiltonian:
    H = J * sum(a_i^dag a_{i+1} + h.c.) + omega_r * sum(n_i)
        + omega_q * sum(sigma_z_i) + g * sum(sigma_+ a + sigma_- a^dag)

    Args:
        Nsites: Number of CV modes (cavity sites).
        Nqubits: Number of qubits (atoms).
        cutoff: Fock space cutoff dimension.
        J: Photon hopping strength.
        omega_r: Resonator frequency.
        omega_q: Qubit frequency.
        g: Qubit-cavity coupling strength.
        tau: Time step duration.
        display_circuit: If True, draw the circuit.

    Returns:
        Qiskit Gate representing one time step of JCH evolution.
    """
    # Initialize registers
    qmr = c2qa.QumodeRegister(
        num_qumodes=Nsites,
        num_qubits_per_qumode=int(np.ceil(np.log2(cutoff)))
    )
    qbr = QuantumRegister(Nqubits)
    circuit = c2qa.CVCircuit(qmr, qbr)

    # Hopping interaction between adjacent qumodes (staggered pattern)
    theta_hop = -J * tau
    for site in range(0, len(qmr) - 1, 2):  # Even sites
        circuit.cv_bs(theta_hop, qmr[site], qmr[site + 1])
    for site in range(1, len(qmr) - 1, 2):  # Odd sites
        circuit.cv_bs(theta_hop, qmr[site], qmr[site + 1])

    # Local resonator evolution
    theta_resonator = omega_r * tau
    for site in range(len(qmr)):
        circuit.cv_r(theta_resonator, qmr[site])

    # Local qubit evolution
    theta_qubit = omega_q * tau
    for qubit in range(len(qbr)):
        circuit.rz(theta_qubit, qbr[qubit])

    # Qubit-qumode coupling (Jaynes-Cummings interaction)
    theta_coupling = g * tau
    for site in range(len(qbr)):
        circuit.cv_jc(theta_coupling, 0, qmr[site], qbr[site])

    if display_circuit:
        circuit.draw('mpl')

    return circuit_to_gate(circuit, label='U')


def circuit_display(Nsites: int, Nqubits: int, cutoff: int,
                    J: float, omega_r: float, omega_q: float,
                    g: float, tau: float, timesteps: int,
                    display_circuit: bool = False) -> c2qa.CVCircuit:
    """
    Create a multi-timestep JCH simulation circuit.

    Unrolls multiple time steps into a single circuit for visualization
    or analysis purposes.

    Args:
        Nsites: Number of CV modes (cavity sites).
        Nqubits: Number of qubits (atoms).
        cutoff: Fock space cutoff dimension.
        J: Photon hopping strength.
        omega_r: Resonator frequency.
        omega_q: Qubit frequency.
        g: Qubit-cavity coupling strength.
        tau: Time step duration.
        timesteps: Number of time steps to simulate.
        display_circuit: If True, draw the circuit.

    Returns:
        CVCircuit with all timesteps unrolled.
    """
    qmr = c2qa.QumodeRegister(
        num_qumodes=Nsites,
        num_qubits_per_qumode=int(np.ceil(np.log2(cutoff))),
        name='qumode'
    )
    qbr = QuantumRegister(Nqubits)
    circuit = c2qa.CVCircuit(qmr, qbr)

    for _ in range(timesteps):
        # Hopping interaction (staggered)
        theta_hop = -J * tau
        for site in range(0, len(qmr) - 1, 2):
            circuit.cv_bs(theta_hop, qmr[site], qmr[site + 1])
        for site in range(1, len(qmr) - 1, 2):
            circuit.cv_bs(theta_hop, qmr[site], qmr[site + 1])

        # Local resonator evolution
        theta_resonator = omega_r * tau
        for site in range(len(qmr)):
            circuit.cv_r(theta_resonator, qmr[site])

        # Local qubit evolution
        theta_qubit = omega_q * tau
        for qubit in range(len(qbr)):
            circuit.rz(theta_qubit, qbr[qubit])

        # Qubit-qumode coupling
        theta_coupling = g * tau
        for site in range(len(qbr)):
            circuit.cv_jc(theta_coupling, 0, qmr[site], qbr[site])

    return circuit
