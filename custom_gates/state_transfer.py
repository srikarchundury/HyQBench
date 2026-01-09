"""
State transfer operations between discrete variable (DV) and continuous variable (CV) systems.

This module implements non-abelian state transfer protocols that map qubit states
to continuous variable Gaussian states and vice versa.
"""

import numpy as np
from qutip import identity, sigmax, sigmay, tensor, destroy, create, Qobj


def Vj(lmbda: float, j: int, n: int, cutoff: int) -> Qobj:
    """
    V_j unitary for DV-to-CV state transfer.

    Implements conditional position displacement controlled by qubit j.

    Args:
        lmbda: Coupling strength parameter.
        j: Qubit index (1-indexed, from 1 to n).
        n: Total number of qubits.
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP unitary operator V_j.
    """
    qubit_pauli = tensor(
        [identity(2)] * (j - 1) + [sigmay()] + [identity(2)] * (n - j)
    )
    position_op = (destroy(cutoff) + create(cutoff)) / np.sqrt(2.0)
    displacement = (1j * np.pi) / (2**(j + 1) * lmbda) * position_op

    return tensor(qubit_pauli, displacement).expm()


def Wj(lmbda: float, j: int, n: int, cutoff: int) -> Qobj:
    """
    W_j unitary for DV-to-CV state transfer.

    Implements conditional momentum displacement controlled by qubit j.

    Args:
        lmbda: Coupling strength parameter.
        j: Qubit index (1-indexed, from 1 to n).
        n: Total number of qubits.
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP unitary operator W_j.
    """
    qubit_pauli = tensor(
        [identity(2)] * (j - 1) + [sigmax()] + [identity(2)] * (n - j)
    )

    # Sign depends on whether j is the last qubit
    if j == n:
        disp_amount = -lmbda * 2**(j - 1)
    else:
        disp_amount = lmbda * 2**(j - 1)

    momentum_op = (destroy(cutoff) - create(cutoff)) / np.sqrt(2.0)

    return tensor(qubit_pauli, disp_amount * momentum_op).expm()


def dv2cv_st_non_abelian(lmbda: float, n: int, cutoff: int) -> Qobj:
    """
    Complete DV-to-CV state transfer unitary (non-abelian protocol).

    Maps an n-qubit discrete variable state to a continuous variable
    Gaussian state using a sequence of V and W gates.

    The protocol applies V_j^dag * W_j^dag for j = n, n-1, ..., 1
    to transfer the qubit information into the CV mode.

    Args:
        lmbda: Coupling strength parameter (controls encoding resolution).
        n: Number of qubits to transfer.
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP unitary operator for the complete state transfer.
    """
    U = tensor([identity(2)] * n + [identity(cutoff)])

    for j in range(n, 0, -1):
        U = Vj(lmbda, j, n, cutoff).dag() * Wj(lmbda, j, n, cutoff).dag() * U

    return U
