"""
State generation utilities for hybrid CV-DV quantum systems.

This module provides functions for generating quantum states and
conditional displacement operations on qubit-qumode systems.
"""

import numpy as np
from qutip import sigmax, sigmay, tensor, position, momentum, displace, Qobj

from .hamiltonian_utils import qproj00, qproj11


def CD_real(cutoff: int, alpha: float) -> Qobj:
    """
    Conditional displacement along the real (position) quadrature.

    Implements sigma_x tensor exp(-i * sqrt(2) * alpha * p).

    Args:
        cutoff: Fock space cutoff dimension.
        alpha: Displacement magnitude.

    Returns:
        QuTiP unitary operator for conditional real displacement.
    """
    p = momentum(cutoff)
    return tensor(sigmax(), -1j * np.sqrt(2) * alpha * p).expm()


def CD_imaginary(cutoff: int, alpha: float) -> Qobj:
    """
    Conditional displacement along the imaginary (momentum) quadrature.

    Implements sigma_y tensor exp(-i * pi * x / (4 * alpha)).

    Args:
        cutoff: Fock space cutoff dimension.
        alpha: Displacement parameter.

    Returns:
        QuTiP unitary operator for conditional imaginary displacement.
    """
    x = position(cutoff)
    return tensor(sigmay(), -1j * np.pi * x / (4 * alpha)).expm()


def Ux_operator(cutoff: int, theta: float, alpha: float, delta: float) -> Qobj:
    """
    Composite conditional displacement operator.

    Combines real and imaginary conditional displacements with
    specified parameters.

    Args:
        cutoff: Fock space cutoff dimension.
        theta: Rotation angle parameter.
        alpha: Displacement scale.
        delta: Squeezing parameter.

    Returns:
        QuTiP unitary operator (CD_real * CD_imag).
    """
    cd_real = tensor(sigmax(), 1j * (theta / alpha) * position(cutoff)).expm()
    cd_imag = tensor(sigmay(), 1j * (2 * theta / alpha) * (delta**2) * momentum(cutoff)).expm()
    return cd_real * cd_imag


def conditional_displacement(cutoff: int, alpha: complex) -> Qobj:
    """
    Qubit-controlled displacement operator.

    Applies +alpha displacement when qubit is |0> and -alpha when qubit is |1>.

    D_cd = |0><0| tensor D(alpha) + |1><1| tensor D(-alpha)

    Args:
        cutoff: Fock space cutoff dimension.
        alpha: Complex displacement parameter.

    Returns:
        QuTiP unitary operator for conditional displacement.
    """
    D_plus = displace(cutoff, alpha)
    D_minus = displace(cutoff, -alpha)
    return tensor(qproj00(), D_plus) + tensor(qproj11(), D_minus)
