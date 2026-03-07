"""
Quantum Fourier Transform operators for continuous variable systems.

This module provides the Fourier transform operator for CV (bosonic) modes.
"""

import numpy as np
from qutip import num, Qobj


def F(cutoff: int) -> Qobj:
    """
    Fourier transform operator for a continuous variable mode.

    Implements the CV Fourier transform as F = exp(i * pi/2 * n),
    where n is the photon number operator.

    This operation rotates phase space by 90 degrees, mapping
    position to momentum and vice versa.

    Args:
        cutoff: Fock space cutoff dimension.

    Returns:
        QuTiP unitary operator representing the Fourier transform.
    """
    return (1j * np.pi / 2 * num(cutoff)).expm()
