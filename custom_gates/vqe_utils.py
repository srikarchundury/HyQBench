"""
VQE utilities for ECD (Echoed Conditional Displacement) ansatz.

This module provides helper functions for variational quantum eigensolvers
using the ECD ansatz on hybrid qubit-qumode systems.
"""

import numpy as np
import qutip as qt

from .hamiltonian_utils import qproj01, qproj10


# =============================================================================
# Parameter Packing/Unpacking
# =============================================================================

def pack_variables(beta_mag: np.ndarray, beta_arg: np.ndarray,
                   theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Pack ECD ansatz parameters into a single 1D array.

    Args:
        beta_mag: ECD displacement magnitudes, shape (ndepth, 2).
        beta_arg: ECD displacement phases, shape (ndepth, 2).
        theta: Qubit rotation angles theta, shape (ndepth, 2).
        phi: Qubit rotation angles phi, shape (ndepth, 2).

    Returns:
        Flattened 1D array of all parameters.
    """
    return np.concatenate([
        beta_mag.ravel(),
        beta_arg.ravel(),
        theta.ravel(),
        phi.ravel()
    ])


def unpack_variables(Xvec: np.ndarray, ndepth: int) -> tuple:
    """
    Unpack a 1D parameter array into ECD ansatz parameter matrices.

    Args:
        Xvec: Flattened parameter array.
        ndepth: Circuit depth (number of ECD layers).

    Returns:
        Tuple of (beta_mag, beta_arg, theta, phi), each with shape (ndepth, 2).
    """
    size = ndepth * 2

    beta_mag = Xvec[:size].reshape((ndepth, 2))
    beta_arg = Xvec[size:2*size].reshape((ndepth, 2))
    theta = Xvec[2*size:3*size].reshape((ndepth, 2))
    phi = Xvec[3*size:].reshape((ndepth, 2))

    return beta_mag, beta_arg, theta, phi


def get_cvec_np(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Convert polar coordinates to complex numbers.

    Args:
        r: Magnitude array.
        theta: Phase array.

    Returns:
        Complex array r * exp(i * theta).
    """
    r = np.array(r)
    theta = np.array(theta)
    return r * np.exp(1j * theta)


# =============================================================================
# Qubit Operations
# =============================================================================

def qubit_rot(theta: float, phi: float) -> qt.Qobj:
    """
    Construct a qubit rotation operator.

    R(theta, phi) = exp[-i (theta/2) (X cos(phi) + Y sin(phi))]

    Args:
        theta: Rotation angle.
        phi: Rotation axis angle in the X-Y plane.

    Returns:
        QuTiP Qobj representing the rotation unitary.
    """
    gen = qt.sigmax() * np.cos(phi) + qt.sigmay() * np.sin(phi)
    H = -1j * (theta / 2) * gen
    return H.expm()


# =============================================================================
# ECD Operations
# =============================================================================

def ecd_op(beta: complex, theta: float, phi: float,
           cind: int, nfocks: list) -> qt.Qobj:
    """
    Construct an ECD (Echoed Conditional Displacement) operator.

    The ECD gate applies conditional displacements on a qumode based on
    the qubit state.

    Args:
        beta: Complex displacement parameter.
        theta: Rotation angle (unused in current implementation).
        phi: Rotation phase (unused in current implementation).
        cind: Qumode index (0 or 1) to apply the displacement.
        nfocks: List of Fock cutoffs [n1, n2] for the two qumodes.

    Returns:
        QuTiP Qobj representing the ECD unitary.

    Raises:
        ValueError: If cind is not 0 or 1.
    """
    if cind not in (0, 1):
        raise ValueError("cind must be 0 or 1")

    if cind == 0:
        E = qt.tensor(qproj10(), qt.displace(nfocks[0], beta / 2))
        E += qt.tensor(qproj01(), qt.displace(nfocks[0], -beta / 2))
        E = qt.tensor(E, qt.qeye(nfocks[1]))
    else:
        E = qt.tensor(qproj10(), qt.qeye(nfocks[0]), qt.displace(nfocks[1], beta / 2))
        E += qt.tensor(qproj01(), qt.qeye(nfocks[0]), qt.displace(nfocks[1], -beta / 2))

    return E


# =============================================================================
# Fock Basis Utilities
# =============================================================================

def generate_triples(nfocks: list) -> np.ndarray:
    """
    Generate all (qubit, n, m) basis state indices for a two-qumode system.

    Args:
        nfocks: List of Fock cutoffs [n1, n2].

    Returns:
        Array of shape (2 * nfocks[0] * nfocks[1], 3) containing all
        (qubit_index, fock_n, fock_m) triples.
    """
    q_range = np.arange(2)
    n_range = np.arange(nfocks[0])
    m_range = np.arange(nfocks[1])

    q_grid, n_grid, m_grid = np.meshgrid(q_range, n_range, m_range, indexing='ij')
    triples = np.stack((q_grid.ravel(), n_grid.ravel(), m_grid.ravel()), axis=-1)

    return triples
