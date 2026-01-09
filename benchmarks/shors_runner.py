"""
Shor's algorithm runner and success probability estimator.

This module provides tools for running Shor's factoring algorithm
using CV-DV hybrid quantum circuits and estimating success probabilities
through repeated trials with momentum-space measurements.
"""

import os
import random
import numpy as np
from datetime import datetime
from fractions import Fraction
from math import gcd

from qiskit import QuantumRegister, ClassicalRegister
from qutip import destroy, squeeze, displace, basis

import c2qa
from custom_gates import shors
from .benchmarks_circuit import shors_circuit


# Configuration
RESULTS_DIR = "results_logs"
os.makedirs(RESULTS_DIR, exist_ok=True)

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = os.path.join(RESULTS_DIR, f"log_{_timestamp}.txt")
_factors_file = os.path.join(RESULTS_DIR, f"factors_{_timestamp}.txt")


def _write_log(msg: str) -> None:
    """Append message to log file."""
    with open(_log_file, "a") as f:
        f.write(msg + "\n")


def _write_factors(msg: str) -> None:
    """Append message to factors file."""
    with open(_factors_file, "a") as f:
        f.write(msg + "\n")


def try_find_factors(N: int, r: int, a: int) -> tuple:
    """
    Attempt to find factors of N from period r and base a.

    Uses the standard classical post-processing step of Shor's algorithm.

    Args:
        N: Number to factor.
        r: Estimated period of a^x mod N.
        a: Base used in modular exponentiation.

    Returns:
        Tuple (p, q) if factors found, None otherwise.
    """
    if r % 2 != 0:
        return None
    candidate = pow(a, r // 2, N)
    if candidate == N - 1 or candidate == 1:
        return None
    p = np.gcd(candidate - 1, N)
    q = np.gcd(candidate + 1, N)
    if p * q == N and p != 1 and q != 1:
        return (p, q)
    return None


def find_valid_a_values(N: int) -> list:
    """
    Find all valid values of 'a' for Shor's algorithm.

    A valid 'a' has gcd(a, N) = 1 and an even period r where
    a^(r/2) != -1 mod N.

    Args:
        N: Number to factor.

    Returns:
        List of (a, r) tuples for valid base values.
    """
    valid_a = []
    for a in range(2, N):
        if gcd(a, N) != 1:
            continue
        r = 1
        while pow(a, r, N) != 1 and r < N:
            r += 1
        if r % 2 == 0 and pow(a, r // 2, N) != N - 1:
            valid_a.append((a, r))
    return valid_a


def sample_p_and_estimate_period(p_dist: np.ndarray, paxis: np.ndarray,
                                  max_denominator: int = 100) -> tuple:
    """
    Sample from momentum distribution and estimate period using continued fractions.

    Args:
        p_dist: Momentum probability distribution.
        paxis: Momentum axis values.
        max_denominator: Maximum denominator for continued fraction approximation.

    Returns:
        Tuple of (estimated_period, (numerator, denominator), sampled_momentum).
    """
    p_dist = np.nan_to_num(p_dist, nan=0.0, posinf=0.0, neginf=0.0)
    p_dist = np.clip(p_dist, 0, None)

    total = np.sum(p_dist)
    if total == 0:
        raise ValueError("Probability distribution sums to zero after sanitization.")

    prob_dist = p_dist / total
    p_sample = np.random.choice(paxis, p=prob_dist)

    # Estimate period using continued fractions
    s_over_r = p_sample / (2 * np.pi)
    frac = Fraction(s_over_r).limit_denominator(max_denominator)
    j, r = frac.numerator, frac.denominator

    estimated_period = r if gcd(j, r) == 1 else None

    return estimated_period, (j, r), p_sample


def generate_gkp_codeword(cutoff: int, delta: float = 0.3,
                          kappa: float = 1.0, logical: int = 0,
                          num_peaks: int = 7):
    """
    Generate approximate GKP codeword state.

    Creates a superposition of displaced squeezed states to approximate
    the ideal GKP logical state.

    Args:
        cutoff: Fock space cutoff dimension.
        delta: Squeezing parameter.
        kappa: Envelope decay rate.
        logical: Logical qubit value (0 or 1).
        num_peaks: Number of displacement peaks.

    Returns:
        QuTiP state vector for the GKP codeword.
    """
    sq = squeeze(cutoff, -np.log(delta))

    state = 0
    spacing = np.sqrt(np.pi)
    for k in range(-num_peaks // 2, num_peaks // 2 + 1):
        shift = (2 * k + logical) * spacing
        envelope = np.exp(-0.5 * (k * kappa) ** 2)
        disp = displace(cutoff, shift)
        peak = disp * sq * basis(cutoff, 0)
        state += envelope * peak

    state = state.unit()
    return state


def estimate_success_probability(N: int, m: int, R: int, delta: float,
                                  cutoff: int, trials: int = 30,
                                  shots: int = 1024) -> tuple:
    """
    Estimate success probability of Shor's algorithm for factoring N.

    Runs multiple trials with different valid 'a' values and samples
    from the resulting momentum distribution to estimate the period.

    Args:
        N: Number to factor.
        m: Modular exponentiation parameter.
        R: Register size parameter.
        delta: GKP squeezing parameter.
        cutoff: Fock space cutoff dimension.
        trials: Number of independent trials.
        shots: Number of measurement samples per trial.

    Returns:
        Tuple of (success_rate, unique_factors, total_successes, total_shots).
    """
    valid_a_r_pairs = find_valid_a_values(N)
    if not valid_a_r_pairs:
        _write_log(f"[N={N}] No valid a values found.")
        return 0.0, [], 0, 0

    total_successes = 0
    total_shots = 0
    all_factors = set()

    for trial in range(trials):
        a, true_r = random.choice(valid_a_r_pairs)
        _write_log(f"[Trial {trial + 1}] a = {a}, expected r = {true_r}")

        circuit = shors_circuit(N, m, R, a, delta, cutoff)

        # Run simulation to get momentum distribution
        stateop, _, _ = c2qa.util.simulate(circuit, shots=1)
        rho_qumode_0 = shors.get_reduced_qumode_density_matrix(
            stateop, qumode_index=0, num_qumodes=3, cutoff=cutoff
        )
        x_dist, xaxis = shors.momentum_plotting(
            rho_qumode_0, cutoff, ax_min=-30, ax_max=30, steps=200
        )

        # Sample from distribution
        for _ in range(shots):
            estimated_r, (j, r), p_sample = sample_p_and_estimate_period(
                x_dist.flatten(), xaxis
            )

            if estimated_r is not None:
                factors = try_find_factors(N, estimated_r, a)
                if factors:
                    total_successes += 1
                    all_factors.update(factors)
                    _write_log(f"Shot success: r={estimated_r}, factors={factors}")
                    _write_factors(f"N={N}, a={a}, r={estimated_r}, factors={factors}")
                else:
                    _write_log(f"Shot fail: r={estimated_r} gave no valid factors.")
            else:
                _write_log("Shot fail: Could not extract valid r.")
            total_shots += 1

    success_rate = total_successes / total_shots if total_shots > 0 else 0.0
    _write_log(f"[N={N}, cutoff={cutoff}] Total success probability: {success_rate:.4f}")

    return success_rate, sorted(all_factors), total_successes, total_shots
