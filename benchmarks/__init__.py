"""
Benchmark circuits and metrics for hybrid CV-DV quantum systems.

This package provides circuit builders, characterization tools, and
benchmarking utilities for continuous-variable discrete-variable (CV-DV)
hybrid quantum systems built on Bosonic Qiskit.

Modules:
    benchmarks_circuit: Circuit builders for state preparation, VQE, QAOA, etc.
    features: Quantum metrics and feature extraction
    circuit_characterization: Comprehensive circuit characterization
    shors_runner: Shor's algorithm runner and success estimation
"""

# Circuit builders
from .benchmarks_circuit import (
    # State preparation
    cat_state_circuit,
    gkp_state_circuit,
    # State transfer
    state_transfer_CVtoDV,
    state_transfer_DVtoCV,
    # VQE
    binary_knapsack_vqe,
    binary_knapsack_vqe_circuit,
    # QAOA
    cv_qaoa,
    cv_qaoa_circuit,
    # QFT
    qft_circuit,
    # JCH simulation
    JCH_simulation_circuit,
    JCH_simulation_circuit_unitary,
    JCH_simulation_circuit_display,
    # Shor's algorithm
    shors_circuit,
)

# Feature extraction and metrics
from .features import (
    collect_cvcircuit_metrics,
    plot_radar_metrics,
    get_reduced_qumode_density_matrix,
    get_reduced_qubit_density_matrix,
    wigner_negativity_all_modes,
    truncation_cost_all_modes,
    average_energy_all,
    evaluate_quantum_metrics,
)

# Circuit characterization
from .circuit_characterization import (
    plot_radar_group,
    average_over_timesteps,
    characterize_circuit,
)

# Shor's algorithm utilities
from .shors_runner import (
    try_find_factors,
    find_valid_a_values,
    sample_p_and_estimate_period,
    generate_gkp_codeword,
    estimate_success_probability,
)


__all__ = [
    # Circuit builders
    'cat_state_circuit',
    'gkp_state_circuit',
    'state_transfer_CVtoDV',
    'state_transfer_DVtoCV',
    'binary_knapsack_vqe',
    'binary_knapsack_vqe_circuit',
    'cv_qaoa',
    'cv_qaoa_circuit',
    'qft_circuit',
    'JCH_simulation_circuit',
    'JCH_simulation_circuit_unitary',
    'JCH_simulation_circuit_display',
    'shors_circuit',
    # Metrics
    'collect_cvcircuit_metrics',
    'plot_radar_metrics',
    'get_reduced_qumode_density_matrix',
    'get_reduced_qubit_density_matrix',
    'wigner_negativity_all_modes',
    'truncation_cost_all_modes',
    'average_energy_all',
    'evaluate_quantum_metrics',
    # Characterization
    'plot_radar_group',
    'average_over_timesteps',
    'characterize_circuit',
    # Shor's utilities
    'try_find_factors',
    'find_valid_a_values',
    'sample_p_and_estimate_period',
    'generate_gkp_codeword',
    'estimate_success_probability',
]
