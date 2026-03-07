"""
Custom quantum gates and algorithms for hybrid CV-DV systems.

This package provides implementations of quantum algorithms using
continuous-variable (bosonic) and discrete-variable (qubit) hybrid systems.

Modules:
    hamiltonian_utils: Hamiltonian construction and Pauli conversion utilities
    vqe_utils: VQE parameter handling and ECD gate utilities
    bosonic_vqe: Variational Quantum Eigensolver with ECD ansatz
    bosonic_qaoa: Quantum Approximate Optimization Algorithm for CV systems
    shors: Shor's factoring algorithm (circuit-based)
    jch_sim: Jaynes-Cummings-Hubbard simulation
    state_transfer: DV-CV state transfer protocols
    state_generation: Conditional displacement operations
    qft: Quantum Fourier Transform for CV modes
"""

# Hamiltonian utilities
from .hamiltonian_utils import (
    qproj00, qproj11, qproj01, qproj10,
    binary_knapsack_ham,
    binary_to_qubit_ham,
    binary_to_pauli_list,
    generate_tensor_product,
    qubit_op_to_ham,
)

# VQE utilities
from .vqe_utils import (
    pack_variables,
    unpack_variables,
    get_cvec_np,
    qubit_rot,
    ecd_op,
    generate_triples,
)

# Bosonic VQE
from .bosonic_vqe import (
    ecd_rot_op,
    ecd_rot_ansatz,
    energy_val,
    ecd_opt_vqe,
    num_prob_basis,
    num_prob_all,
    build_vqe_circuit,
)

# Bosonic QAOA
from .bosonic_qaoa import (
    cost,
    kinetic_mixer,
    cvQAOA,
    results_final,
    build_qaoa_circuit,
)

# Shor's Algorithm
from .shors import (
    hadamard,
    Q_displacement_plus1,
    Q_displacement_minus1,
    single_mode_squeeze,
    Q_control_plus1,
    Q_control_minus1,
    rotation_control,
    multiplication,
    extractLSB,
    extractLSB_dag,
    translation_R,
    control_multiplication,
    V_alpha,
    V_alpha_dag,
    V_aNm,
    V_aNm_dagger,
    U_aNm,
    position_plotting,
    momentum_plotting,
    trace_out_qumode_index,
    get_reduced_qumode_density_matrix,
)

# JCH Simulation
from .jch_sim import (
    sigmax_,
    sigmay_,
    coupling_term,
    createCircuit,
    circuit_display,
)

# State Transfer
from .state_transfer import (
    Vj,
    Wj,
    dv2cv_st_non_abelian,
)

# State Generation
from .state_generation import (
    CD_real,
    CD_imaginary,
    Ux_operator,
    conditional_displacement,
)

# QFT
from .qft import F


__all__ = [
    # Hamiltonian utilities
    'qproj00', 'qproj11', 'qproj01', 'qproj10',
    'binary_knapsack_ham', 'binary_to_qubit_ham', 'binary_to_pauli_list',
    'generate_tensor_product', 'qubit_op_to_ham',
    # VQE utilities
    'pack_variables', 'unpack_variables', 'get_cvec_np',
    'qubit_rot', 'ecd_op', 'generate_triples',
    # Bosonic VQE
    'ecd_rot_op', 'ecd_rot_ansatz', 'energy_val', 'ecd_opt_vqe',
    'num_prob_basis', 'num_prob_all', 'build_vqe_circuit',
    # Bosonic QAOA
    'cost', 'kinetic_mixer', 'cvQAOA', 'results_final', 'build_qaoa_circuit',
    # Shor's Algorithm
    'hadamard', 'Q_displacement_plus1', 'Q_displacement_minus1',
    'single_mode_squeeze', 'Q_control_plus1', 'Q_control_minus1',
    'rotation_control', 'multiplication', 'extractLSB', 'extractLSB_dag',
    'translation_R', 'control_multiplication', 'V_alpha', 'V_alpha_dag',
    'V_aNm', 'V_aNm_dagger', 'U_aNm', 'position_plotting', 'momentum_plotting',
    'trace_out_qumode_index', 'get_reduced_qumode_density_matrix',
    # JCH Simulation
    'sigmax_', 'sigmay_', 'coupling_term', 'createCircuit', 'circuit_display',
    # State Transfer
    'Vj', 'Wj', 'dv2cv_st_non_abelian',
    # State Generation
    'CD_real', 'CD_imaginary', 'Ux_operator', 'conditional_displacement',
    # QFT
    'F',
]
