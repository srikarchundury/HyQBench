"""
Feature extraction and quantum metrics for CV-DV hybrid circuits.

This module provides tools for analyzing quantum circuit characteristics,
computing Wigner negativity, truncation costs, and other performance metrics
for continuous-variable discrete-variable (CV-DV) hybrid quantum systems.
"""

import numpy as np
from collections import Counter

# import c2qa
import bosonic_qiskit as c2qa
from qiskit.quantum_info import partial_trace
from qutip import num


def collect_cvcircuit_metrics(circuit: c2qa.CVCircuit, cutoff: int) -> dict:
    """
    Collect structural metrics from a CV-DV hybrid circuit.

    Analyzes the circuit to count qubits, qumodes, and categorize gates
    into qubit-only, qumode-only, and hybrid operations.

    Args:
        circuit: The CVCircuit to analyze.
        cutoff: Fock space cutoff dimension.

    Returns:
        Dictionary containing:
            - Qubits: Number of qubits
            - Qumodes: Number of qumodes
            - Qubit Gates: Count of qubit-only gates
            - Qumode Gates: Count of qumode-only gates
            - Hybrid Gates: Count of qubit-qumode hybrid gates
            - Circuit Depth: Circuit depth
    """
    # Map each qubit to its register name
    qubit_to_reg = {}
    for reg in circuit.qregs:
        for q in reg:
            qubit_to_reg[q] = reg.name

    # Separate qubit and qumode registers
    qubit_regs = []
    qumode_regs = []

    qumode_tags = ['qmode', 'cv', 'osc', 'qumode', 'qmr']
    for reg in circuit.qregs:
        name = reg.name.lower()
        if any(tag in name for tag in qumode_tags):
            qumode_regs.append(reg)
        else:
            qubit_regs.append(reg)

    num_qubits = sum(len(reg) for reg in qubit_regs)
    num_qumodes = sum(len(reg) for reg in qumode_regs)
    circuit_depth = circuit.depth()

    gate_counts = Counter()
    skip_instrs = {'barrier', 'measure', 'initialize', 'snapshot', 'delay'}

    for instr, qargs, cargs in circuit.data:
        if instr.name in skip_instrs:
            continue

        involved_regs = {qubit_to_reg.get(q, "").lower() for q in qargs}
        qubit_reg_names = [r.name for r in qubit_regs]
        qumode_reg_names = [r.name for r in qumode_regs]

        has_qubit = any(reg in qubit_reg_names for reg in involved_regs)
        has_qumode = any(reg in qumode_reg_names for reg in involved_regs)

        if has_qubit and has_qumode:
            gate_counts['hybrid_gates'] += 1
        elif has_qubit:
            gate_counts['qubit_gates'] += 1
        elif has_qumode:
            gate_counts['qumode_gates'] += 1
        else:
            gate_counts['unknown_gates'] += 1

    return {
        "Qubits": num_qubits,
        "Qumodes": num_qumodes / int(np.ceil(np.log2(cutoff))),
        "Qubit Gates": gate_counts["qubit_gates"],
        "Qumode Gates": gate_counts["qumode_gates"],
        "Hybrid Gates": gate_counts["hybrid_gates"],
        "Circuit Depth": circuit_depth
    }


def plot_radar_metrics(metrics_list: list, labels: list = None,
                       title: str = "CV-DV Radar Chart") -> None:
    """
    Plot radar chart comparing multiple circuit metrics.

    Args:
        metrics_list: List of metric dictionaries to compare.
        labels: Optional labels for each circuit.
        title: Chart title.
    """
    import matplotlib.pyplot as plt

    keys = ['Qubits', 'Qumodes', 'Qubit Gates', 'Qumode Gates',
            'Hybrid Gates', 'Total Gates']
    N = len(keys)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    max_vals = {key: max(metric[key] for metric in metrics_list) for key in keys}

    data = []
    for metric in metrics_list:
        normalized = [
            metric[key] / max_vals[key] if max_vals[key] != 0 else 0
            for key in keys
        ]
        normalized += normalized[:1]
        data.append(normalized)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=14)

    colors = plt.cm.tab10.colors

    for i, d in enumerate(data):
        label = labels[i] if labels else f"Circuit {i+1}"
        color = colors[i % len(colors)]
        ax.plot(angles, d, label=label, color=color)
        ax.fill(angles, d, alpha=0.25, color=color)

        original_metrics = metrics_list[i]
        for j in range(N):
            angle = angles[j]
            r = d[j]
            value = original_metrics[keys[j]]
            ax.text(angle, r + 0.05, f"{value}",
                    ha='center', va='center', fontsize=8, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


def get_reduced_qumode_density_matrix(stateop, qumode_index: int,
                                       num_qumodes: int, cutoff: int):
    """
    Compute reduced density matrix for a single qumode.

    Traces out all other qumodes and qubits to obtain the
    reduced state of the specified qumode.

    Args:
        stateop: Full quantum state (Qiskit DensityMatrix or Statevector).
        qumode_index: Index of the qumode to keep (0-indexed).
        num_qumodes: Total number of qumodes.
        cutoff: Fock space cutoff dimension.

    Returns:
        Reduced density matrix for the specified qumode.
    """
    num_qubits_per_qumode = int(np.ceil(np.log2(cutoff)))
    total_qubits = stateop.num_qubits

    qumode_indices = list(range(
        qumode_index * num_qubits_per_qumode,
        (qumode_index + 1) * num_qubits_per_qumode
    ))

    trace_indices = [i for i in range(total_qubits) if i not in qumode_indices]

    return partial_trace(stateop, trace_indices)


def get_reduced_qubit_density_matrix(stateop, qubit_index: int,
                                      num_qumodes: int, cutoff: int):
    """
    Compute reduced density matrix for a single qubit.

    Traces out all qumodes and other qubits to obtain the
    reduced state of the specified qubit.

    Args:
        stateop: Full quantum state (Qiskit DensityMatrix or Statevector).
        qubit_index: Index of the qubit to keep (0-indexed).
        num_qumodes: Total number of qumodes.
        cutoff: Fock space cutoff dimension.

    Returns:
        Reduced density matrix for the specified qubit.
    """
    num_qubits_per_qumode = int(np.ceil(np.log2(cutoff)))
    offset = num_qumodes * num_qubits_per_qumode
    total_qubits = stateop.num_qubits

    target_index = offset + qubit_index
    trace_indices = [i for i in range(total_qubits) if i != target_index]

    return partial_trace(stateop, trace_indices)


def wigner_negativity_all_modes(stateop, num_qumodes: int, cutoff: int,
                                 axes_min: float = -6, axes_max: float = 6,
                                 axes_steps: int = 500, g: float = None,
                                 method: str = "clenshaw") -> float:
    """
    Compute average Wigner negativity across all qumodes.

    Wigner negativity is a measure of non-classicality. Values > 0
    indicate quantum states that cannot be described classically.

    Args:
        stateop: Full quantum state.
        num_qumodes: Number of qumodes in the system.
        cutoff: Fock space cutoff dimension.
        axes_min: Minimum phase space coordinate.
        axes_max: Maximum phase space coordinate.
        axes_steps: Number of grid points.
        g: Scaling factor (default sqrt(2)).
        method: Wigner function computation method.

    Returns:
        Average Wigner negativity across all modes (0 to 1).
    """
    if g is None:
        g = np.sqrt(2)

    total_negativity = 0
    for i in range(num_qumodes):
        red_dm = get_reduced_qumode_density_matrix(stateop, i, num_qumodes, cutoff)
        xvec = np.linspace(axes_min, axes_max, axes_steps)
        W = c2qa.wigner._wigner(red_dm, xvec, g=g, method=method)

        dx = dy = (axes_max - axes_min) / (axes_steps - 1)
        area = np.sum(W) * dx * dy
        W /= area  # Normalize so integral = 1
        abs_area = np.sum(np.abs(W)) * dx * dy

        negativity = 0.5 * (abs_area - 1.0)
        negativity = min(max(negativity, 0), 1)
        total_negativity += negativity

        print(f"Mode {i}: integral(W) = 1.000, integral(|W|) = {abs_area:.3f}, "
              f"Negativity = {negativity:.3f}")

    return total_negativity / num_qumodes


def truncation_cost_all_modes(stateop, num_qumodes: int, cutoff: int,
                               n_tail: int = 5) -> float:
    """
    Compute average tail probability over all qumodes.

    Measures how much population resides in the highest Fock states,
    indicating potential truncation errors.

    Args:
        stateop: Full quantum state.
        num_qumodes: Number of qumodes.
        cutoff: Fock space cutoff dimension.
        n_tail: Number of highest Fock levels to include in tail.

    Returns:
        Average tail probability across all modes.
    """
    total_tail = 0
    for i in range(num_qumodes):
        red_dm = get_reduced_qumode_density_matrix(stateop, i, num_qumodes, cutoff)
        diag_probs = np.real(np.diag(red_dm.data))
        tail = sum(diag_probs[-n_tail:])
        total_tail += tail

    return total_tail / num_qumodes


def average_energy_all(stateop, num_qumodes: int, num_qubits: int,
                       cutoff: int, omega_qumode: float = 1.0,
                       omega_qubit: float = 1.0) -> float:
    """
    Compute total energy from multiple qumodes and qubits.

    Args:
        stateop: Full quantum state.
        num_qumodes: Number of qumodes.
        num_qubits: Number of qubits.
        cutoff: Fock space cutoff dimension.
        omega_qumode: Qumode frequency (energy scale).
        omega_qubit: Qubit frequency (energy scale).

    Returns:
        Total energy expectation value.
    """
    E = 0

    for i in range(num_qumodes):
        red_dm = get_reduced_qumode_density_matrix(stateop, i, num_qumodes, cutoff)
        n_op = num(cutoff).full()
        E += omega_qumode * np.trace(red_dm.data @ n_op).real

    for j in range(num_qubits):
        red_dm = get_reduced_qubit_density_matrix(stateop, j, num_qumodes, cutoff)
        sz = np.array([[1, 0], [0, -1]])
        E += omega_qubit * np.trace(red_dm.data @ sz).real

    return E


def evaluate_quantum_metrics(circuit: c2qa.CVCircuit, stateop, cutoff: int,
                             num_qumodes: int = 1, num_qubits: int = 1,
                             n_tail: int = 5, omega_qubit: float = 1.0,
                             omega_qumode: float = 1.0) -> tuple:
    """
    Evaluate comprehensive quantum metrics for a CV-DV circuit.

    Computes truncation cost, Wigner negativity, and average energy
    for the given circuit state.

    Args:
        circuit: The CVCircuit (used for context).
        stateop: Full output state from simulation.
        cutoff: Fock space cutoff dimension.
        num_qumodes: Number of qumodes in the circuit.
        num_qubits: Number of qubits in the circuit.
        n_tail: Number of Fock levels for truncation tail.
        omega_qubit: Qubit energy prefactor.
        omega_qumode: Qumode energy prefactor.

    Returns:
        Tuple of (truncation_cost, wigner_negativity, average_energy).
    """
    trunc = truncation_cost_all_modes(stateop, num_qumodes, cutoff, n_tail=n_tail)
    wigner_area = wigner_negativity_all_modes(stateop, num_qumodes, cutoff)
    avg_energy = average_energy_all(
        stateop, num_qumodes, num_qubits, cutoff,
        omega_qubit=omega_qubit, omega_qumode=omega_qumode
    )

    return trunc, wigner_area, avg_energy
