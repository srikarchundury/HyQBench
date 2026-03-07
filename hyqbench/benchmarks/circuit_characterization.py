"""
Circuit characterization and benchmarking for CV-DV hybrid quantum systems.

This module provides tools for characterizing and comparing various quantum
circuits including state preparation, VQE, QAOA, JCH simulation, and Shor's
algorithm implementations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from qiskit import QuantumRegister, ClassicalRegister

# import c2qa
import bosonic_qiskit as c2qa
from custom_gates import bosonic_vqe

from .benchmarks_circuit import (
    cat_state_circuit,
    gkp_state_circuit,
    JCH_simulation_circuit_display,
    binary_knapsack_vqe,
    binary_knapsack_vqe_circuit,
    shors_circuit,
    qft_circuit,
    state_transfer_CVtoDV,
    cv_qaoa,
    cv_qaoa_circuit,
)
from .features import (
    collect_cvcircuit_metrics,
    evaluate_quantum_metrics,
)


# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "circuit_characters")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRUCTURAL_KEYS = ['Qubits', 'Qumodes', 'Qubit Gates', 'Qumode Gates',
                   'Hybrid Gates', 'Circuit Depth']
PERFORMANCE_KEYS = ['Truncation Cost', 'Wigner Negativity', 'Average Energy']


def plot_radar_group(metrics_dict: dict, keys: list, filename: str) -> None:
    """
    Plot radar charts for multiple circuits in a grid layout.

    Creates a multi-panel figure with one radar chart per circuit,
    showing the specified metrics.

    Args:
        metrics_dict: Dictionary mapping circuit names to metric dictionaries.
        keys: List of metric keys to display on the radar chart.
        filename: Output file path for the saved figure.
    """
    num_plots = len(metrics_dict)
    rows = 2
    cols = (num_plots + 1) // 2

    fig, axs = plt.subplots(rows, cols, subplot_kw=dict(polar=True),
                            figsize=(4.2 * cols, 4.2 * rows))
    fig.patch.set_facecolor('white')
    axs = axs.flatten()

    for idx, (label, metrics) in enumerate(metrics_dict.items()):
        ax = axs[idx]
        N = len(keys)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        raw_vals = [metrics.get(k, 0) for k in keys]
        raw_vals += raw_vals[:1]
        max_vals = max(raw_vals)
        max_radius = max_vals * 1.2 if max_vals != 0 else 1.0

        color = plt.cm.tab10(idx % 10)

        ax.set_facecolor((*color[:3], 0.07))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(keys, fontsize=14, color="#222222", fontweight='medium')
        ax.set_yticks([])
        ax.set_ylim(0, max_radius)
        ax.grid(color="gray", linestyle="--", linewidth=0.5)
        ax.spines['polar'].set_visible(False)

        for radius in np.linspace(0.2, 1.0, 4) * max_radius:
            ax.add_patch(Circle((0, 0), radius, transform=ax.transData._b,
                                color=color, alpha=0.08, zorder=0))

        ax.plot(angles, raw_vals, linewidth=1.8, color=color, label=label)
        ax.fill(angles, raw_vals, color=color, alpha=0.25)

        for j in range(N):
            angle = angles[j]
            r = raw_vals[j]
            val = raw_vals[j]
            key_lower = keys[j].lower()
            if isinstance(val, int) or (val == int(val) and
                    key_lower.startswith(("qubit", "qumode", "depth", "gate"))):
                val_str = f"{int(val)}"
            else:
                val_str = f"{val:.2f}"
            ax.text(angle, r + 0.1 * max_radius, val_str,
                    ha='center', va='center', fontsize=12, color="#111111",
                    fontweight='semibold')

        ax.set_title(label, fontsize=16, pad=24, color=color, weight='bold')

    for i in range(len(metrics_dict), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Radar chart saved to {filename}")


def average_over_timesteps(circuit_template: c2qa.CVCircuit, U1,
                           qmr: c2qa.QumodeRegister, qbr: QuantumRegister,
                           cutoff: int, steps: int, dt: float,
                           num_qumodes: int, num_qubits: int,
                           sample_every: int = 5) -> dict:
    """
    Compute time-averaged quantum metrics for a dynamic simulation.

    Repeatedly applies a time evolution gate and samples metrics
    at specified intervals.

    Args:
        circuit_template: Base circuit to evolve.
        U1: Time evolution gate to apply.
        qmr: Qumode register.
        qbr: Qubit register.
        cutoff: Fock space cutoff.
        steps: Total number of time steps.
        dt: Time step duration.
        num_qumodes: Number of qumodes.
        num_qubits: Number of qubits.
        sample_every: Interval between metric samples.

    Returns:
        Dictionary with averaged truncation cost, Wigner negativity, and energy.
    """
    trunc_costs, wigner_negs, energies = [], [], []
    circuit = circuit_template.copy()

    for i, t in enumerate(np.arange(0, steps * dt, dt)):
        # circuit.append(U1, qmr[:] + qbr[:])
        circuit.cv_gate_from_matrix(
            U1.full(),
            qumodes=qmr[:],
            qubits=qbr[:],
            label=f"U1_t{t:.2f}",
        )
        if i % sample_every == 0:
            state, _, _ = c2qa.util.simulate(circuit)
            trunc, wneg, energy = evaluate_quantum_metrics(
                circuit, state, cutoff, num_qumodes, num_qubits
            )
            trunc_costs.append(trunc)
            wigner_negs.append(wneg)
            energies.append(energy)

    return {
        "Truncation Cost": np.mean(trunc_costs),
        "Wigner Negativity": np.mean(wigner_negs),
        "Average Energy": np.mean(energies)
    }


def characterize_circuit(name: str, circuit: c2qa.CVCircuit, cutoff: int,
                         num_qubits: int = 1, num_qumodes: int = 1,
                         stateop=None) -> dict:
    """
    Compute full characterization metrics for a circuit.

    Collects both structural metrics (gate counts, depth) and
    performance metrics (truncation, Wigner negativity, energy).

    Args:
        name: Circuit name for identification.
        circuit: The circuit to characterize.
        cutoff: Fock space cutoff.
        num_qubits: Number of qubits.
        num_qumodes: Number of qumodes.
        stateop: Optional simulated state for quantum metrics.

    Returns:
        Dictionary containing all computed metrics.
    """
    metrics = collect_cvcircuit_metrics(circuit, cutoff)
    if stateop is not None:
        trunc, wneg, energy = evaluate_quantum_metrics(
            circuit, stateop, cutoff, num_qumodes, num_qubits
        )
        metrics.update({
            "Truncation Cost": trunc,
            "Wigner Negativity": wneg,
            "Average Energy": energy
        })
    return metrics


def main():
    """Run comprehensive benchmark characterization for all circuits."""
    cutoff = 64
    struct_all = {}
    perf_all = {}

    # State Transfer CV to DV
    qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=6, name='qumode')
    qbr = QuantumRegister(4)
    cr = ClassicalRegister(4)
    circuit = c2qa.CVCircuit(qmr, qbr, cr)
    circuit = state_transfer_CVtoDV(cutoff, circuit, qmr, qbr, cr, 4)
    state, _, _ = c2qa.util.simulate(circuit)
    metrics = characterize_circuit("StateTransferCVtoDV", circuit, cutoff, 4, 1, state)
    struct_all["StateTransferCVtoDV"] = {k: metrics[k] for k in STRUCTURAL_KEYS}
    perf_all["StateTransferCVtoDV"] = {k: metrics[k] for k in PERFORMANCE_KEYS}

    # Cat State
    qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=6, name='qumode')
    qbr = QuantumRegister(1)
    circuit = c2qa.CVCircuit(qmr, qbr)
    circuit = cat_state_circuit(cutoff, circuit, qbr, qmr, alpha=4)
    state, _, _ = c2qa.util.simulate(circuit)
    metrics = characterize_circuit("Cat State", circuit, cutoff, 1, 1, state)
    struct_all["Cat State"] = {k: metrics[k] for k in STRUCTURAL_KEYS}
    perf_all["Cat State"] = {k: metrics[k] for k in PERFORMANCE_KEYS}

    # GKP State
    qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=6, name='qumode')
    qbr = QuantumRegister(1)
    circuit = c2qa.CVCircuit(qmr, qbr)
    circuit = gkp_state_circuit(cutoff, circuit, qbr, qmr)
    state, _, _ = c2qa.util.simulate(circuit)
    metrics = characterize_circuit("GKP State", circuit, cutoff, 1, 1, state)
    struct_all["GKP State"] = {k: metrics[k] for k in STRUCTURAL_KEYS}
    perf_all["GKP State"] = {k: metrics[k] for k in PERFORMANCE_KEYS}

    # QFT Circuit
    circuit = qft_circuit(16, 1.1, 2, 1, 2)
    state, _, _ = c2qa.util.simulate(circuit)
    metrics = characterize_circuit("QFT Circuit", circuit, 16, 5, 1, state)
    struct_all["QFT Circuit"] = {k: metrics[k] for k in STRUCTURAL_KEYS}
    perf_all["QFT Circuit"] = {k: metrics[k] for k in PERFORMANCE_KEYS}

    # JCH Simulation
    cutoff = 4
    Nsites = 3
    qmr = c2qa.QumodeRegister(Nsites, num_qubits_per_qumode=2)
    qbr = QuantumRegister(Nsites)
    circuit_template = c2qa.CVCircuit(qmr, qbr)
    circuit_template.cv_initialize(2, qmr[0])
    U1 = JCH_simulation_circuit_display(
        Nsites, Nqubits=Nsites, cutoff=cutoff,
        J=0.1, omega_r=2*np.pi*2, omega_q=2*np.pi*3,
        g=2*np.pi*0.5, tau=0.1, timesteps=1
    )
    perf = average_over_timesteps(
        circuit_template, U1, qmr, qbr, cutoff,
        steps=50, dt=0.1, num_qumodes=Nsites, num_qubits=Nsites
    )
    circuit = JCH_simulation_circuit_display(
        Nsites, Nqubits=Nsites, cutoff=cutoff,
        J=0.1, omega_r=2*np.pi*2, omega_q=2*np.pi*3,
        g=2*np.pi*0.5, tau=0.1, timesteps=1
    )
    struct = collect_cvcircuit_metrics(circuit, cutoff)
    struct_all[f"JCH N={Nsites}"] = {k: struct[k] for k in STRUCTURAL_KEYS}
    perf_all[f"JCH N={Nsites}"] = perf

    # Shor's Algorithm
    cutoff = 64
    circuit = shors_circuit(15, 2, 15, 2, 0.222, cutoff)
    state, _, _ = c2qa.util.simulate(circuit)
    metrics = characterize_circuit("Shors Circuit", circuit, cutoff, 1, 3, state)
    struct_all["Shor's Circuit"] = {k: metrics[k] for k in STRUCTURAL_KEYS}
    perf_all["Shor's Circuit"] = {k: metrics[k] for k in PERFORMANCE_KEYS}

    # Generate summary charts
    plot_radar_group(struct_all, STRUCTURAL_KEYS,
                     os.path.join(OUTPUT_DIR, "summary_structural.png"))
    plot_radar_group(perf_all, PERFORMANCE_KEYS,
                     os.path.join(OUTPUT_DIR, "summary_quantum.png"))


if __name__ == "__main__":
    main()
