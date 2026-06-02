from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import qasmpi
import qiskit
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath
from bosonic_converters import CircuitConverters
from bosonic_model.qasm import Translator
from bosonic_sdk.gate_statistics import GateStatistics
from bosonic_sdk.distributor.distributors.bosonic_distributor import BosonicDistributor
from bosonic_sdk.distributor.distributors.disqco_distributor import DisqcoDistributor
from bosonic_sdk.distributor.distributors.hypergraph_distributor import HypergraphDistributor
from disqco import (
    PartitionedCircuitExtractor,
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    set_initial_partition_assignment,
)
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
from disqco.parti import FiducciaMattheyses
from disqco.utils.qiskit_to_op_list import circuit_to_gate_layers, layer_list_to_dict


TWO_QUBIT_GATE_NAMES = {
    "cx",
    "cz",
    "cp",
    "rzz",
    "swap",
    "iswap",
    "ecr",
}


@dataclass
class DisqcoBenchmark:
    key: str
    source: str
    qasm: str
    circuit: Any
    original_stats: dict[str, Any]
    qiskit_circuit: Any
    normalized_circuit: Any
    layers: list[list[dict[str, Any]]]
    temporal_nodes: list[tuple[int, int]]

    @property
    def num_qubits(self) -> int:
        return int(self.normalized_circuit.num_qubits)

    @property
    def temporal_depth(self) -> int:
        return len(self.layers)


@dataclass
class AssignmentSetup:
    qpu_sizes: dict[str, int]
    qpu_labels: list[str]
    qpu_to_index: dict[str, int]
    index_to_qpu: dict[int, str]
    network: Any
    naive_assignment_matrix: np.ndarray
    initial_assignment_matrix: np.ndarray
    round_robin_assignment_matrix: np.ndarray
    naive_assignment: dict[tuple[int, int], str]
    initial_assignment: dict[tuple[int, int], str]
    round_robin_assignment: dict[tuple[int, int], str]


@dataclass
class PartitionResult:
    results: dict[str, Any]
    assignment_matrix: np.ndarray
    assignment: dict[tuple[int, int], str]
    runtime_s: float


@dataclass
class BenchmarkPartitionSummary:
    temporal_hyperedges: list[dict[str, Any]]
    hyperedge_preview_df: pd.DataFrame
    hyperedge_kind_summary_df: pd.DataFrame
    largest_communication_hyperedges_df: pd.DataFrame
    assignment_setup: AssignmentSetup
    naive_cost_df: pd.DataFrame
    initial_cost_df: pd.DataFrame
    baseline_cost_df: pd.DataFrame
    baseline_cost_summary_df: pd.DataFrame
    partition_result: PartitionResult
    assignment_map: dict[str, dict[tuple[int, int], str]]
    assignment_comparison_df: pd.DataFrame
    remote_link_summary_df: pd.DataFrame
    remote_link_summary_display_df: pd.DataFrame


@dataclass
class StateMoveDemo:
    circuit: Any
    normalized_circuit: Any
    layers: list[list[dict[str, Any]]]
    graph: Any
    temporal_hyperedges: list[dict[str, Any]]
    assignment_setup: AssignmentSetup
    partition_result: PartitionResult
    assignment_map: dict[str, dict[tuple[int, int], str]]
    remote_summary_df: pd.DataFrame
    assignment_path_df: pd.DataFrame
    movement_summary_df: pd.DataFrame
    moving_qubits_df: pd.DataFrame
    moving_qubits_display_df: pd.DataFrame
    lowered_circuit: Any
    lowered_runtime_s: float
    lowered_count_df: pd.DataFrame
    lowered_count_display_df: pd.DataFrame


def op_qubits(op: dict[str, Any]) -> tuple[int, ...]:
    return tuple(op.get("qargs", op.get("qubits", ())))


def format_op(op: dict[str, Any]) -> str:
    q_str = ", ".join(f"q{q}" for q in op_qubits(op))
    params = op.get("params", [])
    param_str = ", " + ", ".join(f"{float(p):.3g}" for p in params) if params else ""
    return f"{op['name']}({q_str}{param_str})"


def disqco_layers(circuit: Any) -> list[list[dict[str, Any]]]:
    raw_layer_dict = layer_list_to_dict(circuit_to_gate_layers(circuit))
    return [raw_layer_dict[i] for i in sorted(raw_layer_dict)]


def make_temporal_nodes(num_qubits: int, depth: int) -> list[tuple[int, int]]:
    return [(q, t) for q in range(num_qubits) for t in range(depth)]


def load_benchmark_qasm(key: str, fallback_path: Path) -> tuple[str, str]:
    try:
        return qasmpi.get_circuit(key), f"qasmpi:{key}"
    except Exception as exc:
        if not fallback_path.exists():
            raise RuntimeError(
                f"Could not fetch {key!r} with qasmpi and no local fallback exists "
                f"at {fallback_path}"
            ) from exc
        return fallback_path.read_text(), str(fallback_path)


def load_disqco_benchmark(key: str, fallback_path: Path) -> DisqcoBenchmark:
    benchmark_qasm, benchmark_source = load_benchmark_qasm(key, fallback_path)
    benchmark_circuit = Translator().from_qasm(benchmark_qasm)
    benchmark_original_stats = GateStatistics.stats(benchmark_circuit)
    benchmark_qiskit = CircuitConverters.to_qiskit(benchmark_circuit)
    benchmark_normalized = DisqcoDistributor._normalize_for_disqco(benchmark_qiskit)
    layers = disqco_layers(benchmark_normalized)
    temporal_nodes = make_temporal_nodes(benchmark_normalized.num_qubits, len(layers))
    return DisqcoBenchmark(
        key=key,
        source=benchmark_source,
        qasm=benchmark_qasm,
        circuit=benchmark_circuit,
        original_stats=benchmark_original_stats,
        qiskit_circuit=benchmark_qiskit,
        normalized_circuit=benchmark_normalized,
        layers=layers,
        temporal_nodes=temporal_nodes,
    )


def benchmark_overview_table(benchmark: DisqcoBenchmark) -> pd.DataFrame:
    stats = benchmark.original_stats
    normalized_counts = benchmark.normalized_circuit.count_ops()
    return pd.DataFrame(
        [
            {"metric": "n_qubits", "value": stats["n_qubits"]},
            {"metric": "n_clbits", "value": stats["n_clbits"]},
            {"metric": "original_total_ops", "value": stats["total_ops"]},
            {"metric": "original_depth", "value": stats["depth"]},
            {"metric": "normalized_basis", "value": ", ".join(sorted(normalized_counts))},
            {"metric": "temporal_layers", "value": benchmark.temporal_depth},
        ]
    )


def describe_disqco_layer(op: dict[str, Any]) -> dict[str, Any]:
    if op["type"] != "group":
        return {
            "type": op["type"],
            "root": "",
            "time_span": "",
            "subgate_count": 1,
            "contents": format_op(op),
        }

    subgate_text = []
    times = []
    for sub in op["sub-gates"]:
        times.append(sub["time"])
        subgate_text.append(
            f"{sub['name']}(" + ", ".join(f"q{q}" for q in sub["qargs"]) + f")@t={sub['time']}"
        )
    return {
        "type": "group",
        "root": f"q{op['root']}",
        "time_span": f"{min(times)}-{max(times)}" if times else "",
        "subgate_count": len(subgate_text),
        "contents": "; ".join(subgate_text),
    }


def grouped_layer_tables(disqco_graph: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for t, layer in sorted(disqco_graph.layers.items()):
        for op in layer:
            row = describe_disqco_layer(op)
            row["layer"] = t
            rows.append(row)

    grouped_layer_df = pd.DataFrame(rows)
    layer_type_summary_df = (
        grouped_layer_df.groupby("type", as_index=False)
        .agg(
            objects=("type", "size"),
            layers_touched=("layer", "nunique"),
            total_subgates=("subgate_count", "sum"),
            largest_object=("subgate_count", "max"),
        )
        .sort_values(["type"])
    )
    largest_grouped_objects_df = (
        grouped_layer_df[grouped_layer_df["type"] == "group"]
        .sort_values(["subgate_count", "layer"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )
    return grouped_layer_df, layer_type_summary_df, largest_grouped_objects_df


def is_temporal_state_edge(edge_data: dict[str, Any]) -> bool:
    root_set = set(edge_data["root_set"])
    receiver_set = set(edge_data["receiver_set"])
    if len(root_set) != 1 or len(receiver_set) != 1:
        return False
    (q0, t0), = root_set
    (q1, t1), = receiver_set
    return q0 == q1 and abs(t1 - t0) == 1


def classify_disqco_hyperedge(edge_data: dict[str, Any], attrs: dict[str, Any]) -> str:
    if is_temporal_state_edge(edge_data):
        return "state"
    edge_nodes = set(edge_data["root_set"]) | set(edge_data["receiver_set"])
    if attrs.get("type") == "two-qubit" or len(edge_nodes) == 2:
        return "gate"
    return "communication"


def sorted_temporal_nodes(nodes: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(nodes, key=lambda item: (item[1], item[0]))


def node_span(nodes: set[tuple[int, int]]) -> str:
    if not nodes:
        return ""
    times = [t for _, t in nodes]
    qubits = sorted({q for q, _ in nodes})
    q_text = ", ".join(f"q{q}" for q in qubits[:6])
    if len(qubits) > 6:
        q_text += f", ... +{len(qubits) - 6}"
    return f"{q_text}; t={min(times)}-{max(times)}"


def temporal_hyperedge_tables(
    disqco_graph: Any,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temporal_hyperedges = []
    for h_idx, (edge_id, edge_data) in enumerate(disqco_graph.hyperedges.items()):
        attrs = disqco_graph.hyperedge_attrs.get(edge_id, {})
        kind = classify_disqco_hyperedge(edge_data, attrs)
        name = attrs.get("name") or f"{kind}_{h_idx}"
        temporal_hyperedges.append(
            {
                "name": str(name),
                "kind": kind,
                "edge_id": edge_id,
                "e_root": set(edge_data["root_set"]),
                "e_rec": set(edge_data["receiver_set"]),
            }
        )

    hyperedge_preview_df = pd.DataFrame(
        [
            {
                "name": edge["name"],
                "kind": edge["kind"],
                "root_span": node_span(edge["e_root"]),
                "receiver_span": node_span(edge["e_rec"]),
                "root_nodes": len(edge["e_root"]),
                "receiver_nodes": len(edge["e_rec"]),
                "size": len(edge["e_root"] | edge["e_rec"]),
            }
            for edge in temporal_hyperedges
        ]
    )
    hyperedge_kind_summary_df = (
        hyperedge_preview_df.groupby("kind", as_index=False)
        .agg(
            hyperedges=("kind", "size"),
            total_temporal_nodes=("size", "sum"),
            median_size=("size", "median"),
            max_size=("size", "max"),
        )
        .sort_values("kind")
    )
    largest_communication_hyperedges_df = (
        hyperedge_preview_df[hyperedge_preview_df["kind"] == "communication"]
        .sort_values(["size", "name"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )
    return (
        temporal_hyperedges,
        hyperedge_preview_df,
        hyperedge_kind_summary_df,
        largest_communication_hyperedges_df,
    )


def communication_size_distribution(hyperedge_preview_df: pd.DataFrame) -> pd.DataFrame:
    size_bins = [0, 2, 4, 8, 16, 32, 64, float("inf")]
    size_labels = ["1-2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"]
    return (
        hyperedge_preview_df[hyperedge_preview_df["kind"] == "communication"]
        .assign(size_bucket=lambda df: pd.cut(df["size"], bins=size_bins, labels=size_labels))
        .groupby("size_bucket", observed=True, as_index=False)
        .size()
        .rename(columns={"size": "hyperedges"})
    )


def assignment_from_matrix(
    matrix: np.ndarray,
    index_to_qpu: dict[int, str],
) -> dict[tuple[int, int], str]:
    matrix = np.asarray(matrix, dtype=int)
    return {
        (q, t): index_to_qpu[int(matrix[t][q])]
        for t in range(matrix.shape[0])
        for q in range(matrix.shape[1])
    }


def naive_static_assignment_matrix(
    temporal_num_qubits: int,
    temporal_depth: int,
    qpu_sizes: dict[str, int],
) -> np.ndarray:
    if sum(qpu_sizes.values()) < temporal_num_qubits:
        raise ValueError("QPU capacities are too small for the logical circuit")

    qpu_indices = []
    current_qpu = 0
    used_on_current_qpu = 0
    capacities = list(qpu_sizes.values())
    for _ in range(temporal_num_qubits):
        while used_on_current_qpu >= capacities[current_qpu]:
            current_qpu += 1
            used_on_current_qpu = 0
        qpu_indices.append(current_qpu)
        used_on_current_qpu += 1

    static_row = np.asarray(qpu_indices, dtype=int)
    return np.tile(static_row, (temporal_depth, 1))


def make_assignment_setup(
    disqco_graph: Any,
    temporal_num_qubits: int,
    temporal_depth: int,
    qpu_sizes: dict[str, int],
) -> AssignmentSetup:
    qpu_labels = list(qpu_sizes)
    qpu_to_index = {label: idx for idx, label in enumerate(qpu_labels)}
    index_to_qpu = {idx: label for label, idx in qpu_to_index.items()}
    network = QuantumNetwork.create([qpu_sizes[label] for label in qpu_labels], "all_to_all")
    initial_assignment_matrix = np.asarray(
        set_initial_partition_assignment(disqco_graph, network),
        dtype=int,
    )
    naive_assignment_matrix = naive_static_assignment_matrix(
        temporal_num_qubits,
        temporal_depth,
        qpu_sizes,
    )
    round_robin_assignment_matrix = np.array(
        [
            [q % len(qpu_labels) for q in range(temporal_num_qubits)]
            for _ in range(temporal_depth)
        ],
        dtype=int,
    )
    return AssignmentSetup(
        qpu_sizes=qpu_sizes,
        qpu_labels=qpu_labels,
        qpu_to_index=qpu_to_index,
        index_to_qpu=index_to_qpu,
        network=network,
        naive_assignment_matrix=naive_assignment_matrix,
        initial_assignment_matrix=initial_assignment_matrix,
        round_robin_assignment_matrix=round_robin_assignment_matrix,
        naive_assignment=assignment_from_matrix(naive_assignment_matrix, index_to_qpu),
        initial_assignment=assignment_from_matrix(initial_assignment_matrix, index_to_qpu),
        round_robin_assignment=assignment_from_matrix(round_robin_assignment_matrix, index_to_qpu),
    )


def root_receiver_sets(
    edge: dict[str, Any],
    assignment: dict[tuple[int, int], str],
) -> tuple[set[str], set[str]]:
    r_e = {assignment[v] for v in edge["e_rec"]}
    s_e = {assignment[u] for u in edge["e_root"]}
    return r_e, s_e


def hyperedge_cost(
    edge: dict[str, Any],
    assignment: dict[tuple[int, int], str],
    setup: AssignmentSetup,
) -> tuple[int, set[str], set[str]]:
    r_e, s_e = root_receiver_sets(edge, assignment)
    root_config = tuple(
        1 if setup.index_to_qpu[i] in s_e else 0
        for i in range(len(setup.qpu_labels))
    )
    receiver_config = tuple(
        1 if setup.index_to_qpu[i] in r_e else 0
        for i in range(len(setup.qpu_labels))
    )
    _, network_cost = setup.network.steiner_forest(root_config, receiver_config)
    return network_cost, r_e, s_e


def total_hyperedge_cost(
    hyperedges: list[dict[str, Any]],
    assignment: dict[tuple[int, int], str],
    setup: AssignmentSetup,
) -> int:
    return sum(hyperedge_cost(edge, assignment, setup)[0] for edge in hyperedges)


def cost_table(
    hyperedges: list[dict[str, Any]],
    assignment: dict[tuple[int, int], str],
    label: str,
    setup: AssignmentSetup,
) -> pd.DataFrame:
    rows = []
    for edge in hyperedges:
        c_e, r_e, s_e = hyperedge_cost(edge, assignment, setup)
        rows.append(
            {
                "assignment": label,
                "name": edge["name"],
                "kind": edge["kind"],
                "R_e": sorted(r_e),
                "S_e": sorted(s_e),
                "c_e": int(c_e),
            }
        )
    return pd.DataFrame(rows).sort_values(["assignment", "kind", "name"]).reset_index(drop=True)


def baseline_assignment_cost_tables(
    temporal_hyperedges: list[dict[str, Any]],
    setup: AssignmentSetup,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    naive_cost_df = cost_table(
        temporal_hyperedges,
        setup.naive_assignment,
        "naive static baseline",
        setup,
    )
    initial_cost_df = cost_table(
        temporal_hyperedges,
        setup.initial_assignment,
        "DISQCO initial fill",
        setup,
    )
    baseline_cost_df = pd.concat([naive_cost_df, initial_cost_df], ignore_index=True)
    baseline_cost_summary_df = (
        baseline_cost_df.groupby(["assignment", "kind"], as_index=False)
        .agg(
            hyperedges=("kind", "size"),
            cut_hyperedges=("c_e", lambda values: int((values > 0).sum())),
            cut_cost=("c_e", "sum"),
        )
        .sort_values(["assignment", "kind"])
    )
    return naive_cost_df, initial_cost_df, baseline_cost_df, baseline_cost_summary_df


def run_disqco_partition(
    normalized_circuit: Any,
    disqco_graph: Any,
    setup: AssignmentSetup,
    *,
    passes_per_level: int = 2,
) -> PartitionResult:
    partitioner = FiducciaMattheyses(
        normalized_circuit,
        setup.network,
        setup.initial_assignment_matrix,
        hypergraph=disqco_graph,
        group_gates=True,
    )
    started = perf_counter()
    results = partitioner.multilevel_partition(
        coarsener=HypergraphCoarsener().coarsen_recursive_batches_mapped,
        passes_per_level=passes_per_level,
    )
    runtime_s = perf_counter() - started
    assignment_matrix = np.asarray(results["best_assignment"], dtype=int)
    return PartitionResult(
        results=results,
        assignment_matrix=assignment_matrix,
        assignment=assignment_from_matrix(assignment_matrix, setup.index_to_qpu),
        runtime_s=runtime_s,
    )


def assignment_is_time_dependent(
    assignment: dict[tuple[int, int], str],
    temporal_num_qubits: int,
    temporal_depth: int,
) -> bool:
    return any(
        len({assignment[(q, t)] for t in range(temporal_depth)}) > 1
        for q in range(temporal_num_qubits)
    )


def cut_summary(
    temporal_hyperedges: list[dict[str, Any]],
    assignment: dict[tuple[int, int], str],
    setup: AssignmentSetup,
) -> dict[str, int]:
    df = cost_table(temporal_hyperedges, assignment, "assignment", setup)
    return {
        "state_move_cost": int(df.loc[df["kind"] == "state", "c_e"].sum()),
        "remote_gate_or_group_cost": int(
            df.loc[df["kind"].isin(["gate", "communication"]), "c_e"].sum()
        ),
        "total_cost": int(df["c_e"].sum()),
    }


def assignment_comparison_table(
    temporal_hyperedges: list[dict[str, Any]],
    assignment_map: dict[str, dict[tuple[int, int], str]],
    setup: AssignmentSetup,
    temporal_num_qubits: int,
    temporal_depth: int,
) -> pd.DataFrame:
    rows = []
    for label, assignment in assignment_map.items():
        summary = cut_summary(temporal_hyperedges, assignment, setup)
        rows.append(
            {
                "assignment": label,
                "total_hypergraph_cost": summary["total_cost"],
                "remote_gate_or_group_cost": summary["remote_gate_or_group_cost"],
                "state_move_cost": summary["state_move_cost"],
                "time_dependent": assignment_is_time_dependent(
                    assignment,
                    temporal_num_qubits,
                    temporal_depth,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("total_hypergraph_cost").reset_index(drop=True)


def gate_partition_rows(
    layers: list[list[dict[str, Any]]],
    assignment: dict[tuple[int, int], str],
    label: str,
) -> pd.DataFrame:
    rows = []
    for t, layer_ops in enumerate(layers):
        for op in layer_ops:
            qubits = op_qubits(op)
            if not qubits:
                continue
            qpus = sorted({assignment[(q, t)] for q in qubits})
            rows.append(
                {
                    "assignment": label,
                    "t": t,
                    "gate": op["name"],
                    "qubits": ", ".join(f"q{q}" for q in qubits),
                    "qpus": ", ".join(qpus),
                    "placement": qpus[0] if len(qpus) == 1 else "remote",
                }
            )
    return pd.DataFrame(rows)


def remote_gate_summary(
    layers: list[list[dict[str, Any]]],
    assignment: dict[tuple[int, int], str],
    label: str,
) -> pd.DataFrame:
    gate_df = gate_partition_rows(layers, assignment, label)
    remote_df = gate_df[
        (gate_df["placement"] == "remote")
        & (gate_df["gate"].isin({"cx", "cp", "cz"}))
    ]
    if remote_df.empty:
        return pd.DataFrame(columns=["assignment", "remote_gate_count"])
    return (
        remote_df.groupby("assignment", as_index=False)
        .size()
        .rename(columns={"size": "remote_gate_count"})
    )


def remote_episode_summary(
    hyperedges: list[dict[str, Any]],
    assignment: dict[tuple[int, int], str],
    label: str,
    setup: AssignmentSetup,
) -> pd.DataFrame:
    rows = []
    for edge in hyperedges:
        c_e, _, _ = hyperedge_cost(edge, assignment, setup)
        if c_e == 0:
            continue
        rows.append({"assignment": label, "kind": edge["kind"], "count": int(c_e)})
    if not rows:
        return pd.DataFrame(columns=["assignment", "kind", "count"])
    return pd.DataFrame(rows).groupby(["assignment", "kind"], as_index=False)["count"].sum()


def remote_link_summary_table(
    layers: list[list[dict[str, Any]]],
    temporal_hyperedges: list[dict[str, Any]],
    assignment_map: dict[str, dict[tuple[int, int], str]],
    setup: AssignmentSetup,
) -> pd.DataFrame:
    rows = []
    for label, assignment in assignment_map.items():
        episode_df = remote_episode_summary(temporal_hyperedges, assignment, label, setup)
        remote_gates = remote_gate_summary(layers, assignment, label)
        remote_gate_count = (
            int(remote_gates["remote_gate_count"].sum()) if not remote_gates.empty else 0
        )
        episode_counts = (
            dict(zip(episode_df["kind"], episode_df["count"]))
            if not episode_df.empty
            else {}
        )
        rows.append(
            {
                "assignment": label,
                "remote_two_qubit_gates": remote_gate_count,
                "cut_gate_or_group_episodes": episode_counts.get("gate", 0)
                + episode_counts.get("communication", 0),
                "state_moves": episode_counts.get("state", 0),
                "total_hypergraph_cost": total_hyperedge_cost(
                    temporal_hyperedges,
                    assignment,
                    setup,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("total_hypergraph_cost").reset_index(drop=True)


def qubit_assignment_summary(
    matrix: np.ndarray,
    label: str,
    index_to_qpu: dict[int, str],
) -> pd.DataFrame:
    matrix = np.asarray(matrix, dtype=int)
    rows = []
    for q in range(matrix.shape[1]):
        qpu_path = [index_to_qpu[int(matrix[t][q])] for t in range(matrix.shape[0])]
        compressed_path = []
        for qpu in qpu_path:
            if not compressed_path or compressed_path[-1] != qpu:
                compressed_path.append(qpu)
        rows.append(
            {
                "assignment": label,
                "qubit": f"q{q}",
                "start_qpu": qpu_path[0],
                "end_qpu": qpu_path[-1],
                "distinct_qpus": ", ".join(sorted(set(qpu_path))),
                "moves": sum(
                    left != right for left, right in zip(qpu_path, qpu_path[1:])
                ),
                "compressed_path": " -> ".join(compressed_path),
            }
        )
    return pd.DataFrame(rows)


def assignment_movement_tables(
    assignment_matrices: dict[str, np.ndarray],
    setup: AssignmentSetup,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment_path_df = pd.concat(
        [
            qubit_assignment_summary(matrix, label, setup.index_to_qpu)
            for label, matrix in assignment_matrices.items()
        ],
        ignore_index=True,
    )
    assignment_move_summary_df = (
        assignment_path_df.groupby("assignment", as_index=False)
        .agg(
            total_state_moves=("moves", "sum"),
            moved_logical_qubits=("moves", lambda values: int((values > 0).sum())),
        )
        .sort_values("assignment")
    )
    moving_qubits_df = assignment_path_df[assignment_path_df["moves"] > 0]
    if moving_qubits_df.empty:
        moving_qubits_df = assignment_path_df.sort_values(
            ["assignment", "qubit"]
        ).reset_index(drop=True)
    else:
        moving_qubits_df = moving_qubits_df.sort_values(
            ["assignment", "moves", "qubit"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    return assignment_path_df, assignment_move_summary_df, moving_qubits_df


def extract_lowered_circuit(
    disqco_graph: Any,
    network: Any,
    assignment_matrix: np.ndarray,
) -> tuple[Any, float, pd.DataFrame]:
    extractor = PartitionedCircuitExtractor(
        graph=disqco_graph,
        network=network,
        partition_assignment=assignment_matrix,
    )
    started = perf_counter()
    lowered_circuit = extractor.extract_partitioned_circuit()
    runtime_s = perf_counter() - started
    lowered_count_df = (
        pd.DataFrame(
            [
                {"operation": name, "count": count}
                for name, count in sorted(dict(lowered_circuit.count_ops()).items())
            ]
        )
        .sort_values(["count", "operation"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return lowered_circuit, runtime_s, lowered_count_df


def benchmark_partition_summary(
    benchmark: DisqcoBenchmark,
    disqco_graph: Any,
    *,
    expected_key: str = "knn_n25",
    qpu_sizes: dict[str, int] | None = None,
    passes_per_level: int = 2,
) -> BenchmarkPartitionSummary:
    if benchmark.key != expected_key:
        raise RuntimeError(
            f"This partitioning cell expects {expected_key!r}, "
            f"but the live kernel has {benchmark.key!r}. Re-run the benchmark cells."
        )

    expected_temporal_nodes = benchmark.num_qubits * benchmark.temporal_depth
    if len(disqco_graph.nodes) != expected_temporal_nodes:
        raise RuntimeError(
            "The active hypergraph does not match the active benchmark circuit. "
            "Re-run the hypergraph-building cell before partitioning."
        )

    qpu_sizes = qpu_sizes or {"QPU_0": 13, "QPU_1": 13}
    (
        temporal_hyperedges,
        hyperedge_preview_df,
        hyperedge_kind_summary_df,
        largest_communication_hyperedges_df,
    ) = temporal_hyperedge_tables(disqco_graph)
    assignment_setup = make_assignment_setup(
        disqco_graph,
        benchmark.num_qubits,
        benchmark.temporal_depth,
        qpu_sizes,
    )

    naive_cost_df, initial_cost_df, baseline_cost_df, baseline_cost_summary_df = (
        baseline_assignment_cost_tables(temporal_hyperedges, assignment_setup)
    )
    partition_result = run_disqco_partition(
        benchmark.normalized_circuit,
        disqco_graph,
        assignment_setup,
        passes_per_level=passes_per_level,
    )
    assignment_map = {
        "naive static baseline": assignment_setup.naive_assignment,
        "DISQCO initial": assignment_setup.initial_assignment,
        "DISQCO partitioned": partition_result.assignment,
    }
    assignment_comparison_df = assignment_comparison_table(
        temporal_hyperedges,
        assignment_map,
        assignment_setup,
        benchmark.num_qubits,
        benchmark.temporal_depth,
    )
    remote_link_summary_df = remote_link_summary_table(
        benchmark.layers,
        temporal_hyperedges,
        assignment_map,
        assignment_setup,
    )
    remote_link_summary_display_df = remote_link_summary_df.drop(columns=["state_moves"])
    return BenchmarkPartitionSummary(
        temporal_hyperedges=temporal_hyperedges,
        hyperedge_preview_df=hyperedge_preview_df,
        hyperedge_kind_summary_df=hyperedge_kind_summary_df,
        largest_communication_hyperedges_df=largest_communication_hyperedges_df,
        assignment_setup=assignment_setup,
        naive_cost_df=naive_cost_df,
        initial_cost_df=initial_cost_df,
        baseline_cost_df=baseline_cost_df,
        baseline_cost_summary_df=baseline_cost_summary_df,
        partition_result=partition_result,
        assignment_map=assignment_map,
        assignment_comparison_df=assignment_comparison_df,
        remote_link_summary_df=remote_link_summary_df,
        remote_link_summary_display_df=remote_link_summary_display_df,
    )


def build_state_move_handoff_circuit(
    *,
    num_qubits: int = 25,
    repetitions: int = 60,
    phase: float = 0.3,
) -> Any:
    if num_qubits < 5:
        raise ValueError("The state-move handoff circuit needs at least five qubits")

    circuit = qiskit.QuantumCircuit(num_qubits, name="state_move_handoff")
    split = 1 + (num_qubits - 1) // 2
    left_cluster = list(range(1, split))
    right_cluster = list(range(split, num_qubits))
    if not left_cluster or not right_cluster:
        raise ValueError("The handoff circuit needs non-empty left and right clusters")

    for i in range(repetitions):
        circuit.cp(phase, 0, left_cluster[i % len(left_cluster)])
        circuit.x(0)
    for i in range(repetitions):
        circuit.cp(phase, 0, right_cluster[i % len(right_cluster)])
        circuit.x(0)
    return circuit


def cleaned_lowered_count_table(lowered_count_df: pd.DataFrame) -> pd.DataFrame:
    display_df = lowered_count_df.copy()
    display_df["operation"] = display_df["operation"].astype(str).str.replace(
        r"^circuit-\d+$",
        "wrapper instruction",
        regex=True,
    )
    return (
        display_df.groupby("operation", as_index=False)["count"]
        .sum()
        .sort_values(["count", "operation"], ascending=[False, True])
        .reset_index(drop=True)
    )


def run_state_move_demo(
    *,
    num_qubits: int = 25,
    repetitions: int = 60,
    qpu_sizes: dict[str, int] | None = None,
    passes_per_level: int = 2,
) -> StateMoveDemo:
    qpu_sizes = qpu_sizes or {"QPU_0": 13, "QPU_1": 13}
    circuit = build_state_move_handoff_circuit(
        num_qubits=num_qubits,
        repetitions=repetitions,
    )
    normalized_circuit = DisqcoDistributor._normalize_for_disqco(circuit)
    layers = disqco_layers(normalized_circuit)
    graph = QuantumCircuitHyperGraph(normalized_circuit, group_gates=True)
    temporal_hyperedges, *_ = temporal_hyperedge_tables(graph)
    assignment_setup = make_assignment_setup(
        graph,
        normalized_circuit.num_qubits,
        len(layers),
        qpu_sizes,
    )
    partition_result = run_disqco_partition(
        normalized_circuit,
        graph,
        assignment_setup,
        passes_per_level=passes_per_level,
    )
    assignment_map = {
        "naive static baseline": assignment_setup.naive_assignment,
        "DISQCO initial": assignment_setup.initial_assignment,
        "DISQCO partitioned": partition_result.assignment,
    }
    remote_summary_df = remote_link_summary_table(
        layers,
        temporal_hyperedges,
        assignment_map,
        assignment_setup,
    )
    assignment_path_df, movement_summary_df, moving_qubits_df = assignment_movement_tables(
        {"DISQCO partitioned": partition_result.assignment_matrix},
        assignment_setup,
    )
    moving_qubits_display_df = moving_qubits_df[
        ["qubit", "start_qpu", "end_qpu", "moves", "compressed_path"]
    ]
    lowered_circuit, lowered_runtime_s, lowered_count_df = extract_lowered_circuit(
        graph,
        assignment_setup.network,
        partition_result.assignment_matrix,
    )
    lowered_count_display_df = cleaned_lowered_count_table(lowered_count_df)
    return StateMoveDemo(
        circuit=circuit,
        normalized_circuit=normalized_circuit,
        layers=layers,
        graph=graph,
        temporal_hyperedges=temporal_hyperedges,
        assignment_setup=assignment_setup,
        partition_result=partition_result,
        assignment_map=assignment_map,
        remote_summary_df=remote_summary_df,
        assignment_path_df=assignment_path_df,
        movement_summary_df=movement_summary_df,
        moving_qubits_df=moving_qubits_df,
        moving_qubits_display_df=moving_qubits_display_df,
        lowered_circuit=lowered_circuit,
        lowered_runtime_s=lowered_runtime_s,
        lowered_count_df=lowered_count_df,
        lowered_count_display_df=lowered_count_display_df,
    )


def print_state_move_demo_summary(demo: StateMoveDemo) -> None:
    print("State-move demo circuit:")
    print(f"  logical qubits: {demo.normalized_circuit.num_qubits}")
    print(f"  temporal layers: {len(demo.layers)}")
    print(f"  normalized operations: {dict(demo.normalized_circuit.count_ops())}")


def print_lowered_circuit_summary(
    lowered_circuit: Any,
    runtime_s: float,
) -> None:
    print("Lowered circuit registers:")
    print("  qregs:", [(reg.name, reg.size) for reg in lowered_circuit.qregs])
    print("  cregs:", [(reg.name, reg.size) for reg in lowered_circuit.cregs])
    print(f"Lowered extraction runtime: {runtime_s:.3f} s")


def load_dqcomp_summary_table(readme_path: str | Path) -> tuple[pd.DataFrame, str]:
    readme_path = Path(readme_path)
    readme_lines = readme_path.read_text().splitlines()
    try:
        table_start = next(
            i for i, line in enumerate(readme_lines) if line.startswith("| Circuit")
        )
    except StopIteration as exc:
        raise ValueError(f"Could not find the dqcomp summary table in {readme_path}") from exc

    table_lines = []
    for line in readme_lines[table_start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)

    columns = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        row = dict(zip(columns, cells))
        for column in columns[1:]:
            row[column] = int(row[column])
        rows.append(row)

    return pd.DataFrame(rows, columns=columns), "\n".join(table_lines)


def remote_primitive_counts(stats: dict[str, Any]) -> dict[str, int]:
    return {
        name: count
        for name, count in stats["gate_counts"].items()
        if name.startswith("remote_")
    }


def compact_stats_row(
    label: str,
    stats: dict[str, Any],
    elapsed: float | None = None,
) -> dict[str, Any]:
    remote_primitives = remote_primitive_counts(stats)
    return {
        "backend": label,
        "remote_gates": stats["remote_gate_count"],
        "total_ops": stats["total_ops"],
        "depth": stats["depth"],
        "n_qubits": stats["n_qubits"],
        "n_clbits": stats["n_clbits"],
        "measurements": stats["measure_count"],
        "resets": stats["reset_count"],
        "remote_primitives": remote_primitives or {},
        "basis_gates": ", ".join(stats["basis_gates"]),
        "runtime_s": None if elapsed is None else round(elapsed, 3),
    }


def run_backend_comparison(
    benchmark_circuit: Any,
    benchmark_original_stats: dict[str, Any],
    *,
    nodes: int,
    qubits_per_node: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], pd.DataFrame]:
    rows = [compact_stats_row("Original monolithic", benchmark_original_stats)]
    outputs = {}
    backend_runs = [
        ("Bosonic naive", BosonicDistributor(), False),
        ("Hypergraph lowered", HypergraphDistributor(), True),
        ("DISQCO lowered", DisqcoDistributor(num_passes=2), True),
    ]
    for label, distributor, lowered in backend_runs:
        started = perf_counter()
        distributed = distributor.distribute(
            benchmark_circuit,
            nodes=nodes,
            qubits_per_node=qubits_per_node,
            lowered=lowered,
        )
        elapsed = perf_counter() - started
        stats = GateStatistics.stats(distributed)
        outputs[label] = {
            "distributed": distributed,
            "stats": stats,
            "runtime_s": elapsed,
        }
        rows.append(compact_stats_row(label, stats, elapsed))
    return rows, outputs, pd.DataFrame(rows)


def two_qubit_gate_counts(stats: dict[str, Any]) -> dict[str, int]:
    return {
        name: count
        for name, count in stats["gate_counts"].items()
        if name in TWO_QUBIT_GATE_NAMES
    }


def resource_comparison_table(
    benchmark_rows: list[dict[str, Any]],
    benchmark_outputs: dict[str, dict[str, Any]],
    benchmark_original_stats: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for row in benchmark_rows:
        backend = row["backend"]
        if backend == "Original monolithic":
            stats = benchmark_original_stats
        else:
            stats = benchmark_outputs[backend]["stats"]

        remote_links = remote_primitive_counts(stats)
        two_qubit_gates = two_qubit_gate_counts(stats)
        rows.append(
            {
                "backend": backend,
                "remote_links": sum(remote_links.values()),
                "two_qubit_gates": sum(two_qubit_gates.values()),
                "remote_link_breakdown": remote_links or {},
                "two_qubit_gate_breakdown": two_qubit_gates or {},
            }
        )
    return pd.DataFrame(rows)


def draw_naive_distribution_example() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    x_min, x_max = -0.6, 3.7
    y_pos = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}

    qpu_bands = [
        ("QPU_0", 1.55, 1.9, "#e8f3ff", "#4c78a8", (0, 1)),
        ("QPU_1", -0.45, 1.9, "#eaf7ee", "#59a14f", (2, 3)),
    ]
    for label, y0, height, facecolor, edgecolor, qubits in qpu_bands:
        ax.add_patch(
            FancyBboxPatch(
                (x_min, y0),
                x_max - x_min,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.6,
                alpha=0.95,
            )
        )
        q_text = ", ".join(f"q{q}" for q in qubits)
        ax.text(
            x_max - 0.08,
            y0 + height - 0.18,
            f"{label}: {q_text}",
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=edgecolor,
        )

    for q, y in y_pos.items():
        ax.hlines(y, 0, 3.25, color="#4b5563", linewidth=1.5, alpha=0.8)
        ax.text(-0.12, y, f"q{q}", ha="right", va="center", fontsize=10, fontweight="bold")

    gates = [
        (0.35, (0, 1), "local", "local gate"),
        (1.25, (1, 2), "remote", "+1 remote link"),
        (2.05, (2, 3), "local", "local gate"),
        (2.95, (0, 3), "remote", "+1 remote link"),
    ]
    for x, qubits, kind, label in gates:
        y0, y1 = y_pos[qubits[0]], y_pos[qubits[1]]
        color = "#2d6a4f" if kind == "local" else "#d62728"
        linestyle = "-" if kind == "local" else "--"
        linewidth = 3 if kind == "local" else 3.4
        ax.plot([x, x], [y0, y1], color=color, linestyle=linestyle, linewidth=linewidth, zorder=3)
        for q in qubits:
            ax.add_patch(
                Circle(
                    (x, y_pos[q]),
                    0.095,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.4,
                    zorder=4,
                )
            )
        ax.text(
            x,
            max(y0, y1) + 0.22,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=color,
            fontweight="bold" if kind == "remote" else "normal",
        )

    ax.annotate(
        "Naive rule: keep the placement fixed; every crossing two-qubit gate gets a fresh remote link.",
        xy=(1.25, 1.5),
        xytext=(0.1, -0.95),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.6},
        fontsize=10,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d62728", "lw": 1.2},
    )

    ax.text(0.35, 3.6, "time step 0", ha="center", va="bottom", fontsize=8.5, color="#374151")
    ax.text(1.25, 3.6, "time step 1", ha="center", va="bottom", fontsize=8.5, color="#374151")
    ax.text(2.05, 3.6, "time step 2", ha="center", va="bottom", fontsize=8.5, color="#374151")
    ax.text(2.95, 3.6, "time step 3", ha="center", va="bottom", fontsize=8.5, color="#374151")

    legend_handles = [
        Line2D([0], [0], color="#2d6a4f", lw=3, label="local two-qubit gate"),
        Line2D([0], [0], color="#d62728", lw=3, linestyle="--", label="remote gate/link"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    ax.set_title("Naive distribution: fixed placement plus one remote link per crossing gate")
    ax.set_xlim(-0.6, 3.7)
    ax.set_ylim(-1.25, 3.85)
    ax.axis("off")
    plt.tight_layout()
    plt.show()



def heavy_hex_connectivity_example() -> tuple[nx.Graph, dict[str, tuple[float, float]]]:
    edges = [
        ("q1", "q2"),
        ("q2", "q3"),
        ("q3", "q4"),
        ("q4", "q5"),
        ("q5", "q6"),
        ("q6", "q7"),
        ("q7", "q8"),
        ("q0", "q9"),
        ("q9", "q2"),
        ("q4", "q11"),
        ("q11", "q6"),
        ("q6", "q12"),
        ("q9", "q13"),
        ("q13", "q10"),
        ("q10", "q14"),
        ("q14", "q11"),
        ("q11", "q15"),
        ("q15", "q12"),
        ("q12", "q16"),
    ]
    positions = {
        "q0": (0, 2),
        "q1": (1, 3),
        "q2": (2, 2),
        "q3": (3, 3),
        "q4": (4, 2),
        "q5": (5, 3),
        "q6": (6, 2),
        "q7": (7, 3),
        "q8": (8, 2),
        "q9": (1, 1),
        "q10": (3, 1),
        "q11": (5, 1),
        "q12": (7, 1),
        "q13": (2, 0),
        "q14": (4, 0),
        "q15": (6, 0),
        "q16": (8, 0),
    }
    return nx.Graph(edges), positions


def draw_connectivity_graph(
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    *,
    title: str = "A graph: qubits are nodes, allowed two-qubit interactions are edges",
) -> None:
    plt.figure(figsize=(9, 4.6))
    nx.draw_networkx(
        graph,
        positions,
        node_size=760,
        node_color="#9ecae1",
        edge_color="0.55",
        edgecolors="#1f4e79",
        linewidths=2,
        width=2.2,
        font_size=10,
        font_weight="bold",
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def _path_edges(path: list[str]) -> set[tuple[str, str]]:
    return set(zip(path[:-1], path[1:]))


def draw_shortest_path_graph(
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    start: str,
    target: str,
) -> list[str]:
    shortest_path = nx.shortest_path(graph, start, target)
    shortest_path_edges = _path_edges(shortest_path)
    edge_colors = [
        "#d62728" if edge in shortest_path_edges or edge[::-1] in shortest_path_edges else "0.72"
        for edge in graph.edges()
    ]
    edge_widths = [
        4.5 if edge in shortest_path_edges or edge[::-1] in shortest_path_edges else 1.8
        for edge in graph.edges()
    ]
    node_colors = [
        "#fdae6b"
        if node in {start, target}
        else "#a1d99b"
        if node in shortest_path
        else "#d9f0d3"
        for node in graph.nodes()
    ]

    plt.figure(figsize=(9.5, 5.2))
    nx.draw_networkx(
        graph,
        positions,
        node_size=760,
        node_color=node_colors,
        edge_color=edge_colors,
        edgecolors="#2d6a4f",
        linewidths=2,
        width=edge_widths,
        font_size=10,
        font_weight="bold",
    )
    path_label = " -> ".join(shortest_path)
    plt.title(f"Shortest path from {start} to {target}: {path_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return shortest_path


def weighted_heavy_hex_graph() -> nx.Graph:
    weighted_edges = [
        ("q1", "q2", 1),
        ("q2", "q3", 1),
        ("q3", "q4", 1),
        ("q4", "q5", 1),
        ("q5", "q6", 1),
        ("q6", "q7", 1),
        ("q7", "q8", 1),
        ("q0", "q9", 1),
        ("q9", "q2", 1),
        ("q4", "q11", 4),
        ("q11", "q6", 1),
        ("q6", "q12", 5),
        ("q9", "q13", 3),
        ("q13", "q10", 4),
        ("q10", "q14", 2),
        ("q14", "q11", 1),
        ("q11", "q15", 1),
        ("q15", "q12", 1),
        ("q12", "q16", 1),
    ]
    graph = nx.Graph()
    graph.add_weighted_edges_from(weighted_edges)
    return graph


def draw_weighted_shortest_path_graph(
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    start: str,
    target: str,
) -> tuple[list[str], int | float]:
    weighted_path = nx.shortest_path(graph, start, target, weight="weight")
    weighted_path_edges = _path_edges(weighted_path)
    weighted_path_cost = nx.shortest_path_length(graph, start, target, weight="weight")
    edge_colors = [
        "#d62728" if edge in weighted_path_edges or edge[::-1] in weighted_path_edges else "0.72"
        for edge in graph.edges()
    ]
    edge_widths = [
        4.5 if edge in weighted_path_edges or edge[::-1] in weighted_path_edges else 1.8
        for edge in graph.edges()
    ]
    node_colors = [
        "#fdae6b"
        if node in {start, target}
        else "#a1d99b"
        if node in weighted_path
        else "#d9f0d3"
        for node in graph.nodes()
    ]

    plt.figure(figsize=(9.5, 5.2))
    nx.draw_networkx(
        graph,
        positions,
        node_size=760,
        node_color=node_colors,
        edge_color=edge_colors,
        edgecolors="#2d6a4f",
        linewidths=2,
        width=edge_widths,
        font_size=10,
        font_weight="bold",
    )
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=nx.get_edge_attributes(graph, "weight"),
        font_size=9,
        bbox={"fc": "white", "ec": "none", "alpha": 0.85},
    )
    path_label = " -> ".join(weighted_path)
    plt.title(f"Weighted shortest path from {start} to {target}: {path_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return weighted_path, weighted_path_cost


def example_hypergraph() -> tuple[list[str], list[dict[str, Any]]]:
    nodes = [f"q{i}" for i in range(8)]
    hyperedges = [
        {"name": "e0", "nodes": {"q0", "q1"}, "kind": "pair", "time": 0, "weight": 1.0},
        {"name": "e1", "nodes": {"q1", "q2"}, "kind": "pair", "time": 1, "weight": 1.0},
        {"name": "e2", "nodes": {"q4", "q5"}, "kind": "pair", "time": 1, "weight": 1.0},
        {
            "name": "e3",
            "nodes": {"q1", "q2", "q4", "q5"},
            "kind": "group",
            "time": 1,
            "weight": 0.4,
        },
        {"name": "e4", "nodes": {"q2", "q6"}, "kind": "pair", "time": 2, "weight": 1.0},
        {"name": "e5", "nodes": {"q3", "q7"}, "kind": "pair", "time": 2, "weight": 1.0},
        {
            "name": "e6",
            "nodes": {"q2", "q3", "q6", "q7"},
            "kind": "group",
            "time": 2,
            "weight": 0.4,
        },
    ]
    return nodes, hyperedges


def _convex_hull(points: np.ndarray) -> np.ndarray:
    points = sorted(map(tuple, points))
    if len(points) <= 1:
        return np.array(points)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.array(lower[:-1] + upper[:-1])


def _rounded_hypergraph_blob(
    ax: Any,
    points: np.ndarray,
    color: str,
    *,
    alpha: float = 0.24,
    pad: float = 0.42,
    linewidth: float = 2.4,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)

    if len(points) == 1:
        patch = Ellipse(
            center,
            0.9,
            0.7,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.add_patch(patch)
        return center

    if len(points) == 2:
        delta = points[1] - points[0]
        dist = np.linalg.norm(delta)
        angle = np.degrees(np.arctan2(delta[1], delta[0]))
        patch = Ellipse(
            center,
            width=dist + 0.95,
            height=0.72,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.add_patch(patch)
        return center

    hull = _convex_hull(points)
    hull_center = hull.mean(axis=0)
    expanded = hull_center + (hull - hull_center) * 1.25

    curve_points = []
    for i, point in enumerate(expanded):
        prev_point = expanded[i - 1]
        next_point = expanded[(i + 1) % len(expanded)]
        start = point + pad * (prev_point - point) / np.linalg.norm(prev_point - point)
        end = point + pad * (next_point - point) / np.linalg.norm(next_point - point)
        curve_points.append((start, point, end))

    vertices = [curve_points[0][0]]
    codes = [MplPath.MOVETO]
    for _, point, end in curve_points:
        vertices.extend([point, end])
        codes.extend([MplPath.CURVE3, MplPath.CURVE3])
    vertices.append(curve_points[0][0])
    codes.append(MplPath.CLOSEPOLY)

    patch = PathPatch(
        MplPath(vertices, codes),
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        joinstyle="round",
        capstyle="round",
    )
    ax.add_patch(patch)
    return hull_center


def draw_hypergraph(
    nodes: list[str],
    edges: list[dict[str, Any]],
    assignment: dict[str, str] | None = None,
    title: str = "Example Hypergraph",
    cut_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    positions = {
        "q0": (0, 1),
        "q1": (1, 1.6),
        "q2": (2, 1.2),
        "q3": (3, 1.8),
        "q4": (0, 0),
        "q5": (1, 0.45),
        "q6": (2, 0.15),
        "q7": (3, 0.65),
    }
    palette = {
        "pair": "#7aa6c2",
        "group": "#f2a65a",
        "communication": "#e45756",
        "two_qubit_gate": "#7aa6c2",
        "same_timestep": "#f2a65a",
    }
    module_colors = {
        "Group A": "#9ecae1",
        "Group B": "#a1d99b",
        "Group C": "#fdd0a2",
        "QPU A": "#9ecae1",
        "QPU B": "#a1d99b",
        "QPU C": "#fdd0a2",
    }

    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()

    for edge in sorted(edges, key=lambda edge: len(edge["nodes"]), reverse=True):
        edge_nodes = sorted(edge["nodes"])
        points = np.array([positions[node] for node in edge_nodes])
        color = palette.get(edge["kind"], "#756bb1")
        label_pos = _rounded_hypergraph_blob(ax, points, color=color)
        ax.text(
            label_pos[0],
            label_pos[1],
            edge["name"],
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    if cut_line is not None:
        (x0, y0), (x1, y1) = cut_line
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#444444",
            linestyle="--",
            linewidth=2.4,
            alpha=0.9,
            zorder=2.5,
        )
        ax.text(
            x1 - 0.5,
            y1 - 0.1,
            "cut",
            ha="left",
            va="top",
            fontsize=11,
            color="#444444",
            fontweight="bold",
        )

    for node in nodes:
        module = assignment.get(node) if assignment else None
        ax.scatter(
            *positions[node],
            s=900,
            color=module_colors.get(module, "#f7f7f7"),
            edgecolor="#333333",
            linewidth=1.8,
            zorder=3,
        )
        ax.text(
            *positions[node],
            node,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-0.45, 2.25)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def temporal_positions(nodes: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    return {(q, t): (t, -q) for q, t in nodes}


def draw_temporal_nodes(
    temporal_nodes: list[tuple[int, int]],
    temporal_pos: dict[tuple[int, int], tuple[int, int]],
    temporal_num_qubits: int,
    temporal_depth: int,
) -> nx.Graph:
    temporal_node_graph = nx.Graph()
    temporal_node_graph.add_nodes_from(temporal_nodes)

    plt.figure(figsize=(11.2, 4.0))
    nx.draw_networkx_nodes(
        temporal_node_graph,
        temporal_pos,
        node_size=360,
        node_color="#e8f1fa",
        edgecolors="#1f4e79",
        linewidths=1.8,
    )
    nx.draw_networkx_labels(
        temporal_node_graph,
        temporal_pos,
        labels={node: f"v{node[0]}^{node[1]}" for node in temporal_nodes},
        font_size=8,
    )

    for q in range(temporal_num_qubits):
        plt.text(-0.4, -q, f"q{q}", ha="right", va="center", fontsize=10, fontweight="bold")

    for t in range(temporal_depth):
        plt.text(t, 0.65, f"t={t}", ha="center", va="bottom", fontsize=9)

    plt.title("Temporal nodes: one node per (qubit, timestep)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return temporal_node_graph


def layer_summary_table(layers: list[list[dict[str, Any]]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": range(len(layers)),
            "ops": [
                "; ".join(format_op(op) for op in layer_ops) if layer_ops else "(idle)"
                for layer_ops in layers
            ],
        }
    )


def build_temporal_graph(
    temporal_nodes: list[tuple[int, int]],
    layers: list[list[dict[str, Any]]],
    temporal_num_qubits: int,
    temporal_depth: int,
) -> tuple[nx.Graph, list[tuple[Any, Any]], list[tuple[Any, Any]], list[dict[str, Any]]]:
    temporal_graph = nx.Graph()
    temporal_graph.add_nodes_from(temporal_nodes)

    state_edges = []
    for q in range(temporal_num_qubits):
        for t in range(temporal_depth - 1):
            u = (q, t)
            v = (q, t + 1)
            temporal_graph.add_edge(u, v, kind="state")
            state_edges.append((u, v))

    gate_edges = []
    gate_events = []
    for t, layer_ops in enumerate(layers):
        for op in layer_ops:
            qubits = op_qubits(op)
            if len(qubits) != 2:
                continue
            q_i, q_j = qubits
            u = (q_i, t)
            v = (q_j, t)
            temporal_graph.add_edge(u, v, kind="gate", gate=op["name"])
            gate_edges.append((u, v))
            gate_events.append({"t": t, "gate": op["name"], "q_i": q_i, "q_j": q_j})

    return temporal_graph, state_edges, gate_edges, gate_events


def draw_temporal_graph(
    temporal_graph: nx.Graph,
    temporal_pos: dict[tuple[int, int], tuple[int, int]],
    temporal_nodes: list[tuple[int, int]],
    state_edges: list[tuple[Any, Any]],
    gate_edges: list[tuple[Any, Any]],
) -> None:
    plt.figure(figsize=(11.6, 4.2))
    nx.draw_networkx_edges(
        temporal_graph,
        temporal_pos,
        edgelist=state_edges,
        edge_color="#3b82f6",
        width=1.8,
    )
    nx.draw_networkx_edges(
        temporal_graph,
        temporal_pos,
        edgelist=gate_edges,
        edge_color="#ef4444",
        style="--",
        width=2.6,
    )
    nx.draw_networkx_nodes(
        temporal_graph,
        temporal_pos,
        node_size=330,
        node_color="#ffffff",
        edgecolors="#1f4e79",
        linewidths=1.3,
    )
    nx.draw_networkx_labels(
        temporal_graph,
        temporal_pos,
        labels={node: f"({node[0]},{node[1]})" for node in temporal_nodes},
        font_size=8,
    )

    plt.title("Temporal graph with state edges E_s (blue) and gate edges E_g (red dashed)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def preview_temporal_hyperedges(
    normalized_circuit: Any,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    preview_graph = QuantumCircuitHyperGraph(normalized_circuit, group_gates=True)
    preview_hyperedges = []
    for h_idx, (edge_id, edge_data) in enumerate(preview_graph.hyperedges.items()):
        attrs = preview_graph.hyperedge_attrs.get(edge_id, {})
        kind = classify_disqco_hyperedge(edge_data, attrs)
        preview_hyperedges.append(
            {
                "name": attrs.get("name", f"{kind}_{h_idx}"),
                "kind": kind,
                "e_root": set(edge_data["root_set"]),
                "e_rec": set(edge_data["receiver_set"]),
            }
        )

    communication_edges = [edge for edge in preview_hyperedges if edge["kind"] == "communication"]
    gate_hyperedges = [edge for edge in preview_hyperedges if edge["kind"] == "gate"]
    visible_hyperedges = sorted(
        communication_edges,
        key=lambda edge: len(edge["e_root"] | edge["e_rec"]),
        reverse=True,
    )[:1]
    visible_hyperedges += gate_hyperedges[:2]
    return preview_graph, preview_hyperedges, visible_hyperedges


def _draw_temporal_hyperedge_blob(
    ax: Any,
    nodes: set[tuple[int, int]],
    temporal_pos: dict[tuple[int, int], tuple[int, int]],
    *,
    color: str,
    label: str,
    alpha: float = 0.22,
) -> None:
    points = np.array([temporal_pos[node] for node in sorted_temporal_nodes(nodes)], dtype=float)
    center = points.mean(axis=0)

    if len(points) == 1:
        patch = Ellipse(
            center,
            0.78,
            0.56,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=2.2,
        )
    elif len(points) == 2:
        delta = points[1] - points[0]
        dist = np.linalg.norm(delta)
        angle = np.degrees(np.arctan2(delta[1], delta[0]))
        patch = Ellipse(
            center,
            dist + 0.88,
            0.58,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=2.2,
        )
    else:
        hull = _convex_hull(points)
        hull_center = hull.mean(axis=0)
        expanded = hull_center + (hull - hull_center) * 1.24
        patch = Polygon(
            expanded,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=1.8,
            joinstyle="round",
        )
        center = hull_center

    ax.add_patch(patch)
    ax.text(
        center[0],
        center[1],
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=color,
    )


def draw_graph_to_hypergraph_bridge(
    temporal_graph: nx.Graph,
    temporal_pos: dict[tuple[int, int], tuple[int, int]],
    temporal_nodes: list[tuple[int, int]],
    state_edges: list[tuple[Any, Any]],
    gate_edges: list[tuple[Any, Any]],
    visible_hyperedges: list[dict[str, Any]],
    temporal_num_qubits: int,
    temporal_depth: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 4.8), sharey=True)
    ax_graph, ax_hypergraph = axes

    nx.draw_networkx_edges(
        temporal_graph,
        temporal_pos,
        edgelist=state_edges,
        edge_color="#3b82f6",
        width=1.6,
        alpha=0.85,
        ax=ax_graph,
    )
    nx.draw_networkx_edges(
        temporal_graph,
        temporal_pos,
        edgelist=gate_edges,
        edge_color="#ef4444",
        style="--",
        width=2.4,
        alpha=0.9,
        ax=ax_graph,
    )
    nx.draw_networkx_nodes(
        temporal_graph,
        temporal_pos,
        node_size=310,
        node_color="#ffffff",
        edgecolors="#1f4e79",
        linewidths=1.25,
        ax=ax_graph,
    )
    nx.draw_networkx_labels(
        temporal_graph,
        temporal_pos,
        labels={node: f"({node[0]},{node[1]})" for node in temporal_nodes},
        font_size=7.5,
        ax=ax_graph,
    )
    ax_graph.set_title("Temporal graph\nstate and gate relationships are pairwise")

    nx.draw_networkx_edges(
        temporal_graph,
        temporal_pos,
        edgelist=state_edges,
        edge_color="#bfdbfe",
        width=1.2,
        alpha=0.65,
        ax=ax_hypergraph,
    )

    hyperedge_colors = {"communication": "#e45756", "gate": "#7aa6c2"}
    for edge in sorted(
        visible_hyperedges,
        key=lambda item: len(item["e_root"] | item["e_rec"]),
        reverse=True,
    ):
        kind = edge["kind"]
        nodes = edge["e_root"] | edge["e_rec"]
        label = (
            "one\ncommunication\nepisode"
            if kind == "communication"
            else "single gate\nhyperedge"
        )
        _draw_temporal_hyperedge_blob(
            ax_hypergraph,
            nodes,
            temporal_pos,
            color=hyperedge_colors[kind],
            label=label,
            alpha=0.14 if kind == "communication" else 0.18,
        )

    root_nodes = set().union(*(edge["e_root"] for edge in visible_hyperedges))
    receiver_nodes = set().union(*(edge["e_rec"] for edge in visible_hyperedges))
    node_colors = []
    for node in temporal_nodes:
        if node in root_nodes and node in receiver_nodes:
            node_colors.append("#bcbddc")
        elif node in root_nodes:
            node_colors.append("#fdae6b")
        elif node in receiver_nodes:
            node_colors.append("#a1d99b")
        else:
            node_colors.append("#ffffff")

    nx.draw_networkx_nodes(
        temporal_graph,
        temporal_pos,
        node_size=310,
        node_color=node_colors,
        edgecolors="#1f4e79",
        linewidths=1.25,
        ax=ax_hypergraph,
    )
    nx.draw_networkx_labels(
        temporal_graph,
        temporal_pos,
        labels={node: f"({node[0]},{node[1]})" for node in temporal_nodes},
        font_size=7.5,
        ax=ax_hypergraph,
    )
    ax_hypergraph.set_title("Temporal hypergraph\ncompatible relationships become hyperedges")

    for ax in axes:
        for q in range(temporal_num_qubits):
            ax.text(
                -0.45,
                -q,
                f"q{q}",
                ha="right",
                va="center",
                fontsize=9.5,
                fontweight="bold",
            )
        for t in range(temporal_depth):
            ax.text(t, 0.7, f"t={t}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xlim(-0.7, temporal_depth - 0.15)
        ax.set_ylim(-temporal_num_qubits + 0.35, 0.95)
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color="#3b82f6", lw=2, label="state edge / continuity"),
        Line2D(
            [0],
            [0],
            color="#ef4444",
            lw=2.4,
            linestyle="--",
            label="two-qubit gate edge",
        ),
        Line2D(
            [0],
            [0],
            color="#e45756",
            lw=8,
            alpha=0.14,
            label="grouped communication hyperedge",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#fdae6b",
            markeredgecolor="#1f4e79",
            markersize=9,
            label="root nodes",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#a1d99b",
            markeredgecolor="#1f4e79",
            markersize=9,
            label="receiver nodes",
        ),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(
        "Graph to hypergraph: pairwise edges are reorganized into communication objects",
        y=1.02,
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()
