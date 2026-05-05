import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from pathlib import Path
from itertools import islice
import qiskit
import qiskit.qasm2
import qiskit_ibm_runtime
import qiskit_aer
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR / "transpiled-circuits"
_BACKEND = None
_ERROR_TABLE = None


def load_qasm(filepath):
    circuit = qiskit.qasm2.load(
        filepath,
        custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    return circuit


def circuit_count(filepath):
    m = re.search(r"-(\d{4})\.qasm$", filepath.name)
    if m:
        return int(m.group(1))


def instruction_fidelity(instruction, circuit, backend):
    name = instruction.operation.name
    if name == 'reset':
        return 1
        
    op = backend.target[name]
    qubit_inds = tuple(circuit.qubits.index(q) for q in instruction.qubits)
    return 1 - op[qubit_inds].error
    

def log_circuit_fidelity(circuit, backend=_BACKEND):
    log_fidelity = lambda inst: np.log(instruction_fidelity(inst, circuit, backend))
    return sum(log_fidelity(inst) for inst in circuit)


def circuit_metrics(circuit: qiskit.QuantumCircuit, backend=_BACKEND):    
    metrics = {
        'num_qubits': circuit.num_qubits,
        'num_clbits': circuit.num_clbits,
        'num_parameters': circuit.num_parameters,
        'size': circuit.size(),
        'depth': circuit.depth(),
        'duration_s': circuit.estimate_duration(target=backend.target, unit='s'),
        'duration_dt': circuit.estimate_duration(target=backend.target, unit='dt'),
        'log_fidelity': log_circuit_fidelity(circuit, backend=backend),
    }
    metrics.update(circuit.count_ops())
    return metrics


def build_gate_error_table(backend):
    target = backend.target
    table = {}  # (name, qubit_tuple) -> error

    for op, qtuple in target.instructions:
        if qtuple is not None:
            props = target[op.name].get(qtuple)
            err = getattr(props, "error", None)
            if err is None:
                continue
            table[(op.name, qtuple)] = err
            
    while np.maximum(table.values() == 1.0):
        # find a gate with an error rate of 1
        bad_gate = next(gate for gate, err in table.items() if err == 1.0)
        
        # get the average of all errors for that gate < 1
        avg_log_err = np.mean(
            np.log(err) for gate, err in table.items() if gate[0] == bad_gate[0]
        )
        
        # replace all bad values with the fixed value
        for gate in [gate for gate, err in table.items() if err == 1.0]:
            table[gate] = np.exp(avg_log_err)
        
        # repeat until there are no more bad gates

    return table


def instruction_fidelity_fast(inst, qindex, error_table):
    name = inst.operation.name
    qtuple = tuple(qindex[q] for q in inst.qubits)
    return 1.0 - error_table.get((name, qtuple), 0.0)


def log_circuit_fidelity_fast(circuit, qindex, error_table):
    return sum(
        np.log(instruction_fidelity_fast(inst, qindex, error_table)) for inst in circuit
    )


def circuit_metrics_fast(
    circuit: qiskit.QuantumCircuit, 
    qindex, 
    error_table,
    backend=_BACKEND,
):    
    metrics = {
        'num_qubits': circuit.num_qubits,
        'num_clbits': circuit.num_clbits,
        'num_parameters': circuit.num_parameters,
        'size': circuit.size(),
        'depth': circuit.depth(),
        'duration_s': circuit.estimate_duration(target=backend.target, unit='s'),
        'duration_dt': circuit.estimate_duration(target=backend.target, unit='dt'),
        'log_fidelity': log_circuit_fidelity_fast(circuit, qindex, error_table),
    }
    metrics.update(circuit.count_ops())
    return metrics


def init_worker():
    global _BACKEND, _ERROR_TABLE
    _BACKEND = qiskit_ibm_runtime.fake_provider.FakeSherbrooke()
    _ERROR_TABLE = build_gate_error_table(_BACKEND)


def process_file(filepath):
    data = {
        'filename': filepath.name,
        'count': circuit_count(filepath)
    }
    circuit = load_qasm(filepath)
    qindex = {q: i for i, q in enumerate(circuit.qubits)}
    metrics = circuit_metrics_fast(circuit, qindex, _ERROR_TABLE, backend=_BACKEND)
    data.update(metrics)
    return data


def main():
    print("Initializing...")
    qasm_files = list(DATASET_DIR.rglob("*.qasm"))
    
    with ProcessPoolExecutor(initializer=init_worker) as ex:
        data = list(tqdm(
            ex.map(process_file, qasm_files, chunksize=16),
            total=len(qasm_files),
            desc="Processing circuits",
        ))

    pd.DataFrame(data).to_csv(SCRIPT_DIR / "data.csv", index=False)
    
    return None
    
if __name__ == '__main__':
    main()