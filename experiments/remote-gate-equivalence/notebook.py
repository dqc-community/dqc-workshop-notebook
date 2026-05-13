import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import itertools
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity, random_statevector

    return (
        DensityMatrix,
        QuantumCircuit,
        Statevector,
        itertools,
        mo,
        np,
        partial_trace,
        state_fidelity,
    )


@app.cell
def _(QuantumCircuit):
    remote_circuit = QuantumCircuit(2, name="remote")
    remote_circuit.h(0)
    remote_circuit.x(1)
    remote_circuit.z(0)
    remote_circuit.z(1)
    remote_circuit.cx(0, 1)
    remote_circuit.draw()
    return (remote_circuit,)


@app.cell
def _(QuantumCircuit, Statevector, remote_circuit):
    psi_minus = QuantumCircuit(2)
    psi_minus.append(remote_circuit.to_gate(), [0, 1])
    Statevector.from_circuit(psi_minus).to_dict()
    return


@app.cell
def _(QuantumCircuit, remote_circuit):
    pre = QuantumCircuit(4)
    pre.append(remote_circuit.to_gate(), [1, 3])
    pre.cx(0, 1)
    pre.draw()
    return (pre,)


@app.cell
def _(Statevector, pre):
    psi_pre = Statevector.from_circuit(pre)
    psi_pre.to_dict()
    return (psi_pre,)


@app.cell
def _(psi_pre):
    psi_pre_probs = psi_pre.probabilities([1])
    psi_pre_probs
    return (psi_pre_probs,)


@app.cell
def _(psi_pre):
    psi_pre.copy().measure([1])
    return


@app.cell
def _(psi_pre, psi_pre_probs):
    branches = {}
    while len(branches) < sum([p > 0 for p in psi_pre_probs]):
        _outcome, _statevector =  psi_pre.copy().measure([1])
        branches[_outcome] = _statevector
    branches
    return (branches,)


@app.cell
def _(branches):
    {k: v.to_dict() for k, v in branches.items()}
    return


@app.cell
def _(QuantumCircuit, mo, np):
    post = {k: QuantumCircuit(4) for k in ['0', '1']}
    post['0'].cx(3, 2)
    post['1'].rx(np.pi, 3)
    post['1'].cx(3, 2)
    mo.vstack([v.draw() for v in post.values()])
    return (post,)


@app.cell
def _(branches, post):
    for _outcome in ['0', '1']:
        branches[_outcome].evolve(post[_outcome])

    {k: v.to_dict() for k, v in branches.items()}
    return


@app.cell
def _(DensityMatrix, branches, partial_trace):
    rho = {k: partial_trace(DensityMatrix(v), [1, 3]) for k, v in branches.items()}
    rho
    return


@app.cell
def _(QuantumCircuit, Statevector):
    def embed_state(psi_pre, qubits, n):
        psi_post = Statevector.from_label('0' * n)
        circ = QuantumCircuit(n)
        circ.initialize(psi_pre.data, qubits)
        return psi_post.evolve(circ)

    return (embed_state,)


@app.cell
def _(DensityMatrix, QuantumCircuit, partial_trace):
    def local_cx(psi):
        circ = QuantumCircuit(4)
        circ.cx(0, 2)
        rho = DensityMatrix(psi.evolve(circ))
        return partial_trace(rho, [1, 3])    

    return (local_cx,)


@app.cell
def _(DensityMatrix, QuantumCircuit, Statevector, np, partial_trace):
    def remote_cx(psi):
        # Define remote circuit
        remote_circuit = QuantumCircuit(2)
        remote_circuit.h(0)
        remote_circuit.x(1)
        remote_circuit.z(0)
        remote_circuit.z(1)
        remote_circuit.cx(0, 1)

        # Set up remote gate up to measurement
        pre = QuantumCircuit(4)
        pre.append(remote_circuit.to_gate(), [1, 3])
        pre.cx(0, 1)
        psi_pre = Statevector.from_circuit(pre)

        # Branch on measurement outcomes
        branches = {}
        while len(branches) < sum([p > 0 for p in psi_pre.probabilities([1])]):
            outcome, statevector =  psi_pre.copy().measure([1])
            branches[outcome] = statevector

        # Continue circuit on both branches
        post = {k: QuantumCircuit(4) for k in ['0', '1']}
        post['0'].cx(3, 2)
        post['1'].rx(np.pi, 3)
        post['1'].cx(3, 2)
        for outcome, statevector in branches.items():
            branches[outcome] = statevector.evolve(post[outcome])

        rho = {outcome: partial_trace(DensityMatrix(statevector), [1, 3]) for outcome, statevector in branches.items()}
        return rho

    return (remote_cx,)


@app.cell
def _(Statevector, embed_state, labels, local_cx):
    def _testlab(label):
        psi02 = Statevector.from_label(label)
        psi4 = embed_state(psi02, [0, 2], 4) 
        return local_cx(psi4).to_dict()

    {lab: _testlab(lab) for lab in labels}
    return


@app.cell
def _(embed_state, local_cx, remote_cx, state_fidelity):
    def compare_remote_local(psi):
        psi4 = embed_state(psi, [0, 2], 4)
        rho_local = local_cx(psi4)
        rho_remote = remote_cx(psi4)
        return {outcome: state_fidelity(rho, rho_local) for outcome, rho in rho_remote.items()}

    return (compare_remote_local,)


@app.cell
def _(Statevector, compare_remote_local):
    _label = '1+'
    _psi = Statevector.from_label(_label)
    compare_remote_local(_psi)
    return


@app.cell
def _(itertools):
    _lablist = '01'
    labels = [''.join(p) for p in itertools.product(_lablist, _lablist)]
    labels
    return (labels,)


@app.cell
def _(Statevector, compare_remote_local, labels):
    results = {lab: compare_remote_local(Statevector.from_label(lab)) for lab in labels}
    results
    return


if __name__ == "__main__":
    app.run()
