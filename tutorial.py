# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "ipykernel",
#     "marimo",
#     "pandas",
#     "matplotlib",
#     "networkx",
#     "scikit-learn",
#     "qiskit>=1.0",
#     "qiskit-aer",
#     "qiskit-ibm-runtime",
#     "bosonic-sdk[disqco,hypergraph]",
# ]
# 
# [[tool.uv.index]]
# name = "test-pypi"
# url = "https://test.pypi.org/simple/"
# default = false
# 
# [[tool.uv.index]]
# name = "pypi"
# url = "https://pypi.org/simple/"
# default = true
# 
# [tool.uv.sources]
# bosonic-sdk = { index = "test-pypi" }
# 
# [tool.uv]
# index-strategy = "unsafe-best-match"
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    # Setup Cell: import all dependencies and define global constants

    import marimo as mo

    # standard/numerical
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd

    # qiskit
    # we avoid the pattern `from qiskit.submodule import Class` to make it clear in each cell where each function/class comes from
    import qiskit
    import qiskit.visualization
    import qiskit_ibm_runtime
    import qiskit_aer
    from qiskit.exceptions import MissingOptionalLibraryError

    # bosonic
    import bosonic_sdk
    from bosonic_converters import CircuitConverters # sufficiently self-explanatory

    # filter warnings
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
    )

    # config constants
    VERIFY_CFG = {
        'N_LIST': range(3, 21), # circuit sizes to sweep over
        'SHOTS': 2048, # number of times to simulate each circuit
        'SEED': 1234, # RNG seed to ensure reproducibility
    }

    SCALING_CFG = {
        'N_LIST': range(3, 128),
        'QUBITS_PER_TRAP': 32,
    }

    TTS_CFG = {
        'SHOTS': 1024,
        'N_LIST': range(5, 126, 5),
        'QUBITS_PER_TRAP': 128,
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Introduction


    The goal of the notebook is to explore how distributed quantum computing approaches compare to their monolithic counterparts. To keep scaling behavior consistent, we choose an easy scalable circuit family: GHZ states.

    The notebook is split into three main sections:

    1. **Generating GHZ circuits** – We use GHZ circuits because they are simple and easy to follow. This simplicity (one more qubit = one more CNOT gate) lets us focus on differences between monolithic and distributed approaches without having to worry about other effects as we scale.
    2. **Validation** – We validate correctness by sampling GHZ circuits and checking expected outputs using two simulators:
       - Qiskit Aer (monolithic simulator)
       - Bosonic simulator (distributed simulator)
    3. **Exploration** – After validation, we study scaling from two perspectives:
       - Circuit metrics: depth, gate counts, and related compiled-circuit characteristics
       - Hardware metrics: an execution-time model that includes practical hardware considerations

    Finally, we will use the following notation:

    - *$n$*: number of qubits
    - "remote" or "cross-module" gates: two-qubit interactions whose endpoints lie in different modules (two-qubit gates between qubits in different modules).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # GHZ Circuit Generator

    Now that we have installed all dependencies and given an overview of the notebook content, we can start building our test circuit. A Greenberger-Horne-Zeilinger (GHZ) state on $n$ qubits is defined as:

    $$
    \lvert \mathrm{GHZ}_n \rangle
    =
    \frac{1}{\sqrt{2}}
    \left(
    \lvert 00\ldots 0_n \rangle
    +
    \lvert 11\ldots 1_n \rangle
    \right)
    =
    \frac{1}{\sqrt{2}}
    \left(
    \lvert 0 \rangle^{\otimes n}
    +
    \lvert 1 \rangle^{\otimes n}
    \right).
    $$

    After measuring all qubits in the computational basis, the ideal outcome distribution is:

    $$
    \Pr(x) = \begin{cases}
    \frac{1}{2}, & x = 0^n \text{ or } 1^n \\
    0, & \text{otherwise}
    \end{cases}
    $$

    The circuit that generates this state is simple and involves two steps:

    1. Apply $H$ to qubit $0$.
    2. Apply $\mathrm{CX}(0 \to i)$ for $i = 1, \dots, n-1$.
    """)
    return


@app.function
def ghz_circuit(n: int, measure: bool = True) -> qiskit.QuantumCircuit:
    """Create a GHZ state circuit on n qubits."""
    qc = qiskit.QuantumCircuit(n, n)
    # qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    if measure:
        qc.measure(range(n), range(n))
    return qc


@app.cell
def _():
    ghz_circuit(3, measure=False).draw("mpl")
    return


@app.cell
def _():
    ghz_circuit(5, measure=False).draw("mpl")
    return


@app.cell
def _():
    ghz_circuit(8, measure=False).draw("mpl")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Validating the Circuits

    Now that we have defined our circuit, the next step is to verify their behavior by confirming that simulations return the expected measurement distribution.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The simulators return measurement `counts` over `SHOTS` samples. For an ideal GHZ state, we expect roughly an equal amount of $0^n$ and $1^n$ measurements:

    $$
    \texttt{counts}[0^n] \approx \frac{\texttt{SHOTS}}{2},
    \qquad
    \texttt{counts}[1^n] \approx \frac{\texttt{SHOTS}}{2},
    $$

    and all other bitstrings should never be measured.

    We use these counts to define a simple fidelity proxy to ensure that we are measuring what we expect:

    $$
    \widehat{F}_{\mathrm{GHZ}}(n)
    =
    \frac{\texttt{counts}[0^n] + \texttt{counts}[1^n]}{\texttt{SHOTS}}.
    $$

    - $\widehat{F}_{\mathrm{GHZ}}(n) = 1$ means that we measured nothing but the expected bitstrings and our simulators are working as intended.
    - $\widehat{F}_{\mathrm{GHZ}}(n) < 1$ means that we measured at least one incorrect bitstring. Without any noise models this means that our simulators are not working as expected.

    In the following section, for each circuit size $n$, we:

    - Simulate the circuit
    - Collect measurement counts
    - Compute and plot the proxy scores

    We start by defining the parameters of our circuit sweep:
    """)
    return


@app.cell
def _():
    VERIFY_CFG
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Monolithic Circuit Simulation

    To simulate monolithic circuits, we will use Qiskit's Aer simulator transpiled to IBM's [`FakeSherbrooke`](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider-fake-sherbrooke) backend, a (non-existent) 127-qubit QPU.
    """)
    return


@app.cell
def load_sherbrooke_1():
    FAKE_IBM_BACKEND = qiskit_ibm_runtime.fake_provider.FakeSherbrooke()
    try:
        qiskit.visualization.plot_gate_map(FAKE_IBM_BACKEND)
    except MissingOptionalLibraryError:
        _graph = nx.Graph()
        _graph.add_nodes_from(range(FAKE_IBM_BACKEND.num_qubits))
        _graph.add_edges_from(FAKE_IBM_BACKEND.coupling_map.get_edges())
        _pos = nx.spring_layout(_graph, seed=VERIFY_CFG['SEED'])
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_edges(_graph, _pos, alpha=0.25, width=0.8)
        nx.draw_networkx_nodes(_graph, _pos, node_size=55)
        plt.title('FakeSherbrooke Coupling Map')
        plt.axis('off')
        plt.show()
    return (FAKE_IBM_BACKEND,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice how the qubits are laid out. All of the CNOT gates involved in our GHZ circuit can't happen directly because each qubit is touching at most three neighboring qubits. This means lots of SWAP gates, with the number of SWAPs increasing as more distant qubits need to be entangled.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's start with the basics of running a circuit through the IBM simulator. First, we'll define a GHZ circuit with only $n=3$ qubits:
    """)
    return


@app.cell
def _():
    ibm_circ3 = ghz_circuit(3)
    ibm_circ3.draw('mpl')
    return (ibm_circ3,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This is called the *logical circuit* and has only a few gate layers.
    """)
    return


@app.cell
def _(ibm_circ3):
    ibm_circ3.depth()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, we *transpile* the circuit to the hardware backend, replacing the logical gates with equivalent operations that are allowed on the hardware, restricted by:

    1. **Native gate set** – set of unitaries the hardware can physically execute
    2. **Connectivity** – entangling gates can only be performed on qubits that are adjacent in the hardware gate map
    """)
    return


@app.cell
def _(FAKE_IBM_BACKEND, ibm_circ3):
    ibm_transpiled3 = qiskit.transpile(ibm_circ3, backend=FAKE_IBM_BACKEND)
    ibm_transpiled3.draw('mpl')
    return (ibm_transpiled3,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice how we no longer have H and CNOT gates; they've been replaced with the gates that can actually be performed by the hardware, and you'll notice that the first and third qubits are never directly entangled (because they aren't physically connected on the QPU). Transpilation will typically increase the gate depth because logical gates have to be broken down into multiple physical gates to actually run on hardware.
    """)
    return


@app.cell
def _(ibm_transpiled3):
    ibm_transpiled3.depth()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we have a circuit that can run on real (fake?) hardware, let's simulate it!
    """)
    return


@app.cell
def _(ibm_transpiled3):
    ibm_result3 = qiskit_aer.AerSimulator().run(ibm_transpiled3, shots=VERIFY_CFG['SHOTS']).result()
    ibm_result3.to_dict()
    return (ibm_result3,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You can see there's a lot of information, but we only care about the measurement counts, which we can access with the `get_counts()` method:
    """)
    return


@app.cell
def _(ibm_result3):
    ibm_result3.get_counts()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This is what we expected: only the outcomes `000` and `111` are measured, meaning we (most likely) successfully created a GHZ state!

    Let's go ahead and compute the proxy measure we came up with earlier:

    $$
    \widehat{F}_{\mathrm{GHZ}}(n)
    =
    \frac{\texttt{counts}[0^n] + \texttt{counts}[1^n]}{\texttt{SHOTS}}.
    $$
    """)
    return


@app.function
def ghz_fidelity_proxy(counts):
    n = min(len(k) for k in counts.keys()) # compute n using length of measured bitstrings
    shots = sum(counts.values()) # sum measurement counts over all bitstrings
    return (counts.get('0' * n, 0) + counts.get('1' * n, 0)) / shots


@app.cell
def _(ibm_result3):
    ghz_fidelity_proxy(ibm_result3.get_counts())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we know how the simulator behaves and what data we need, let's write a function to handle the other cases we want to study:
    """)
    return


@app.cell
def _(FAKE_IBM_BACKEND):
    def verify_ghz_ibm(n, shots=VERIFY_CFG['SHOTS']):
        circ = qiskit.transpile(ghz_circuit(n), backend=FAKE_IBM_BACKEND)
        counts = qiskit_aer.AerSimulator().run(circ, shots=shots).result().get_counts()
        data = {
            'backend': 'IBM',
            'n': n,
            'depth': circ.depth(),
            'count0': counts.get('0' * n, 0),
            'count1': counts.get('1' * n, 0),
        }
        return data

    return (verify_ghz_ibm,)


@app.cell
def _(verify_ghz_ibm):
    ibm_data = [verify_ghz_ibm(n) for n in VERIFY_CFG['N_LIST']]
    pd.DataFrame(ibm_data)
    return (ibm_data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Distributed Circuit Simulation

    Next, we simulate the same circuit distributed over multiple modules using the [Bosonic SDK]([https://](https://github.com/dqc-community/dqcomp)):
    """)
    return


@app.function
# TODO: explain distributor.distribute API (or write docs upstream and show them here)
def compile_bosonic_circuit(circuit, n, modules, distributor):
    """Distribute a Qiskit circuit across Bosonic modules and return a Qiskit circuit."""
    distributed = distributor.distribute(
        CircuitConverters.from_qiskit(circuit),
        nodes=int(modules),
        qubits_per_node=np.ceil(n / modules).astype(int),
        lowered=True,
    ).as_monolithic_circuit()
    return CircuitConverters.to_qiskit(distributed)


@app.cell
def _():
    compile_bosonic_circuit(
        ghz_circuit(3), 3, 2, bosonic_sdk.BosonicDistributor()
    ).depth()
    return


@app.function
def verify_ghz_bosonic(n, shots=VERIFY_CFG['SHOTS'], traps=2):
    distributor = bosonic_sdk.BosonicDistributor()
    circuit = compile_bosonic_circuit(ghz_circuit(n), n, traps, distributor)
    counts, _ = bosonic_sdk.Simulator().run_counts(circuit, ignore_c_remote=True, shots=shots)
    data = {
        'backend': 'Bosonic',
        'n': n,
        'depth': circuit.depth(),
        'count0': counts.get('0' * n, 0),
        'count1': counts.get('1' * n, 0),
    }
    return data


@app.cell
def _():
    bosonic_data = [verify_ghz_bosonic(n) for n in VERIFY_CFG['N_LIST']]
    pd.DataFrame(bosonic_data)
    return (bosonic_data,)


@app.cell
def _(bosonic_data, ibm_data):
    verify_df = pd.DataFrame(ibm_data + bosonic_data)
    verify_df
    return (verify_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analysis and Comparison

    Let's first verify that our circuits have produced the expected measurement distribution.

    First, we want to check that we aren't measuring any bitstrings other than $0^n$ and $1^n$, so we can verify:

    $$
    \frac{
        \texttt{counts}\left[ 0^n \right] + \texttt{counts}\left[ 1^n \right]
    }{
        \texttt{SHOTS}
    } = 1
    $$
    """)
    return


@app.cell
def _(verify_df):
    verify_df['fidelity'] = (verify_df['count0'] + verify_df['count1']) / VERIFY_CFG['SHOTS']
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Also, we want to make sure that both outcomes are being measured with equal probability.

    $$
    \left\vert
    \frac{
        \texttt{counts}\left[ 0^n \right] - \texttt{counts}\left[ 1^n \right]
    }{
        \texttt{SHOTS}
    } \right\vert \approx 0
    $$
    """)
    return


@app.cell
def _(verify_df):
    verify_df['dispersion'] = np.abs(verify_df['count0'] - verify_df['count1']) / VERIFY_CFG['SHOTS']
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we just check that:

    1. `fidelity` is never below 1
    2. `dispersion` is never too far above 0
    """)
    return


@app.cell
def _(verify_df):
    verify_df.groupby('backend').agg(
        min_fidelity=('fidelity', 'min'),
        max_dispersion=('dispersion', 'max'),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Both checks pass! To get a better sense of what the data look like, let's plot the sampled measurement probabilities in 2D space:
    """)
    return


@app.cell
def _(verify_df):
    for _backend in ['IBM', 'Bosonic']:
        subdf = verify_df[verify_df['backend'] == _backend]
        plt.scatter(
            subdf['count0'] / VERIFY_CFG['SHOTS'], 
            subdf['count1'] / VERIFY_CFG['SHOTS'], 
            label=_backend,
            alpha=0.5,
        )

    plt.xlabel(r'$\Pr(0^n)$')
    plt.xlim(0.45, 0.55)

    plt.ylabel(r'$\Pr(1^n)$')
    plt.ylim(0.45, 0.55)

    plt.title('Measurement Results')
    plt.legend(title='Backend')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice that all measurement probabilities:

    1. Fall on the line where $\Pr(0^n) + \Pr(1^n) = 1$
    2. Are close to $\Pr(0^n) = \Pr(1^n) = 0.5$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Scaling Behavior

    Now that we have verified that our simulators are behaving as expected, we can explore how these circuits scale in each system.

    For now we will focus on the following metrics (but you can look at other metrics later):

    - `depth`: longest sequential gate path (time-step count)
    - `two_qubit_depth`: longest chain of explicit two-qubit-operation layers
    - `two_qubit_count`: total number of two-qubit operations
    - `total_ops`: total operation count

    For each backend and circuit size $n$, we:

    1. Build the logical GHZ circuit
    2. Compile and measure monolithic metrics
    3. Compile and measure distributed metrics with dynamic module count $k = \lceil \frac{n}{20} \rceil$
    4. Plot each metric against $n$ to compare growth trends

    To keep the comparison fair, we use a single metric engine ([`GateStatistics.stats`](https://github.com/dqc-community/dqcomp/blob/5e72e369cc81efcc279ceb63468a10659f01a872/packages/bosonic-sdk/bosonic_sdk/gate_statistics.py#L157-L163)) for both paths so metric definitions stay consistent.
    """)
    return


@app.cell
def _():
    SCALING_CFG
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Defining Metrics

    We want to include all of the data we can get. First, let's see what kind of information gets collected in `bosonic_sdk.GateStatistics.stats`:
    """)
    return


@app.cell
def _():
    bosonic_sdk.GateStatistics.stats(
        CircuitConverters.from_qiskit(ghz_circuit(5))
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This tells us plenty about local vs. remote operations, but it doesn't provide granular information about which types of local gates were executed, etc. We can collect this directly from the Qiskit circuit representation:
    """)
    return


@app.function
def qiskit_metrics(circuit):
    count_ops = lambda x: sum(inst.operation.name == x for inst in circuit.data)
    gate_ops = [inst for inst in circuit.data if inst.operation.name not in ['measure', 'reset', 'barrier']]
    data = {
        'measure_count': count_ops('measure'),
        'reset_count': count_ops('reset'),
        'barrier_count': count_ops('barrier'),
        'single_qubit_count': sum(len(inst.qubits) == 1 for inst in gate_ops),
        'two_qubit_count': sum(len(inst.qubits) == 2 for inst in gate_ops),
        'multi_qubit_count': sum(len(inst.qubits) > 2 for inst in gate_ops),
    }
    return data


@app.function
def circuit_metrics(circuit):
    converted = CircuitConverters.from_qiskit(circuit)
    data = bosonic_sdk.GateStatistics.stats(converted)
    data.update(qiskit_metrics(circuit))
    return data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Collecting Data

    As before, we set up a pipeline for each backend where we:

    1. Construct the logical circuit
    2. Transpile to the native gate set of the hardware
    3. Collect metrics about the circuit
    """)
    return


@app.cell
def _(FAKE_IBM_BACKEND):
    def scale_ibm(n, constructor=ghz_circuit, backend=FAKE_IBM_BACKEND, **transpile_kwargs):
        circuit = qiskit.transpile(constructor(n), backend=FAKE_IBM_BACKEND, **transpile_kwargs)
        data = {'backend': 'IBM', 'n': n, 'k': 1, 'circuit': circuit}
        return data

    return (scale_ibm,)


@app.function
def scale_bosonic(
    n,
    constructor=ghz_circuit,
    qubits_per_trap=SCALING_CFG['QUBITS_PER_TRAP'],
    distributor=bosonic_sdk.BosonicDistributor(),
):
    k = np.ceil(n / qubits_per_trap).astype(int)
    circuit = compile_bosonic_circuit(constructor(n), n, k, distributor)
    data = {'backend': 'Bosonic', 'n': n, 'k': k, 'circuit': circuit}
    return data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we loop over every value of $n$ we defined in our config, compile a circuit, and collect the circuit metrics into a `DataFrame`.
    """)
    return


@app.cell
def _(scale_ibm):
    if mo.running_in_notebook():
        _iter = mo.status.progress_bar(
            SCALING_CFG['N_LIST'],
            title='Compiling',
            subtitle='Monolithic IBM backend',
        )
    else:
        _iter = SCALING_CFG['N_LIST']

    ibm_circuits = [scale_ibm(n, optimization_level=3) for n in _iter]
    return (ibm_circuits,)


@app.cell
def _():
    if mo.running_in_notebook():
        _iter = mo.status.progress_bar(
            SCALING_CFG['N_LIST'],
            title='Compiling',
            subtitle='Distributed Bosonic backend',
        )
    else:
        _iter = SCALING_CFG['N_LIST']

    bosonic_circuits = [scale_bosonic(n) for n in _iter]
    return (bosonic_circuits,)


@app.cell
def _(bosonic_circuits, ibm_circuits):
    circuit_df = pd.DataFrame(ibm_circuits + bosonic_circuits)
    circuit_df.loc[:, circuit_df.columns != 'circuit'] # including circuit column breaks marimo display
    return (circuit_df,)


@app.cell
def _(circuit_df):
    _metrics = lambda g: pd.Series(circuit_metrics(g['circuit']))
    scaling_df = circuit_df.join(circuit_df.apply(_metrics, axis=1))
    scaling_df.loc[:, scaling_df.columns != 'circuit']
    return (scaling_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploring Results

    That's a lot of columns, so let's just remind ourselves quickly of what all the variables are:
    """)
    return


@app.cell
def _(scaling_df):
    list(scaling_df.columns)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's define a function that plots a metric against the circuit size $n$ and test it for a few different metrics:
    """)
    return


@app.function
def plot_scaling_metric(df, metric, **kwargs):
    for _backend in ['IBM', 'Bosonic']:
        subdf = df[df['backend'] == _backend]
        plt.plot(subdf['n'], subdf[metric], label=_backend)

    plt.title(kwargs.get('title', 'Scaling Behavior'))

    plt.xlabel('Number of Qubits')
    plt.xscale(kwargs.get('xscale', 'linear'))
    plt.ylabel(kwargs.get('ylabel', metric))
    plt.yscale(kwargs.get('yscale', 'linear'))
    plt.legend(title='Backend')

    plt.tight_layout()
    plt.show()


@app.cell
def _(scaling_df):
    plot_scaling_metric(
        scaling_df, 'depth', title='GHZ Scaling', ylabel='Circuit Depth'
    )
    return


@app.cell
def _(scaling_df):
    plot_scaling_metric(
        scaling_df, 'two_qubit_count', title='GHZ Scaling', ylabel='Two-Qubit Gates'
    )
    return


@app.cell
def _(scaling_df):
    plot_scaling_metric(
        scaling_df, 'single_qubit_count', title='GHZ Scaling', ylabel='Single-Qubit Gates'
    )
    return


@app.cell
def _(scaling_df):
    plot_scaling_metric(
        scaling_df, 'qubit_teleportation_count', title='GHZ Scaling', ylabel='Teleportation Gates'
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interpreting Results

    Quantum computation is expensive and error-prone, so we generally want to do as little of it as we can get away with. This means we want circuits with fewer operations and shallower paths.

    Two-qubit gates are especially important because they are typically slower and noisier (lower fidelity) than single-qubit gates, which is why we tracked `two_qubit_gates` as a metric.

    Connectivity plays a huge role in how many additional gates your circuit needs as it scales. In monolithic nearest-neighbor hardware, non-local interactions require routing through SWAP operations. This routing overhead increases depth and total operation count as circuits grows. In the distributed approach we still have to worry about connectivity, but due to the modularity of the hardware communication costs grow linearly with circuit size (i.e., at the same rate as the logical circuit) rather than exponentially.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Execution Time

    So far we have seen how scaling affects the circuit we are implementing. This is useful for building intuition, but at the end of the day we care about the time-to-solution (TTS) when executing on hardware. Accessing real hardware involves provider cost, queue times, and cloud submission overhead. That cost is worth it when we care about running an application circuit, but here we only care about the scaling behavior. To simplify things, we construct a worst-case model for time-to-solution that incorporates the metrics we care about. First, we define the relevant parameters:
    """)
    return


@app.cell
def _():
    TTS_CFG
    return


@app.cell
def _():
    device_df = pd.DataFrame(
        [
            {
                "backend": "IBM",
                "t1q": 2e-8,
                "t2q": 2e-7,
                "t_meas": 1e-6,
                "t_overhead": 2e-4,
                "e1q": 5e-4,
                "e2q": 3e-3,
            },
            {
                "backend": "Bosonic",
                "t1q": 1e-6,
                "t2q": 3e-5,
                "t_meas": 4e-4,
                "t_overhead": 1e-3,
                "e1q": 1e-6,
                "e2q": 1e-4,
            },
        ]
    )
    device_df
    return (device_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can merge this device data directly into our experiment data to compute execution times at each row.
    """)
    return


@app.cell
def _(device_df, scaling_df):
    _merged = scaling_df.merge(device_df, on='backend')
    _merged.loc[:, _merged.columns != 'circuit']
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Building the Model

    To begin, we treat a compiled circuit as a collection of single-qubit gates and two-qubit gates. Different gate types can have different execution times, but for a simpler model we assume an average single-qubit gate time $t_{1q}$ and an average two-qubit gate time $t_{2q}$. If we further assume that all gates are executed sequentially (a strong assumption, but it provides an upper bound), one shot of the circuit takes:

    $$
    T_{\text{shot}}(n)=t_{1q}N_{1q}(n) + t_{2q}N_{2q}(n)
    $$

    We also assume that measurement takes time $t_{\text{measure}}$ per qubit, and we include a fixed overhead term for classical control, reset, and other per-shot orchestration costs that are only negligibly affected by qubit count. Our time per shot then becomes:

    $$
    T_{\text{shot}}(n)=t_{1q}N_{1q}(n) + t_{2q}N_{2q}(n) + t_{\text{measure}}n + t_{\text{overhead}}.
    $$

    - $t_{1q}$ – execution time of a single-qubit gate
    - $N_{1q}$ – number of compiled single-qubit gates
    - $t_{2q}$ – execution time of a two-qubit gate
    - $N_{2q}$ – number of compiled two-qubit gates
    - $t_{\text{measure}}$ – measurement time per qubit
    - $t_{\text{overhead}}$ – per-shot overhead for classical control and reset
    - $n$ – circuit size in number of qubits
    """)
    return


@app.function
def T_shot(data):
    t_compute = (
        data['t1q'] * data['single_qubit_count'] + 
        data['t2q'] *  data['two_qubit_count'] + 
        data['t_meas'] * data['measure_count']
    )
    return t_compute + data['t_overhead']


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In an ideal (or at least fault-tolerant) world, we would almost surely obtain the intended result of each shot, so TTS is just $T_\text{shot}$ times the number of shots. (We may still want to run multiple shots because we care about the distribution over measurements and not a single solution bitstring.) But taking into account noise (and the absence of error correction in our circuits) the probability of measuring the correct result vanishes exponentially with circuit depth. We will take a look at what circuits with error correcting codes look like in a future demo, but for our present model we assume that the success probability of the full circuit is the product of the success probabilities of the gates that make it up:

    $$
    P_{\text{success}}(n)=
    (1-\epsilon_{1q})^{N_{1q}(n)}
    (1-\epsilon_{2q})^{N_{2q}(n)}.
    $$

    - $\epsilon_{1q}$ – error rate of a single-qubit gate
    - $\epsilon_{2q}$ – error rate of a two-qubit gate
    - $N_{1q}(n)$ – number of single-qubit gates in the compiled circuit
    - $N_{2q}(n)$ – number of two-qubit gates in the compiled circuit
    """)
    return


@app.function
def shot_success_log_prob(data):
    """Independent-gate success proxy from 1Q/2Q error rates."""
    return (
        data['single_qubit_count'] * np.log1p(-data['e1q']) + 
        data['two_qubit_count'] * np.log1p(-data['e2q'])
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If only a fraction $P_{\text{success}}$ of shots succeed, then the expected time-to-solution increases by a factor of $1/P_{\text{success}}$. Using that correction, the success-adjusted model becomes:

    $$
    \mathrm{TTS}(n,k) \approx
    {S} \cdot
    \frac{T_{\text{shot}}(n)}{P_{\text{success}}(n)}.
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Applying the Model

    Now that we have defined the basic model functions defined, we can start building some functions to collect simulation data.
    """)
    return


@app.function
def tts_data_series(df, shots=TTS_CFG['SHOTS']):
    data = {'log_pr_success': shot_success_log_prob(df)}
    data['t_shot'] = T_shot(df)
    data['log_t_shot'] = np.log(data['t_shot'])
    data['t_shot_compute'] = data['t_shot'] - df['t_overhead']
    data['log_tts_ideal'] = np.log(shots) + np.log(data['t_shot'])
    data['log_tts'] = data['log_tts_ideal'] - data['log_pr_success']
    data['tts'] = np.exp(data['log_tts'])
    return pd.Series(data)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    All we have to do is apply this function to our existing simulation data.
    """)
    return


@app.cell
def _(device_df, scaling_df):
    tts_df = scaling_df.join(
        scaling_df.merge(device_df, on='backend').apply(tts_data_series, axis=1)
    )
    tts_df.loc[:, tts_df.columns != 'circuit']
    return (tts_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploring Results

    Since we're using the same data structure, we can also re-use our plotting funtions from the previous section:
    """)
    return


@app.cell
def _(tts_df):
    plot_scaling_metric(
        tts_df, 't_shot', title='GHZ Scaling', ylabel='Shot Duration (seconds)', yscale='log',
    )
    return


@app.cell
def _(tts_df):
    plot_scaling_metric(
        tts_df, 'tts', title='GHZ Scaling', ylabel='Time-to-Solution (seconds)', yscale='log',
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interpreting Results

    Both of these plots show the same pattern. The monolithic superconducting backend has lower shot times and TTS everywhere, but both of them grow quickly with circuit size.

    1. Shot time grows because of the number of SWAP operations necessary to entangle increasingly distant qubits.
    2. TTS grows exponential (i.e., its log grows linearly) because not only are shots getting longer, but gate depth and therefore shot fidelity is decreasing, which exponentially increases the number of shots required to reach a solution.

    For the trapped-ion backend, shot speed and TTS is much higher, but both grow much more slowly. Shot speed grows approximately linearly, since each additional qubit is just another direct CNOT. TTS grows faster, but the high physical fidelities of trapped ions keep the extra number of shots from growing too quickly.

    We've reached the maximum number of qubits in the `FakeSherbrooke` backend, but it looks like the two TTS lines would cross if we could keep adding qubits. So let's try that!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Extrapolating to Utility Scale

    The IBM device caps out at 127 qubits, to get around this limitation (and avoid expensive compilation of gigantic circuits) we can use the data we have from our current simulation and assume that we would see a similar pattern as we extend to utility-scale regimes (assuming that the IBM chip would have the same heavy-hex structure at larger qubit counts).

    The graphs suggest we see a linear increase in gate counts on both monolithic and distributed systems. Assuming we continue to scale linearly, we can use a line of best fit to extrapolate to the utility scale regime we want to explore.

    For each hardware backend, we:

    1. Fit a linear model of gate type (single-qubit, two-qubit, measurement) on $n$
    2. Use predicted gate counts in the same TTS model
    3. Plot projected complexity and TTS on log scales (to handle very large numbers)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fitting a Linear Model

    We already have all the data we need to from our earlier simulations; the tricky part is getting it into the shape that we need. We can do some `pandas` magic to get a `DataFrame` where each row is a data point we can use in our linear model by combining some of our columns using `melt`:
    """)
    return


@app.cell
def _(scaling_df):
    df_linear_fit = scaling_df.melt(
        id_vars = ['backend', 'n'],
        value_vars = ['single_qubit_count', 'two_qubit_count', 'measure_count'],
        var_name = 'gate_type',
        value_name = 'gate_count',
    )
    df_linear_fit
    return (df_linear_fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we have our linear predictor `n` and our response `gate_count`, and we can group the data by `backend` and `gate_type` to get one fit for each combination:
    """)
    return


@app.cell
def _(df_linear_fit):
    def _fit(g):
        slope, intercept = np.polyfit(g['n'], g['gate_count'], 1)
        return pd.Series({'slope': slope, 'intercept': intercept})

    fits = (
        df_linear_fit.groupby(['backend', 'gate_type'])
        .apply(_fit, include_groups=False)
        .reset_index()
    )
    fits
    return (fits,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now if we add `n` as a column, we can compute the predicted `gate_count` in each row:
    """)
    return


@app.cell
def _(fits):
    _preds = fits.copy()
    _preds['n'] = 10000
    _preds['gate_count_predicted'] = np.ceil(
        _preds['intercept'] + _preds['slope'] * _preds['n']
    ).astype(int)
    _preds
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we just need to add more values for $n$! We can create a new `DataFrame` and perform a cross join to get all possible combinations (one $n$ for each backend and gate type):
    """)
    return


@app.cell
def _(fits):
    fits.merge(pd.DataFrame({'n': np.logspace(3, 5, 10).astype(int)}), how='cross')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's put the whole process into a function:
    """)
    return


@app.function
def gate_count_prediction(df, nvals):
    def _fit(g):
        slope, intercept = np.polyfit(g['n'], g['gate_count'], 1)
        return pd.Series({'slope': slope, 'intercept': intercept})

    fits = (
        df.melt(
            id_vars = ['backend', 'n'],
            value_vars = ['single_qubit_count', 'two_qubit_count', 'measure_count'],
            var_name = 'gate_type',
            value_name = 'gate_count',
        ).groupby(['backend', 'gate_type'])
        .apply(_fit, include_groups=False)
        .reset_index()
    )
    preds = fits.merge(pd.DataFrame({'n': nvals}), how='cross')
    preds['gate_count_predicted'] = np.ceil(
        preds['intercept'] + preds['slope'] * preds['n']
    ).astype(int)
    return preds.pivot_table( # go back to one column for each gate type
        index=['backend', 'n'],
        columns='gate_type',
        values='gate_count_predicted',
        aggfunc='first',
    ).reset_index()


@app.cell
def _(scaling_df):
    pred_df = gate_count_prediction(scaling_df, np.logspace(1, 5, 24).astype(int))
    pred_df
    return (pred_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we have our gate counts, we can reuse our TTS data computation functions from earlier to get our new results:
    """)
    return


@app.cell
def _(device_df, pred_df):
    extrapolation_df = pred_df.join(
        pred_df.merge(device_df, on='backend').apply(tts_data_series, axis=1)
    )
    return (extrapolation_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interpreting Results

    As before, we plot the (predicted) metrics we care about to see how they scale with circuit size.
    """)
    return


@app.cell
def _(extrapolation_df):
    plot_scaling_metric(
        extrapolation_df, 'two_qubit_count', 
        title='Extrapolated GHZ Scaling', 
        xscale='log', yscale='log', ylabel='Projected Two-Qubit Gate Count'
    )
    return


@app.cell
def _(extrapolation_df):
    # TODO: add ylim handling
    plot_scaling_metric(
        extrapolation_df, 'log_tts', 
        title='Extrapolated GHZ Scaling', 
        xscale='log', yscale='linear', ylabel='Log Time-to-Solution (seconds)'
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    These projections highlight a key scaling challenge for monolithic approaches.

    At small sizes, monolithic circuits with fast native gates can outperform distributed ones by far. As size grows, routing overhead begins to dominate. We explored this on a superconducting system, where distant qubits are entangled via SWAP chains, but routing is still an issue in architectures where qubits can be physically moved around the QPU (shuttling ions with QCCD, shuttling neutral atoms with AOMs, shuttling spin qubits from cell to cell across a silicon chip).

    Distributed systems introduce their own challenges (for example remote entanglement and coordination overhead), but they provide a practical path to larger effective system sizes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Conclusion

    In this notebook we explored how scaling quantum circuits scale between monolithic and distributed architectures strictly in terms of gate counts.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Relaxing Assumptions

    We assumed that gates are executed sequentially, but in many architectures multiple gates on different qubits can happen simultaneously. For the fake IBM backend, we can actually use Qiskit it to inspect the gate timing of a transpiled circuit:
    """)
    return


@app.cell
def _(FAKE_IBM_BACKEND):
    # compilation is non-deterministic – run this cell a few times to see the changes
    qiskit.visualization.timeline.draw(
        qiskit.transpile(ghz_circuit(7), backend=FAKE_IBM_BACKEND, optimization_level=3), 
        target=FAKE_IBM_BACKEND.target, 
        show_idle=False,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Since transpilation is non-deterministic, transpiling the same circuit multiple times results in a distribution of outcomes. (How would you modify the earlier experiments to use this information?)
    """)
    return


@app.cell
def _(FAKE_IBM_BACKEND):
    def _duration(circuit):
        transpiled =  qiskit.transpile(
            circuit, backend=FAKE_IBM_BACKEND, optimization_level=3
        )
        return transpiled.estimate_duration(target=FAKE_IBM_BACKEND.target)

    _circuit = ghz_circuit(7)
    plt.hist(pd.DataFrame({'duration': [_duration(_circuit) for _ in range(100)]}))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Other Dimensions

    However, we neglected one of the biggest advantages that distributed systems have: their ability to run computations in parallel on different modules/QPUs. One can also use this ability to run computations in parallel to run different parts of a circuit at the same time! This is one of the biggest advantages of distributed quantum computing but implementation is not as trivial as one may think. Refer back to the slides to learn more!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    # Try it Yourself!

    You may have noticed that the scaling behaviour for our circuits had the same shape across different metrics. This was due to our choice of using GHZ circuits. As we scaled up we added two exactly one CNOT which made all of our scaling trivial. Below is a sandbox where you can design your own circuit and use all of the tools we have defined to explore scaling using your own circuit!

    Start in the cell below by defining your circuit! The only requirement is to add some dependency on the number of qubits $n$ so that your circuit has some sort of scaling and is compatible with all of our functions! This is left as an excercise for your own time or if you get through the notebook earlier than others!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build Your Own Experiment

    In this section, you can swap in your own circuit family and reuse the same compilation, metric, TTS, and extrapolation tools from the rest of the notebook.

    The only requirement is that `circuit_fn(n)` returns a measured Qiskit circuit with `n` qubits.
    """)
    return


@app.cell(disabled=True)
def _():
    YOUR_CFG = {
        'N_LIST': range(9, 46, 9),
        'N_EXTRAP': np.logspace(1, 5, 24).astype(int),
        'QUBITS_PER_TRAP': 20,
        'PLOT_METRICS':  ['two_qubit_count', 'single_qubit_count', 'total_ops']
    }

    # Default is a Shor 9 Qubit error correcting circuit 
    def circuit_fn(n: int):
        if n % 9 != 0 or n < 9:
            raise ValueError('Use n as a positive multiple of 9 (e.g., 9, 18, 27, ...).')
        _qc = qiskit.QuantumCircuit(n, n)
        n_blocks = n // 9
        for b in range(n_blocks):
            o = 9 * b
            _qc.cx(o + 0, o + 3)
            _qc.cx(o + 0, o + 6)
            _qc.h(o + 0)
            _qc.h(o + 3)
            _qc.h(o + 6)
            for r in [o + 0, o + 3, o + 6]:
                _qc.cx(r, r + 1)
                _qc.cx(r, r + 2)
        _qc.measure(range(n), range(n))
        return _qc

    return YOUR_CFG, circuit_fn


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now compile your circuit family in the same two ways we used for GHZ circuits: once for the monolithic IBM backend and once for the distributed Bosonic backend.
    """)
    return


@app.cell(disabled=True)
def _(YOUR_CFG, circuit_fn, scale_ibm):
    if mo.running_in_notebook():
        _iter = mo.status.progress_bar(
            YOUR_CFG['N_LIST'],
            title='Compiling your circuits',
            subtitle='Monolithic IBM backend',
        )
    else:
        _iter = YOUR_CFG['N_LIST']

    your_ibm = [scale_ibm(n, optimization_level=3, constructor=circuit_fn) for n in _iter]
    return (your_ibm,)


@app.cell(disabled=True)
def _(YOUR_CFG, circuit_fn):
    if mo.running_in_notebook():
        _iter = mo.status.progress_bar(
            YOUR_CFG['N_LIST'],
            title='Compiling your circuits',
            subtitle='Distributed Bosonic backend',
        )
    else:
        _iter = YOUR_CFG['N_LIST']

    your_bosonic = [scale_bosonic(n, constructor=circuit_fn) for n in _iter]
    return (your_bosonic,)


@app.cell(disabled=True)
def _(your_bosonic, your_ibm):
    your_circuit_df = pd.DataFrame(your_ibm + your_bosonic)
    your_circuit_df.loc[:, your_circuit_df.columns != 'circuit']
    return (your_circuit_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, collect the same circuit metrics and TTS estimates used earlier in the notebook.
    """)
    return


@app.cell(disabled=True)
def _(device_df, your_circuit_df):
    _your_metrics = lambda g: pd.Series(circuit_metrics(g['circuit']))
    your_scaling_df = your_circuit_df.join(your_circuit_df.apply(_your_metrics, axis=1))
    your_tts_df = your_scaling_df.join(
        your_scaling_df.merge(device_df, on='backend').apply(tts_data_series, axis=1)
    )
    your_tts_df.loc[:, your_tts_df.columns != 'circuit']
    return your_scaling_df, your_tts_df


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Choose the metrics you want to inspect by editing `YOUR_METRICS_TO_PLOT` in the first cell of this section.
    """)
    return


@app.cell(disabled=True)
def _(YOUR_CFG, your_tts_df):
    for _metric in YOUR_CFG['PLOT_METRICS']:
        plot_scaling_metric(
            your_tts_df,
            _metric,
            title='Your Circuit Scaling',
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finally, extrapolate your measured gate counts and reuse the same TTS model to estimate how the circuit family might behave at larger sizes.
    """)
    return


@app.cell(disabled=True)
def _(YOUR_CFG, your_scaling_df):
    your_pred_df = gate_count_prediction(your_scaling_df, YOUR_CFG['N_EXTRAP'])
    your_extrapolation_df = your_pred_df.join(your_pred_df.apply(tts_data_series, axis=1))
    return (your_extrapolation_df,)


@app.cell(disabled=True)
def _(your_extrapolation_df):
    plot_scaling_metric(
        your_extrapolation_df,
        'two_qubit_count',
        title='Your Circuit Extrapolated Scaling',
        xscale='log',
        yscale='log',
        ylabel='Projected Two-Qubit Gate Count',
    )

    plot_scaling_metric(
        your_extrapolation_df,
        'log_tts',
        title='Your Circuit Extrapolated Scaling',
        xscale='log',
        yscale='linear',
        ylabel='Log Time-to-Solution (seconds)',
    )
    return


if __name__ == "__main__":
    app.run()
