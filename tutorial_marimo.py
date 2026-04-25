# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "ipykernel",
#     "marimo",
#     "pandas",
#     "matplotlib",
#     "qiskit>=1.0",
#     "qiskit-aer",
#     "qiskit-ibm-runtime",
#     "bosonic-sdk",
#     "bosonic-disqco",
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
# bosonic-disqco = { index = "test-pypi" }
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    # Setup Cell: import all dependencies and define global constants

    import marimo as mo

    # standard/numerical
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # qiskit
    # we avoid the pattern `from qiskit.submodule import Class` to make it clear in each cell where each function/class comes from
    import qiskit
    import qiskit_ibm_runtime
    import qiskit_aer

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
        'N_LIST': range(5, 126, 5),
        'QUBITS_PER_TRAP': 32,
    }

    TTS_CFG = {
        'SHOTS': 1024,
        'N_LIST': range(5, 126, 5),
        'QUBITS_PER_TRAP': 128,
        'IBM_TIMING': {
            't1q': 2e-8, # average one-qubit gate duration
            't2q': 2e-7, # average two-qubit gate duration
            't_meas': 1e-6, # average measurement duration
            't_overhead': 2e-4, # device/control overhead
            'e1q': 5e-4, # average one-qubit gate error
            'e2q': 3e-3, # average two-qubit gate error
        },
        'BOSONIC_TIMING': {
            't1q': 1e-6,
            't2q': 3e-5,
            't_meas': 4e-4,
            't_overhead': 1e-3,
            'e1q': 1e-6,
            'e2q': 1e-4,
        },
        'TIMING': {
            'IBM': {
                't1q': 2e-8, # average one-qubit gate duration
                't2q': 2e-7, # average two-qubit gate duration
                't_meas': 1e-6, # average measurement duration
                't_overhead': 2e-4, # device/control overhead
                'e1q': 5e-4, # average one-qubit gate error
                'e2q': 3e-3, # average two-qubit gate error
            },
            'Bosonic': {
                't1q': 1e-6,
                't2q': 3e-5,
                't_meas': 4e-4,
                't_overhead': 1e-3,
                'e1q': 1e-6,
                'e2q': 1e-4,
            },
        },
        'P_SUCCESS_FLOOR': 1e-300,
        'TTS_PLOT_MAX': 1e+12,
        'GROWTH_SWEEP_MAX_N': 126,
        'IBM_OPTIMIZATION_LEVEL': 1, # just to speed things up
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
    qiskit.visualization.plot_gate_map(FAKE_IBM_BACKEND)
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


@app.cell
def _():
    # the hypergraph partitioning algorithm depends on KaHyPar, which does not run on Windows (https://github.com/kahypar/kahypar#requirements)
    # import sys
    # if sys.platform != 'win32':
    #     from bosonic_sdk.distributor.distributors.hypergraph_distributor import HypergraphDistributor
    # else:
    #     HypergraphDistributor = None
    return


@app.function
def module_count(n, qubits_per_module):
    """Minimum module count needed to host n qubits at fixed capacity."""
    return max(1, np.ceil(int(n) / int(qubits_per_module)))


@app.function
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
    3. Compile and measure distributed metrics with dynamic module count $k = \lceil \frac{n}{20}/ 0 \rceil$
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
def update_metrics(data, circuit):
    converted = CircuitConverters.from_qiskit(circuit)
    data.update(bosonic_sdk.GateStatistics.stats(converted))
    data.update(qiskit_metrics(circuit))
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
    def scaling_ibm(n, constructor=ghz_circuit, backend=FAKE_IBM_BACKEND, **transpile_kwargs):
        data = {'backend': 'IBM', 'n': n, 'k': 1}
        circuit = qiskit.transpile(constructor(n), backend=FAKE_IBM_BACKEND, **transpile_kwargs)
        return update_metrics(data, circuit)

    return (scaling_ibm,)


@app.function
def scaling_bosonic(
    n, 
    constructor=ghz_circuit, 
    qubits_per_trap=SCALING_CFG['QUBITS_PER_TRAP'], 
    distributor=bosonic_sdk.BosonicDistributor()
):
    k = np.ceil(n / qubits_per_trap).astype(int)
    data = {'backend': 'Bosonic', 'n': n, 'k': k}
    circuit = compile_bosonic_circuit(constructor(n), n, k, distributor)
    return update_metrics(data, circuit)


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
    Now we loop over every value of $n$ we defined in our config and collect the results into a `DataFrame`.
    """)
    return


@app.cell
def _(scale_ibm):
    ibm_circuits = [scale_ibm(n, optimization_level=3) for n in mo.status.progress_bar(
        range(3, 128),
        title='Compiling',
        subtitle='Monolithic IBM backend',
    )]
    return (ibm_circuits,)


@app.cell
def _():
    bosonic_circuits = [scale_bosonic(n) for n in mo.status.progress_bar(
        range(3, 128),
        title='Compiling',
        subtitle='Distributed Bosonic backend',
    )]
    return (bosonic_circuits,)


@app.cell
def _(bosonic_circuits, ibm_circuits):
    circuit_df = pd.DataFrame(ibm_circuits + bosonic_circuits)
    circuit_df.loc[:, circuit_df.columns != 'circuit']
    return (circuit_df,)


@app.cell
def _(circuit_df):
    metrics_df = pd.DataFrame(circuit_df['circuit'].apply(circuit_metrics).tolist())
    metrics_df
    return (metrics_df,)


@app.cell
def _(circuit_df, metrics_df):
    scale_df = circuit_df.join(metrics_df)
    scale_df.loc[:, scale_df.columns != 'circuit']
    return (scale_df,)


@app.cell
def _(scale_df):
    # scaling_df = pd.DataFrame(
    #     [scaling_ibm(n) for n in SCALING_CFG['N_LIST']] +
    #     [scaling_bosonic(n) for n in SCALING_CFG['N_LIST']]
    # )
    scaling_df = scale_df
    return (scaling_df,)


@app.cell
def _(FAKE_IBM_BACKEND, ibm_circuits):
    ibm_circuits[100]['circuit'].estimate_duration(target=FAKE_IBM_BACKEND.target)
    return


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
def plot_scaling_metric(df, metric, title='GHZ Circuit Scaling'):
    for _backend in ['IBM', 'Bosonic']:
        subdf = df[df['backend'] == _backend]
        plt.plot(subdf['n'], subdf[metric], label=_backend)

    plt.title(title)
    plt.xlabel('Number of Qubits')
    plt.ylabel(metric)
    plt.legend(title='Backend')
    
    plt.tight_layout()
    plt.show()


@app.cell
def _(scaling_df):
    plot_scaling_metric(scaling_df, 'depth')
    return


@app.cell
def _(scaling_df):
    plot_scaling_metric(scaling_df, 'two_qubit_count')
    return


@app.cell
def _(scaling_df):
    plot_scaling_metric(scaling_df, 'single_qubit_count')
    return


@app.cell
def _(scaling_df):
    plot_scaling_metric(scaling_df, 'qubit_teleportation_count')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interpreting the Results

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
def T_shot(circuit_data, device=TTS_CFG['IBM_TIMING']):
    t_compute = (
        device['t1q'] * circuit_data['single_qubit_count'] + 
        device['t2q'] *  circuit_data['two_qubit_count'] + 
        device['t_meas'] * circuit_data['measure_count']
    )
    return t_compute + device['t_overhead']


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
def shot_success_log_prob(circuit_data, device=TTS_CFG['IBM_TIMING']):
    """Independent-gate success proxy from 1Q/2Q error rates."""
    return (
        circuit_data['single_qubit_count'] * np.log1p(-device['e1q']) + 
        circuit_data['two_qubit_count'] * np.log1p(-device['e2q'])
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
def tts_data_row(circuit_data, device, shots=TTS_CFG['SHOTS']):
    data = {'log_pr_success': shot_success_log_prob(circuit_data, device=device)}
    data['t_shot'] = T_shot(circuit_data, device=device)
    data['log_t_shot'] = np.log(data['t_shot'])
    data['t_shot_compute'] = data['t_shot'] - device['t_overhead']
    data['log_tts_ideal'] = np.log(shots) + np.log(data['t_shot'])
    data['log_tts'] = data['log_tts_ideal'] - data['log_pr_success']
    data['tts'] = np.exp(data['log_tts'])

    circuit_data.update(data)
    return circuit_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's test out this function for a simple $n=3$ circuit:
    """)
    return


@app.cell
def _():
    tts_data_row(
        scaling_bosonic(3, qubits_per_trap=TTS_CFG['QUBITS_PER_TRAP']), 
        device=TTS_CFG['BOSONIC_TIMING']
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    All the data that we had before is still there (which means we can reuse our plotting functions!), but now we have time/shot data as well.

    Now let's generate all of our simulation data:
    """)
    return


@app.cell
def _(scaling_ibm):
    def generate_tts_data(cfg, constructor):
        ibm_data = [
            tts_data_row(
                scaling_ibm(
                    n, 
                    constructor=constructor, 
                    optimization_level=cfg['IBM_OPTIMIZATION_LEVEL']
                ),
                device=cfg['IBM_TIMING'],
            ) for n in cfg['N_LIST']
        ]
        bosonic_data = [
            tts_data_row(
                scaling_bosonic(
                    n, 
                    constructor=constructor, 
                    qubits_per_trap=cfg['QUBITS_PER_TRAP']
                ),
                device=cfg['BOSONIC_TIMING'],
            ) for n in cfg['N_LIST']
        ]
        return pd.DataFrame(ibm_data + bosonic_data)

    return (generate_tts_data,)


@app.cell
def _(generate_tts_data):
    tts_df = generate_tts_data(TTS_CFG, constructor=ghz_circuit)
    tts_df
    return (tts_df,)


@app.cell
def _(tts_df):
    plot_scaling_metric(tts_df, 'log_t_shot')
    return


@app.cell
def _(tts_df):
    plot_scaling_metric(tts_df, 'log_tts')
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
    # Extrapolating to Useful Circuit Sizes

    The IBM device caps out at 127 qubits, to get around this and avoid expensive simulation we can use the data we have from our current simulation and assume that we would see a similar pattern as we extent to utility scale regimes (assuming that the IBM chip would have the same structure but grow in qubit counts).

    The graphs suggest we see a linear increase in gate counts between monolithic and distributed system. Assuming we continue to scale linearly we can use a line of best fit to extrapolate to the utility scale regime we want to explore.

    For each scenario (`monolithic`, `distributed`), we:
    1. fit linear trends on transpiled points:
       - `N1(n) ~ a1 * n + b1`
       - `N2(n) ~ a2 * n + b2`
    2. Use projected `N1, N2, N_meas` in the same TTS model.
    3. Plot projected complexity and projected `TTS_shots` on log scales.
    """)
    return


@app.cell
def _(scaling_df):
    # use scaling data to get linear coefficients
    scaling_df.columns.tolist()
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


@app.cell
def _():
    return


@app.cell
def _(fits):
    preds = fits.merge(pd.DataFrame({'n': np.logspace(1, 5, 24).astype(int)}), how='cross')
    preds['gate_count_predicted'] = np.ceil(preds['intercept'] + preds['slope'] * preds['n']).astype(int)
    preds
    return (preds,)


@app.cell
def _(preds):
    pred_df = preds.pivot_table(
        index=['backend', 'n'],
        columns='gate_type',
        values='gate_count_predicted',
        aggfunc='first',
    ).reset_index()
    pred_df
    return (pred_df,)


@app.function
def gate_count_prediction(fits, nvals):
    preds = fits.merge(pd.DataFrame({'n': nvals}), how='cross')
    preds['gate_count_predicted'] = np.ceil(
        preds['intercept'] + preds['slope'] * preds['n']
    ).astype(int)
    return preds.pivot_table(
        index=['backend', 'n'],
        columns='gate_type',
        values='gate_count_predicted',
        aggfunc='first',
    ).reset_index()


@app.function
def tts_data_series(circuit_data, shots=TTS_CFG['SHOTS']):
    device = TTS_CFG['TIMING'][circuit_data['backend']]
    data = {'log_pr_success': shot_success_log_prob(circuit_data, device=device)}
    data['t_shot'] = T_shot(circuit_data, device=device)
    data['log_t_shot'] = np.log(data['t_shot'])
    data['t_shot_compute'] = data['t_shot'] - device['t_overhead']
    data['log_tts_ideal'] = np.log(shots) + np.log(data['t_shot'])
    data['log_tts'] = data['log_tts_ideal'] - data['log_pr_success']
    data['tts'] = np.exp(data['log_tts'])
    return pd.Series(data)


@app.cell
def _(pred_df):
    pred_df.join(pred_df.apply(tts_data_series, axis=1))
    return


@app.cell
def _():
    return


@app.cell
def _(
    BosonicDistributor,
    IBM_OPT_LEVEL_GROWTH,
    QUBITS_PER_TRAP_1,
    SHOTS_2,
    TTS_from_shots,
    T_shot_from_counts,
    compile_bosonic,
    compile_monolithic_ibm_fake,
    distributed_trap_count,
    gate_counts,
    shot_success_probability,
):
    N_EXTRAP = np.unique(np.logspace(0, 5, 24).astype(int)).tolist()
    GROWTH_SWEEP_MAX_N = 126

    def fit_linear(n_vals, y_vals):
        _n = np.asarray(n_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        mask = np.isfinite(_n) & np.isfinite(y)
        _n = _n[mask]
        y = y[mask]
        if _n.size < 2:
            return None
        a, b = np.polyfit(_n, y, 1)
        return (float(a), float(b))

    def predict_linear(n_vals, params):
        if params is None:
            return np.full(len(n_vals), np.nan)
        a, b = params
        _n = np.asarray(n_vals, dtype=float)
        return a * _n + b
    rows = []
    n_growth_list = list(range(2, GROWTH_SWEEP_MAX_N + 1))
    for _n in n_growth_list:
        qc = ghz_circuit(_n)
        qc_ibm = compile_monolithic_ibm_fake(qc, optimization_level=IBM_OPT_LEVEL_GROWTH)
        N1_i, N2_i, Nm_i = gate_counts(qc_ibm)
        rows.append({'n': _n, 'scenario': 'monolithic', 'label': f'IBM FakeSherbrooke (opt={IBM_OPT_LEVEL_GROWTH})', 'N1': N1_i, 'N2': N2_i, 'N_meas': Nm_i})
        k_growth = distributed_trap_count(_n)
        qc_bos = compile_bosonic(qc, _n, k_growth, BosonicDistributor())
        N1_b, N2_b, Nm_b = gate_counts(qc_bos)
        rows.append({'n': _n, 'scenario': 'distributed', 'label': f'Bosonic (traps=ceil(n/{QUBITS_PER_TRAP_1}))', 'N1': N1_b, 'N2': N2_b, 'N_meas': Nm_b})
    df_growth_sweep = pd.DataFrame(rows)
    fit_rows = []
    _proj_rows = []
    source = df_growth_sweep.copy()
    for _scenario in ['monolithic', 'distributed']:
        _sub = source[source['scenario'] == _scenario].sort_values('n')
        _p1 = fit_linear(_sub['n'].values, _sub['N1'].values)
        _p2 = fit_linear(_sub['n'].values, _sub['N2'].values)
        _pm = fit_linear(_sub['n'].values, _sub['N_meas'].values)
        fit_rows.append({'scenario': _scenario, 'N1_slope': np.nan if _p1 is None else _p1[0], 'N1_intercept': np.nan if _p1 is None else _p1[1], 'N2_slope': np.nan if _p2 is None else _p2[0], 'N2_intercept': np.nan if _p2 is None else _p2[1], 'Nmeas_slope': np.nan if _pm is None else _pm[0], 'Nmeas_intercept': np.nan if _pm is None else _pm[1]})
        _n_proj = np.array(N_EXTRAP, dtype=float)
        _n1_proj = np.maximum(predict_linear(_n_proj, _p1), 0.0)
        _n2_proj = np.maximum(predict_linear(_n_proj, _p2), 0.0)
        _nm_proj = np.maximum(predict_linear(_n_proj, _pm), 0.0)
        for _n_val, _n1_hat, _n2_hat, _nm_hat in zip(_n_proj, _n1_proj, _n2_proj, _nm_proj):
            _k_modules = 1 if _scenario == 'monolithic' else distributed_trap_count(_n_val)
            _p_succ_hat = shot_success_probability(_n1_hat, _n2_hat, scenario=_scenario)
            _t_shot_hat = T_shot_from_counts(_n1_hat, _n2_hat, N_meas=_nm_hat, scenario=_scenario)
            tts_ideal_hat = SHOTS_2 * _t_shot_hat
            _tts_hat = TTS_from_shots(_t_shot_hat, p_success=_p_succ_hat, shots=SHOTS_2)
            _proj_rows.append({'n': int(_n_val), 'scenario': _scenario, 'k_modules': int(_k_modules), 'N1_hat': float(_n1_hat), 'N2_hat': float(_n2_hat), 'N_meas_hat': float(_nm_hat), 'P_success_hat': float(_p_succ_hat), 'T_shot_hat': float(_t_shot_hat), 'TTS_shots_ideal_hat': float(tts_ideal_hat), 'TTS_shots_hat': float(_tts_hat)})
    df_proj = pd.DataFrame(_proj_rows)
    return df_proj, fit_linear, predict_linear


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### We now plot extrapolated results on log-log axes. On a log-log plot, straight lines indicate power-law-like scaling behavior over the displayed range.
    """)
    return


@app.cell
def _(TTS_PLOT_MAX, df_proj):
    if df_proj.empty:
        raise RuntimeError('df_proj is missing or empty. Run the extrapolation-fit cell first.')
    _scenario_labels = {'monolithic': 'Monolithic (IBM fake)', 'distributed': 'Distributed (Bosonic, traps=ceil(n/128))'}
    scenarios_present = [s for s in df_proj['scenario'].dropna().unique().tolist()]
    plt.figure(figsize=(8.5, 4.5))
    for _scenario in scenarios_present:
        prj = df_proj[df_proj['scenario'] == _scenario].sort_values('n')
        y = prj['N2_hat'].replace([np.inf, -np.inf], np.nan)
        y = y.clip(lower=1.0)
        m = np.isfinite(y) & np.isfinite(prj['n'])
        if m.any():
            plt.plot(prj.loc[m, 'n'], y[m], marker='o', label=_scenario_labels.get(_scenario, _scenario))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=1.0)
    plt.xlabel('n qubits (log scale)')
    plt.ylabel('Projected 2Q gate count (log scale)')
    plt.title('Projected 2Q growth for n in [10^0, 10^5]')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.show()
    plt.figure(figsize=(8.5, 4.5))
    cutoff_notes = []
    for _scenario in scenarios_present:
        prj = df_proj[df_proj['scenario'] == _scenario].sort_values('n').copy()
        y = prj['TTS_shots_hat'].replace([np.inf, -np.inf], np.nan)
        valid = np.isfinite(y) & np.isfinite(prj['n']) & (y > 0) & (y <= TTS_PLOT_MAX) & (y <= 1000000000000.0)
        if valid.any():
            x_ok = prj.loc[valid, 'n']
            y_ok = y[valid]
            plt.plot(x_ok, y_ok, marker='o', label=_scenario_labels.get(_scenario, _scenario))
        invalid = ~np.isfinite(y) | (y <= 0) | (y > TTS_PLOT_MAX) | (y > 1000000000000.0)
        if invalid.any():
            n_cut = int(prj.loc[invalid, 'n'].iloc[0])
            cutoff_notes.append(f'{_scenario_labels.get(_scenario, _scenario)}: truncated at n >= {n_cut}')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1.0, 1000000000000.0)
    plt.xlabel('n qubits (log scale)')
    plt.ylabel('Projected Time To Solution (s, log scale)')
    plt.title('Extrapolated Time To Solution for n in [10^0, 10^5]')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.show()
    if cutoff_notes:
        print('Extrapolation truncation notes (overflow/instability guardrail):')
        for note in cutoff_notes:
            print(' -', note)
        print('Values beyond these cutoffs are omitted from the plot to avoid numerical overflow errors')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Interpreting the Results

    These projections highlight a key scaling challenge for monolithic approaches.

    At small sizes, monolithic circuits with fast native gates can outperform distributed ones by far. As size grows, routing overhead in nearest-neighbor architectures creates fast growth in two-qubit costs and greatly imparcting Time-To-Solution. This is a key challenge monolithic approaches face and why as we try to scale to useful qubit counts the industry is turning to distributed quantum computing.

    Distributed systems introduce their own challenges (for example remote entanglement and coordination overhead), but they provide a practical path to larger effective system sizes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    --------------------------------------------------------------------------------------------------------------------------------------

    # Distributing Circuits

    In this notebook we explored how scaling quantum circuits scale between monolithic and distributed architectures stricly in terms of gate counts. However, we neglected one of the biggest advantages that distributed systems have, their ability to run computations in parallel on different modules/QPUs. One can also use this ability to run computations in parallel to run different parts of a circuit at the same time! This is one of the biggest advantages of distributed quantum computing but implementation is not as trivial as one may think. Refer back to the slides to learn more!
    """)
    return


@app.cell
def _(FAKE_IBM_BACKEND, ibm_circuits):
    qiskit.visualization.timeline.draw(
        ibm_circuits[3]['circuit'], 
        target=FAKE_IBM_BACKEND.target, 
        show_idle=False,
    )
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
    --------------------------------------------------------------------------------------------------------------------------------------
    # Try It Yourself: Build Your Own Scaling Experiment
    - here we compare two different distribution plugins: BosonicDistributor (simple naive implementation) and DisqcoDistributor (state of the art algorithm; https://github.com/felix-burt/DISQCO)
    """)
    return


@app.cell(disabled=True)
def _(QuantumCircuit):
    YOUR_N_LIST = range(9,100, 9)
    YOUR_N_EXTRAP = np.unique(np.logspace(0, 5, 24).astype(int)).tolist()
    YOUR_BOSONIC_QUBITS_PER_MODULE = 20

    # Default is a Shor 9 Qubit error correcting circuit 
    def circuit_fn(n: int):
        if n % 9 != 0 or n < 9:
            raise ValueError('Use n as a positive multiple of 9 (e.g., 9, 18, 27, ...).')
        _qc = QuantumCircuit(n, n)
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

    return (
        YOUR_BOSONIC_QUBITS_PER_MODULE,
        YOUR_N_EXTRAP,
        YOUR_N_LIST,
        circuit_fn,
    )


@app.cell(disabled=True)
def _():
    ## These are a list of available metrics to plot, you do not need to edit this cell it serves only as a library of available metrics. 

    YOUR_METRIC_REGISTRY = {
        'depth': lambda s, g, q: s['depth'],
        'two_qubit_depth': lambda s, g, q: q['two_qubit_depth'],
        'two_qubit_count': lambda s, g, q: q['two_qubit_count'],
        'single_qubit_count': lambda s, g, q: q['single_qubit_count'],
        'total_ops': lambda s, g, q: s['total_ops'],
        'measure_count': lambda s, g, q: q['measure_count'],
        'reset_count': lambda s, g, q: q['reset_count'],
        'remote_link_count': lambda s, g, q: g.get('remote_link_psi_minus', 0) + g.get('remote_link_psi_plus', 0),
        'local_gate_count': lambda s, g, q: s['local_gate_count'],
        'remote_gate_count': lambda s, g, q: s['remote_gate_count'],
        'qubit_teleportation_count': lambda s, g, q: s['qubit_teleportation_count'],
    }
    YOUR_METRIC_LABELS = {
        'depth': 'Depth',
        'two_qubit_depth': 'Two-Qubit Layer Depth',
        'two_qubit_count': 'Two-Qubit Gate Count',
        'single_qubit_count': 'Single-Qubit Gate Count',
        'total_ops': 'Total Ops',
        'measure_count': 'Measurement Count',
        'reset_count': 'Reset Count',
        'remote_link_count': 'Remote Link Count',
        'local_gate_count': 'Local Gate Count',
        'remote_gate_count': 'Remote Gate Count',
        'qubit_teleportation_count': 'Qubit Teleportation Count',
    }
    return YOUR_METRIC_LABELS, YOUR_METRIC_REGISTRY


@app.cell(disabled=True)
def _(
    YOUR_BOSONIC_QUBITS_PER_MODULE,
    YOUR_METRIC_LABELS,
    YOUR_METRIC_REGISTRY,
    extract_metrics,
):
    # Here you can adjust the list "YOUR_METRICS_TO_PLOT" to choose which metrics you would like to plot! 
    YOUR_METRICS_TO_PLOT = ['two_qubit_count', 'single_qubit_count', 'total_ops']

    AVAILABLE_METRICS = list(YOUR_METRIC_REGISTRY.keys())
    for metric in YOUR_METRICS_TO_PLOT:
        if metric not in AVAILABLE_METRICS:
            raise ValueError(f'Unknown metric selected for plotting: {metric}')
    YOUR_PLOT_METRIC_PAIRS = [(metric, YOUR_METRIC_LABELS[metric]) for metric in YOUR_METRICS_TO_PLOT]

    def your_module_count(n, qubits_per_module=YOUR_BOSONIC_QUBITS_PER_MODULE):
        return module_count(n, qubits_per_module)

    def extract_your_metrics(qc, metric_keys=YOUR_METRICS_TO_PLOT):
        metric_pairs = [(metric, YOUR_METRIC_LABELS[metric]) for metric in metric_keys]
        return extract_metrics(qc, YOUR_METRIC_REGISTRY, metric_pairs)

    return YOUR_PLOT_METRIC_PAIRS, extract_your_metrics, your_module_count


@app.cell(disabled=True)
def _(fit_linear, math, predict_linear):
    def run_your_benchmark(circuit_fn, n_list, distributor_list, compile_dist_fn, compile_mono_fn, build_tts_fn, extract_metrics_fn, module_count_fn, qubits_per_module):
        rows = []
        for n in n_list:
            qc = circuit_fn(n)
            qc_mono = compile_mono_fn(qc)
            print(f"compiling circuit({n}) with IBM monolith")
            rows.append({
                **build_tts_fn(n, 'monolithic', 1, qc_comp=qc_mono),
                'distributor': 'monolithic',
                'label': 'Monolithic (IBM fake)',
                **extract_metrics_fn(qc_mono),
            })
        for dist_name, distributor in distributor_list:
            for n in n_list:
                print(f"compiling circuit({n}) with distributor {distributor}")
                qc = circuit_fn(n)
                k = module_count_fn(n)
                qc_dist = compile_dist_fn(qc, n, k, distributor)
                rows.append({
                    **build_tts_fn(n, 'distributed', k, qc_comp=qc_dist),
                    'distributor': dist_name,
                    'label': f'Distributed ({dist_name}, k=ceil(n/{qubits_per_module}))',
                    **extract_metrics_fn(qc_dist),
                })
        return pd.DataFrame(rows)

    def plot_your_metrics(df, metric_pairs, qubits_per_module):
        dist_names = [d for d in df['distributor'].unique() if d != 'monolithic']
        all_lines = [('monolithic', 'Monolithic (IBM fake)')] + [
            (d, f'Distributed ({d}, k=ceil(n/{qubits_per_module}))') for d in dist_names
        ]
        n_metrics = len(metric_pairs)
        nrows = math.ceil(n_metrics / 2)
        fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows))
        axes = axes.flatten()
        for ax, (metric, metric_title) in zip(axes, metric_pairs):
            for dist_name, label in all_lines:
                sub = df[df['distributor'] == dist_name].sort_values('n')
                if not sub.empty:
                    ax.plot(sub['n'], sub[metric], marker='o', label=label)
            ax.set_title(metric_title)
            ax.set_xlabel('n qubits')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
        for ax in axes[n_metrics:]:
            ax.set_visible(False)
        axes[0].legend()
        fig.suptitle('Your circuit: selected complexity metrics', y=1.02)
        plt.tight_layout()
        plt.show()

    def extrapolate_and_plot(df, n_extrap, qubits_per_module):
        dist_names = [d for d in df['distributor'].unique() if d != 'monolithic']
        proj_rows = []
        n_proj = np.asarray(n_extrap, dtype=float)
        for dist_name in ['monolithic'] + dist_names:
            sub = df[df['distributor'] == dist_name].sort_values('n')
            pt = fit_linear(sub['n'], sub['total_ops'])
            nt_proj = np.maximum(predict_linear(n_proj, pt), 0.0)
            for n_val, nt_hat in zip(n_proj, nt_proj):
                proj_rows.append({'n': int(n_val), 'distributor': dist_name, 'total_ops_hat': float(nt_hat)})
        df_proj = pd.DataFrame(proj_rows)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for dist_name in ['monolithic'] + dist_names:
            sub = df_proj[df_proj['distributor'] == dist_name].sort_values('n')
            label = 'Monolithic (IBM fake)' if dist_name == 'monolithic' else f'Distributed ({dist_name}, k=ceil(n/{qubits_per_module}))'
            y_tot = sub['total_ops_hat'].replace([np.inf, -np.inf], np.nan).clip(lower=1.0)
            m_tot = np.isfinite(y_tot)
            ax.plot(sub.loc[m_tot, 'n'], y_tot[m_tot], marker='o', label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Projected total ops')
        ax.set_xlabel('n qubits (log scale)')
        ax.set_ylabel('Projected total ops (log scale)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return extrapolate_and_plot, plot_your_metrics, run_your_benchmark


@app.cell(disabled=True)
def _(
    BosonicDistributor,
    DisqcoDistributor,
    HypergraphDistributor,
    YOUR_BOSONIC_QUBITS_PER_MODULE,
    YOUR_N_LIST,
    build_tts_row,
    circuit_fn,
    compile_bosonic,
    compile_monolithic_ibm_fake,
    extract_your_metrics,
    run_your_benchmark,
    your_module_count,
):
    distributor_list = [
        ('bosonic', BosonicDistributor()),
        ('disqco', DisqcoDistributor()),
    ]
    if HypergraphDistributor is not None:
        distributor_list.append(('hypergraph', HypergraphDistributor()))
    df_your = run_your_benchmark(
        circuit_fn=circuit_fn,
        n_list=YOUR_N_LIST,
        distributor_list=distributor_list,
        compile_dist_fn=compile_bosonic,
        compile_mono_fn=compile_monolithic_ibm_fake,
        build_tts_fn=build_tts_row,
        extract_metrics_fn=extract_your_metrics,
        module_count_fn=your_module_count,
        qubits_per_module=YOUR_BOSONIC_QUBITS_PER_MODULE,
    )
    return (df_your,)


@app.cell(disabled=True)
def _(
    YOUR_BOSONIC_QUBITS_PER_MODULE,
    YOUR_PLOT_METRIC_PAIRS,
    df_your,
    plot_your_metrics,
):
    plot_your_metrics(df_your, YOUR_PLOT_METRIC_PAIRS, YOUR_BOSONIC_QUBITS_PER_MODULE)
    return


@app.cell(disabled=True)
def _(
    YOUR_BOSONIC_QUBITS_PER_MODULE,
    YOUR_N_EXTRAP,
    df_your,
    extrapolate_and_plot,
):
    extrapolate_and_plot(df_your, YOUR_N_EXTRAP, YOUR_BOSONIC_QUBITS_PER_MODULE)
    return


if __name__ == "__main__":
    app.run()
