import marimo

__generated_with = "0.23.5"
app = marimo.App()

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import altair as alt
    import os
    import re
    from pathlib import Path
    from itertools import islice
    import qiskit
    import qiskit.qasm2
    import qiskit_ibm_runtime
    import qiskit_aer

    FILE_DIR = Path(__file__).parent.resolve()
    DATASET_DIR = FILE_DIR / "transpiled-circuits"
    BACKEND = qiskit_ibm_runtime.fake_provider.FakeSherbrooke()


@app.cell
def _():
    mo.md(r"""
    # GHZ Duration Experiment

    The goal of this experiment is to explore the scaling behavior of GHZ circuits on the `FakeSherbrooke` backend using the information provided by the backend itself. The reason we call this an "experiment" is that transpilation is non-deterministic by default, giving us a distribution of circuits for each number of qubits $n$ rather than a single deterministic outcome.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Previous Steps

    This experiment has some long-running steps that we offload to scripts that should run before this notebook.

    1. `transpile-all.py` generates $m$ transpiled circuits for each number of qubits $n$ (default $m=1000$) and write them to disk as QASM files (takes hours to run)
    2. `parse-data.py` loads the generated QASM files back into Qiskit and extracts circuit features into `circuit-stats.arrow` (takes ~10 minutes to run after optimizing the code)

    You can run these steps on your own machine to generate `circuit-stats.arrow` if it does not (we don't commit datasets to git).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load the Data
    """)
    return


@app.cell
def _():

    df = pd.read_feather(FILE_DIR / "circuit-stats.arrow")
    df.info()
    return (df,)


@app.cell
def _(df):
    qmin = int(df["num_clbits"].min())
    qmax = int(df["num_clbits"].max())

    qubit_range = mo.ui.range_slider(start=qmin, stop=qmax, value=[qmin, qmax], label="Qubit range")
    y_metric = mo.ui.dropdown(
        options=[
            "depth", "size", "duration_s", "log_fidelity", "log_fidelity_measure",
            "depth_per_qubit", "size_per_qubit", "ecr_per_qubit", "two_qubit_fraction"
        ],
        value="depth",
        label="Y metric",
    )
    color_metric = mo.ui.dropdown(
        options=["ecr", "size", "depth", "duration_s", "log_fidelity", "two_qubit_fraction"],
        value="ecr",
        label="Color metric",
    )
    mo.hstack([qubit_range, y_metric, color_metric])
    return qmax, qmin


@app.cell
def _(qmax, qmin):
    qubit_slider = mo.ui.slider(
        start=qmin,
        stop=qmax,
        step=1,
        value=qmin,
        label="Number of qubits",
        show_value=True,
    )

    bins_slider = mo.ui.slider(
        start=5,
        stop=100,
        step=1,
        value=40,
        label="Bins",
        show_value=True,
    )

    mo.hstack([qubit_slider, bins_slider])
    return bins_slider, qubit_slider


@app.cell
def _(bins_slider, df, qubit_slider):
    _filtered_df = df[df["num_clbits"] == qubit_slider.value]
    chart = alt.Chart(_filtered_df).mark_bar().encode(
        x=alt.X(
            "duration_s:Q",
            bin=alt.Bin(maxbins=bins_slider.value),
            title="duration_s",
        ),
        y=alt.Y("count()", title="Count"),
        tooltip=[alt.Tooltip("count()", title="Rows")],
    ).properties(
        width=700,
        height=350,
        title=f"duration_s histogram for {qubit_slider.value} qubits (n={len(_filtered_df)})",
    )
    mo.ui.altair_chart(chart)
    return


@app.cell
def _(df):
    _summary = (
        df.groupby("num_clbits")['duration_dt']
          .agg(
              count="count",
              mean="mean",
              std="std",
              min="min",
              p10=lambda s: s.quantile(0.10),
              median="median",
              p90=lambda s: s.quantile(0.90),
              max="max",
          )
          .reset_index()
    )

    _summary
    return


@app.cell
def _(df):
    rep_idx = df.index.repeat(df["count"])
    df_expanded = df.loc[rep_idx].drop(columns=["count"]).reset_index(drop=True)
    df_expanded.info()
    return (df_expanded,)


@app.cell
def _(df_expanded):
    stat_df = df_expanded.melt(
        id_vars=['num_clbits'],
        value_vars=['duration_dt', 'size', 'depth', 'log_fidelity', 'log_fidelity_measure'],
        var_name='variable',
        value_name='value',
    ).groupby(['num_clbits', 'variable'])['value'].agg(
        mean='mean',
        std='std',
        min='min',
        p10=lambda s: s.quantile(0.1),
        median='median',
        p90=lambda s: s.quantile(0.9),
        max='max',
    ).reset_index()

    stat_df
    return (stat_df,)


@app.cell
def _(stat_df):
    variable_dropdown = mo.ui.dropdown(
        options=sorted(stat_df["variable"].unique().tolist()),
        value=sorted(stat_df["variable"].unique().tolist())[0],
        label="Variable to plot",
        searchable=True,
    )
    variable_dropdown
    return


app._unparsable_cell(
    r"""
    _chart_df = stat_df[stat_df"'variable"] == variable_dropdown.value]

    _titles = {
        "depth": "Circuit Depth",
        "size": "Circuit Gate Count",
        "log_fidelity": "Log Circuit Fidelity (Excluding Measurement)",
        "log_fidelity_measure": "Log Circuit Fidelity (Including Measurement)",
    }

    _base = alt.Chart(_chart_df).encode(
        x=alt.X("num_clbits:Q", title="Number of Qubits"),
    )

    _band = _base.mark_area(opacity=0.15).encode(
        y="p10:Q",
        y2="p90:Q",
    )

    _line = _base.mark_line().encode(
        y="median:Q",
    )

    _chart = (_band + _line).properties(
        title=variable_dropdown.value,
        width=500,
    )

    _chart
    """,
    name="_"
)


@app.cell
def _(stat_df):
    _dt_df = stat_df[
        stat_df['variable'] == 'duration_dt'
    ][
        ['num_clbits', 'median']
    ].rename(
        columns={'num_clbits': 'n', 'median': 'dt'}
    ).assign(
        dt=lambda df: df['dt'].astype(int)
    ).set_index('n')

    _dt_df.to_csv('duration_dt.csv')
    _dt_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Backend Parameters

    We want to check the values of our configuration in the main tutorial. We need average gate durations and errors for one-qubit and two-qubit gates. The backend stores this information for each available instruction; we just need to extract that information to a `DataFrame` and compute the average (across qubit sets) of each gate type.

    First, we extract the information from the backend's instruction set:
    """)
    return


@app.function
def instruction_data(inst, backend):
    op, qtuple = inst
    data = {
        'name': op.name,
        'qubits': qtuple,
    }
    props = backend.target[op.name].get(qtuple) if qtuple else None
    if props:
        data.update({
            'duration': props.duration,
            'error': props.error,
        })

    return data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A quick sanity check that this works for a single instruction:
    """)
    return


@app.cell
def _():
    _inst = BACKEND.target.instructions[300]
    instruction_data(_inst, BACKEND)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we can apply it to every instruction and create a `DataFrame`:
    """)
    return


@app.cell
def _():
    op_df = pd.DataFrame(
        [instruction_data(inst, BACKEND) for inst in BACKEND.target.instructions]
    )
    op_df
    return (op_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For some reason (I've looked and can't find a straight answer), some [ECR gates](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.ECRGate) have error rates of 1, presumably because these gates aren't allowed in that direction.
    """)
    return


@app.cell
def _(op_df):
    op_df.sort_values('error', ascending=False)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To get the average gate durations and errors, we:

    1. Filter out spurious gates
    2. [`melt`](https://pandas.pydata.org/docs/reference/api/pandas.melt.html) so we can group by gate type and metric
    3. Compute the mean for each group
    4. "un-melt" using [`pivot`](https://pandas.pydata.org/docs/reference/api/pandas.pivot.html) to get back one column for each metric
    """)
    return


@app.cell
def _(op_df):
    op_table = op_df[
        op_df['error'] < 0.5 # ignore weird gate data
    ].melt(
        id_vars=['name'],
        value_vars=['duration','error'],
        var_name='variable',
        value_name='value',
    ).groupby(
        ['name', 'variable']
    ).agg(
        mean=('value', 'mean'),
    ).reset_index().pivot(
        index='name',
        columns='variable',
        values='mean',
    )
    op_table
    return


if __name__ == "__main__":
    app.run()
