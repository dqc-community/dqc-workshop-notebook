#!/usr/bin/env python3
"""
transpile-all.py

Generate a corpus of transpiled GHZ circuits for FakeSherbrooke and persist
them to disk as QASM files.

Usage
-----
    python transpile-all.py --n-list 3-21 --m 1000 --seed 1234 --n-workers 8

    python transpile-all.py --n-list 3-127 --m 1000  # full sweep

Exit codes
----------
    0  – completed successfully
    1  – argument error
    2  – internal error (see stderr)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import re
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic as _now

import numpy as np
import qiskit
import qiskit.qasm2 as qasm2
import qiskit_ibm_runtime
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = _SCRIPT_DIR / "transpiled-circuits"


# ---------------------------------------------------------------------------
# GHZ circuit factory
# ---------------------------------------------------------------------------

def ghz_circuit(n: int, measure: bool = True) -> qiskit.QuantumCircuit:
    """Create a GHZ state circuit on n qubits (measured by default)."""
    qc = qiskit.QuantumCircuit(n, n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    if measure:
        qc.measure(range(n), range(n))
    return qc


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _hash_qasm(qasm_str: str) -> str:
    """SHA-256 hex digest (first 16 hex chars) of a QASM string."""
    return hashlib.sha256(qasm_str.encode()).hexdigest()[:16]


def _next_counter(subdir: Path, h: str) -> int:
    """
    Return the next counter value for hash prefix ``h`` in ``subdir``.

    Scans files matching ``<h>-*.qasm``, extracts the 4-digit counter from
    each filename, and returns 1 + the maximum counter found.
    Returns 1 if no matching files exist.
    """
    files = sorted(subdir.glob(f"{h}-*.qasm"))
    if not files:
        return 1
    counters = []
    for fp in files:
        m = re.search(r"-(\d{4})\.qasm$", fp.name)
        if m:
            counters.append(int(m.group(1)))
    return max(counters) + 1 if counters else 1


def write_qasm(subdir: Path, qasm_str: str) -> Path | None:
    """
    Write a QASM string to the dataset directory.

    Naming convention: ``HASH-xxxx.qasm`` where xxxx is a zero-padded
    4-digit counter tracking how many times this hash has been observed.

    Exactly one file per unique hash is written to disk: the first
    transpilation to produce a hash writes ``HASH-0001.qasm``; all
    subsequent transpilations with the same hash are counted by incrementing
    the counter but no new file is written.
    """
    h = _hash_qasm(qasm_str)
    counter  = _next_counter(subdir, h)
    filename = f"{h}-{counter:04d}.qasm"
    if counter == 1:
        # First time this hash has been seen — write the file.
        (subdir / filename).write_text(qasm_str)
    else:
        # Subsequent occurrences: increment the counter (tracked via the
        # filename pattern) but do NOT write another QASM file.
        oldname = f"{h}-{counter-1:04d}.qasm"
        os.rename(subdir / oldname, subdir / filename)
    return None


# ---------------------------------------------------------------------------
# Transpilation worker (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _transpile_one(
    n: int,
    seed: int,
    optimization_level: int,
) -> tuple[str, str, int, float]:
    """
    Transpile one GHZ circuit for FakeSherbrooke and return metadata.

    The backend is constructed here (not passed as an argument) so that it
    is not pickled and sent to every worker process — each worker builds its
    own instance.

    The seed is converted to a Python int here because
    np.random.default_rng(seed).integers() returns numpy.int64, which Qiskit
    validates as not a native int ("Expected non-negative integer").

    Returns
    -------
    qasm_str     : OpenQASM 2.0 string of the transpiled circuit.
    circuit_hash : SHA-256 prefix of qasm_str.
    """
    backend = qiskit_ibm_runtime.fake_provider.FakeSherbooke()
    rng = np.random.default_rng(seed)
    transpiled = qiskit.transpile(
        ghz_circuit(n, measure=True),
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=int(rng.integers(0, 2**31)),
    )

    # Qiskit 2.x removed QuantumCircuit.qasm(); use qiskit.qasm2.dumps() instead.
    return qiskit.qasm2.dumps(transpiled)

# ---------------------------------------------------------------------------
# Per-trial worker (module-level for pickling)
#
# All state that was previously captured by a local closure is explicitly
# passed in via the task dict so that ProcessPoolExecutor can serialise it.
# ---------------------------------------------------------------------------

def _trial_worker(task: dict) -> dict:
    """
    Worker function passed to ProcessPoolExecutor.

    Parameters
    ----------
    task : dict
        n, seed, optimization_level  — circuit/transpiler parameters
        subdir, lockpath            — string paths to the output directory and lock file
        n_qubits                   — qubit count, included in the result dict

    Returns
    -------
    dict with n, hash, depth, duration_ns.
    """
    qasm_str = _transpile_one(
        n=task["n"],
        seed=task["seed"],
        optimization_level=task["optimization_level"],
    )

    # ── critical section: update counter & write QASM ──────────────────────
    with open(Path(task["lockpath"]), "r+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            write_qasm(Path(task["subdir"]), qasm_str)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    # ── end critical section ───────────────────────────────────────────────

    return dict(n=task["n"], hash=_hash_qasm(qasm_str))


# ---------------------------------------------------------------------------
# Per-trial worker (module-level so it can be pickled)
#
# All state that was previously captured by a local closure is explicitly
# passed in via the task dict so that ProcessPoolExecutor can serialise it.
# ---------------------------------------------------------------------------

def _trial_worker(task: dict) -> dict:
    """
    Worker function passed to ProcessPoolExecutor.

    Parameters
    ----------
    task : dict
        n, seed, optimization_level  — as per transpile_batch_for_n
        lockpath, subdir             — Paths to the lock file and subdirectory.
        n_qubits                     — The qubit count (n), included in the result.

    Returns
    -------
    dict with n, hash, depth, duration_ns.
    """
    n              = task["n_qubits"]
    subdir         = Path(task["subdir"])
    lockpath       = Path(task["lockpath"])

    qasm_str, circuit_hash, depth, duration_ns = _transpile_one(
        n=task["n"],
        seed=task["seed"],
        optimization_level=task["optimization_level"],
    )

    # ── critical section: update counter & write QASM ──────────────────────
    with open(lockpath, "r+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            write_qasm(subdir, circuit_hash, qasm_str)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    # ── end critical section ───────────────────────────────────────────────

    return dict(n=n, hash=circuit_hash, depth=depth, duration_ns=duration_ns)


# ---------------------------------------------------------------------------
# Batch transpilation (parallel over m only)
# ---------------------------------------------------------------------------

def transpile_batch_for_n(
    n: int,
    m: int,
    seed_base: int,
    optimization_level: int,
    dataset_dir: Path,
    n_workers: int,
) -> tuple[list[dict], dict]:
    """
    Transpile ``m`` circuits for a single qubit count ``n``, parallelised
    over ``m`` workers.

    All workers share one subdirectory.  Counter access is serialised via
    ``fcntl.flock`` on ``_counter.lock`` to prevent two workers from picking
    the same counter value.

    Returns
    -------
    (records, stats)
        records : list of dicts with keys n, hash, depth, duration_ns.
        stats   : dict with n_written, n_unique, elapsed_s.
    """
    t0 = _now()

    rng_global = np.random.default_rng(seed_base + n)
    subdir     = dataset_dir / f"n_{n:03d}"
    lockpath   = subdir / "_counter.lock"
    subdir.mkdir(parents=True, exist_ok=True)
    lockpath.touch()

    # Build task list — every captured variable is explicit so it can be pickled.
    # Note: int() is required because rng.integers returns numpy.int64, which
    # qiskit's seed_transpiler parameter rejects ("Expected non-negative integer").
    tasks = [
        dict(
            n                  = n,
            seed               = int(rng_global.integers(0, 2**31)),
            optimization_level = optimization_level,
            subdir             = str(subdir),
            lockpath           = str(lockpath),
        )
        for _ in range(m)
    ]

    actual_workers = min(n_workers or os.cpu_count() or 4, m)
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        records = list(executor.map(
            _trial_worker,
            tasks,
            chunksize=max(1, m // actual_workers),
        ))

    seen = {r["hash"] for r in records}
    stats = dict(
        n_written = len(records),
        n_unique  = len({r["hash"] for r in records}),
        elapsed_s = _now() - t0,
    )
    return records, stats


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def generate_dataset(
    n_list: list[int],
    m: int,
    seed: int,
    optimization_level: int = 3,
    dataset_dir: Path | str = DATASET_DIR,
    n_workers: int | None = None,
    progress=None,
) -> list[dict]:
    """
    Generate the full transpiled-circuit dataset.

    Loops sequentially over each ``n`` value; parallelism is applied *within*
    each ``n`` batch (parallel over ``m``).

    Parameters
    ----------
    n_list            : List of qubit counts to sweep.
    m                 : Number of transpilations per n.
    seed              : Base RNG seed.
    optimization_level: Qiskit transpiler optimisation level.
    dataset_dir       : Root of the transpiled-circuits directory.
    n_workers         : Max worker processes (applied per-n). None → os.cpu_count().
    progress          : rich.Progress instance (created internally if None).

    Returns
    -------
    List of dicts with keys: n, hash, depth, duration_ns.
    """
    from rich.progress import TaskID

    dataset_dir = Path(dataset_dir)
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    all_records: list[dict] = []
    total_n     = len(n_list)

    if progress is None:
        console  = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=32),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]({task.fields[n_val]}-qubit batch)[/dim]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
            refresh_per_second=8,
        )
        console.print()
        progress.start()
        on_exit = progress.stop
    else:
        on_exit = lambda: None

    try:
        n_task: TaskID = progress.add_task("transpiling", total=total_n, n_val="??")

        for idx, n in enumerate(n_list, start=1):
            progress.update(n_task, description=f"n={n:03d}", n_val=str(n))

            records, _stats = transpile_batch_for_n(
                n=n,
                m=m,
                seed_base=seed,
                optimization_level=optimization_level,
                dataset_dir=dataset_dir,
                n_workers=n_workers,
            )
            all_records.extend(records)

            progress.update(n_task, completed=idx, n_val=str(n))

    finally:
        on_exit()

    return all_records


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a corpus of transpiled GHZ circuits for FakeSherbrooke.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-list",
        type=str,
        default="3-21",
        metavar="SPEC",
        help="Qubit range.  '3-21' / '3,7,11' / '5-100:5' (with step). Default: 3-21.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=1000,
        metavar="M",
        help="Number of transpilations per n (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        metavar="S",
        help="RNG seed (default: 1234).",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        metavar="L",
        help="Qiskit transpiler optimisation level 0–3 (default: 3).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        metavar="W",
        help="Max worker processes per n (default: auto-detect via os.cpu_count()).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=f"Override dataset directory (default: {DATASET_DIR}).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except the final summary line.",
    )
    return parser.parse_args()


def _expand_n_list(s: str) -> list[int]:
    """Parse compact n-list spec: '3-20', '3,7,11', '3-127:5', etc."""
    result: list[int] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            range_part, step_part = token.rsplit(":", 1)
            parts = range_part.split("-", 1)
            result.extend(range(int(parts[0]), int(parts[1]) + 1, int(step_part)))
        elif "-" in token:
            parts = token.split("-", 1)
            result.extend(range(int(parts[0]), int(parts[1]) + 1))
        else:
            result.append(int(token))
    return sorted(set(result))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args    = _parse_args()
    n_list  = _expand_n_list(args.n_list)
    dataset = Path(args.dataset_dir) if args.dataset_dir else DATASET_DIR
    workers = args.n_workers or os.cpu_count() or 4

    console = Console()
    console.print()
    console.rule("[bold]transpile-all")
    console.print(
        f"  [cyan]n[/cyan]      : {n_list[0]}–{n_list[-1]}  ({len(n_list)} values)\n"
        f"  [cyan]m[/cyan]      : {args.m}\n"
        f"  [cyan]seed[/cyan]  : {args.seed}\n"
        f"  [cyan]opt[/cyan]   : {args.optimization_level}\n"
        f"  [cyan]workers[/cyan]: {workers}  [dim](per n batch)[/dim]\n"
        f"  [cyan]dataset[/cyan]: {dataset}"
    )
    console.print()

    try:
        if args.quiet:
            records = generate_dataset(
                n_list=n_list,
                m=args.m,
                seed=args.seed,
                optimization_level=args.optimization_level,
                dataset_dir=dataset,
                n_workers=workers,
                progress=None,
            )
        else:
            records = generate_dataset(
                n_list=n_list,
                m=args.m,
                seed=args.seed,
                optimization_level=args.optimization_level,
                dataset_dir=dataset,
                n_workers=workers,
            )
    except Exception as exc:
        console.print(f"[red]\n[ERROR][/red] {exc}")
        return 2

    # ── final summary ───────────────────────────────────────────────────────

    n_total  = sum(1 for _ in dataset.rglob("*.qasm") if _.name != "_counter.lock")
    n_unique = len({r["hash"] for r in records})

    console.print()
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("metric",  style="dim")
    table.add_column("value",   justify="right")
    table.add_row("records generated", str(len(records)))
    table.add_row("files on disk",     str(n_total))
    table.add_row("unique hashes",     str(n_unique))
    table.add_row("collision rate", f"{1 - n_unique/len(records):.2%}" if records else "—")
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
