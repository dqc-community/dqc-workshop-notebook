# DQC Workshop Notebook

This repository contains the workshop notebooks and marimo app for the DQC tutorial.

## Quick Start

Dependencies are bundled with the notebook using [Marimo's sandbox mode](https://docs.marimo.io/guides/package_management/inlining_dependencies/).

```bash
UV_INDEX_STRATEGY=unsafe-best-match marimo edit --sandbox tutorial_marimo.py
```

Or using uv:

```bash
UV_INDEX_STRATEGY=unsafe-best-match uv run marimo edit --sandbox tutorial_marimo.py
```

## Requirements

- Python 3.10+
- [marimo](https://marimo.io)
- [uv](https://github.com/astral-sh/uv)

## Development

For development, you can sync dependencies with uv:

```bash
uv sync
```

Optional comparison extras:

```bash
uv sync --extra comparison
```

## Run

Run the marimo tutorial (in development mode):

```bash
marimo edit tutorial_marimo.py
```

Run the Jupyter notebook:

```bash
jupyter lab tutorial.ipynb
```

If you use classic notebook UI:

```bash
jupyter notebook tutorial.ipynb
```
