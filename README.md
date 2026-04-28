# DQC Workshop Notebook

This repository contains materials for the DQC tutorial workshop.

## Quick Start

Dependencies are bundled with the notebook using [Marimo's sandbox mode](https://docs.marimo.io/guides/package_management/inlining_dependencies/). This means you can run the tutorial directly from GitHub (no clone required!) using the following command:

```bash
uvx marimo edit --sandbox https://github.com/dqc-community/dqc-workshop-notebook/blob/main/tutorial.py
```

## Requirements

- Python 3.10+
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
uv run marimo edit --sandbox tutorial.py
```

Run the Jupyter notebook:

```bash
uv run jupyter lab tutorial.ipynb
```

If you use classic notebook UI:

```bash
uv run jupyter notebook tutorial.ipynb
```
