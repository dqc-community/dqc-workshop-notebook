# DQC Workshop Notebook

This repository contains the workshop notebooks and web application for the DQC (Distributed Quantum Computing) tutorial.

## Quick Start

### Quarto Website (Recommended)

The tutorial is available as a Quarto website with light/dark theme support:

```bash
uv run quarto preview
```

To render to static HTML:

```bash
uv run quarto render
```

### Marimo App

Dependencies are bundled with the notebook using [Marimo's sandbox mode](https://docs.marimo.io/guides/package_management/inlining_dependencies/).

```bash
marimo edit --sandbox tutorial_marimo.py
```

Or using uv:

```bash
uv run marimo edit --sandbox tutorial_marimo.py
```

## Requirements

- Python 3.10
- [marimo](https://marimo.io)
- [uv](https://github.com/astral-sh/uv)
- [Quarto](https://quarto.org) (for website rendering)

## Development

For development, you can sync dependencies with uv:

```bash
uv sync
```

Optional comparison extras:

```bash
uv sync --extra comparison
```
