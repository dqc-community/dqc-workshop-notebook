# DQC Workshop Notebook

This repository contains the workshop notebooks and marimo app for the DQC tutorial.

## Setup (recommended)

This project is configured to use the pip installer script:
`install_tutorial_dependencies_pip.py`

1. Create and activate an environment.
2. Run the installer script from that active environment.

Example (bash):

```bash
python -m venv tutorial-venv
source tutorial-venv/bin/activate
python install_tutorial_dependencies_pip.py
```

Windows (PowerShell):

```powershell
python -m venv tutorial-venv
.\tutorial-venv\Scripts\Activate.ps1
python install_tutorial_dependencies_pip.py
```

macOS (Homebrew):

```bash
brew install python
python3 -m venv tutorial-venv
source tutorial-venv/bin/activate
python install_tutorial_dependencies_pip.py
```

Optional comparison extras:

```bash
python install_tutorial_dependencies_pip.py --comparison
```

## Run

Run the marimo tutorial:

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