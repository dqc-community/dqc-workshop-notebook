from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TEST_PYPI_INDEX_URL = "https://test.pypi.org/simple/"
PYPI_EXTRA_INDEX_URL = "https://pypi.org/simple/"
BASE_PACKAGE = "bosonic-sdk-felix"
COMPARISON_EXTRA = "comparison"
RUNTIME_PACKAGES = [
    "ipykernel",
    "marimo",
    "pandas",
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def install_packages(
    python_exe: str,
    *,
    include_comparison: bool,
    index_url: str,
    extra_index_url: str,
) -> str:
    package_name = BASE_PACKAGE
    if include_comparison:
        package_name = f"{BASE_PACKAGE}[{COMPARISON_EXTRA}]"

    run([python_exe, "-m", "pip", "install", "-U", "pip"])
    run(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "--index-url",
            index_url,
            "--extra-index-url",
            extra_index_url,
            "-U",
            package_name,
            *RUNTIME_PACKAGES,
        ]
    )
    return package_name


def register_kernel(python_exe: str, *, kernel_name: str, display_name: str) -> None:
    run(
        [
            python_exe,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            kernel_name,
            "--display-name",
            display_name,
        ]
    )


def verify_imports(python_exe: str) -> None:
    verification = (
        "from bosonic_converters import CircuitConverters; "
        "from bosonic_sdk import BosonicDistributor, Simulator; "
        "print('BosonicDistributor ->', BosonicDistributor); "
        "print('Simulator ->', Simulator); "
        "print('CircuitConverters ->', CircuitConverters)"
    )
    run([python_exe, "-c", verification])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install workshop dependencies from TestPyPI into the current Python environment."
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Install optional comparison backends via bosonic-sdk-felix[comparison].",
    )
    parser.add_argument(
        "--skip-kernel",
        action="store_true",
        help="Do not register a Jupyter kernel for the current environment.",
    )
    parser.add_argument(
        "--kernel-name",
        default="tutorial-venv",
        help="Kernel name to register with Jupyter.",
    )
    parser.add_argument(
        "--display-name",
        default="tutorial-venv (Bosonic TestPyPI)",
        help="Kernel display name to register with Jupyter.",
    )
    parser.add_argument(
        "--index-url",
        default=TEST_PYPI_INDEX_URL,
        help="Primary package index URL.",
    )
    parser.add_argument(
        "--extra-index-url",
        default=PYPI_EXTRA_INDEX_URL,
        help="Fallback package index URL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exe = sys.executable

    print("Installer               :", Path(__file__).resolve())
    print("Current python          :", python_exe)
    print("Primary index           :", args.index_url)
    print("Fallback index          :", args.extra_index_url)

    package_name = install_packages(
        python_exe,
        include_comparison=args.comparison,
        index_url=args.index_url,
        extra_index_url=args.extra_index_url,
    )

    if not args.skip_kernel:
        register_kernel(
            python_exe,
            kernel_name=args.kernel_name,
            display_name=args.display_name,
        )

    verify_imports(python_exe)

    print()
    print("Setup complete.")
    print("  Installed package     ->", package_name)
    print("  Current python        ->", python_exe)
    if args.skip_kernel:
        print("  Jupyter kernel        -> skipped")
    else:
        print("  Jupyter kernel        ->", args.kernel_name)
        print("  Kernel display name   ->", args.display_name)
    print()
    print("Next step:")
    print("  marimo edit tutorial_marimo.py")


if __name__ == "__main__":
    main()
