import importlib
import platform
import sys
from importlib import metadata


PACKAGES = {
    "jupyter": "jupyter",
    "notebook": "notebook",
    "ipykernel": "ipykernel",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scikit-learn": "scikit-learn",
    "statsmodels": "statsmodels",
    "stumpy": "stumpy",
    "numba": "numba",
    "hmmlearn": "hmmlearn",
    "yfinance": "yfinance",
    "pyarrow": "pyarrow",
    "fastparquet": "fastparquet",
    "tqdm": "tqdm",
    "plotly": "plotly",
    "tslearn": "tslearn",
    "joblib": "joblib",
    "networkx": "networkx",
    "ruptures": "ruptures",
}

IMPORTS = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "sklearn",
    "stumpy",
    "numba",
    "hmmlearn",
    "pyarrow",
    "yfinance",
    "tqdm",
    "plotly",
    "tslearn",
    "ruptures",
]


def package_version(distribution_name):
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main():
    print("Python executable:")
    print(f"  {sys.executable}")
    print()

    print("Python version:")
    print(f"  {platform.python_version()}")
    print()

    print("Package versions:")
    for label, distribution_name in PACKAGES.items():
        print(f"  {label}: {package_version(distribution_name)}")
    print()

    print("Import checks:")
    failed = []
    for module_name in IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"  OK   {module_name}")
        except Exception as exc:
            failed.append(module_name)
            print(f"  FAIL {module_name}: {exc.__class__.__name__}: {exc}")

    if failed:
        print()
        print(f"Failed imports: {', '.join(failed)}")
        raise SystemExit(1)

    print()
    print("All requested imports succeeded.")


if __name__ == "__main__":
    main()
