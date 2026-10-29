"""Execute the EDA notebook in place using this project's Python interpreter."""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "exploration.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=600, resources={"metadata": {"path": str(root)}}
    )
    # Ensure execution uses the invoking interpreter rather than a stale user kernelspec.
    client.create_kernel_manager()
    client.km.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Executed and validated {path}")


if __name__ == "__main__":
    main()
