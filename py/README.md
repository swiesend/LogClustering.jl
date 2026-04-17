# logclustering-py

Python companion for [`LogClustering.jl`](../). Hosts the HuggingFace-only
pieces of the pipeline — embedders (BGE / GTE / E5 / Nomic / Jina),
UMAP-learn, HDBSCAN, and LILAC — and is called from Julia via
`PythonCall.jl`.

## Managed with [uv](https://docs.astral.sh/uv/)

```sh
cd py
uv sync            # create .venv and install locked deps
uv sync --group dev  # include dev tools (pytest, ruff)
```

`uv.lock` pins exact versions for reproducible environments.

## Point Julia's `PythonCall` at this environment

`PythonCall` defaults to `CondaPkg`; we disable that and hand it the uv
virtualenv directly. From the repo root:

```sh
cd py && uv sync && cd ..
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE="$(pwd)/py/.venv/bin/python"
julia --project
```

On Windows the executable is at `py\.venv\Scripts\python.exe`.
