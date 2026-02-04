# Agent Guide for PyLops

## Project structure (high level)
- `pylops/`: core library source code.
- `pytests/`: pytest test suite (discovered via `pytests/*.py`).
- `docs/`: Sphinx documentation sources and build output.
- `examples/`: runnable examples and small scripts.
- `tutorials/`: tutorial notebooks/scripts used in docs.
- `testdata/`: data used by tests/examples.
- `build/`, `pylops.egg-info/`: build artifacts (usually not edited).

## Common commands (prefer Makefile targets)
- `make tests`: run CPU tests with pytest.
- `make tests_cpu_ongpu`: run CPU tests on GPU systems (disables CuPy usage).
- `make tests_gpu`: run GPU tests (requires CuPy; sets TEST_CUPY_PYLOPS=1).
- `make lint`: run `flake8` on `docs/`, `examples/`, `pylops/`, `pytests/`, `tutorials/`.
- `make typeannot`: run `mypy` on `pylops/`.
- `make doc`: clean docs build artifacts and build HTML docs.
- `make docupdate`: rebuild HTML docs without a clean.
- `make servedoc`: serve docs from `docs/build/html/`.
- `make coverage`: run tests with coverage and build HTML/XML reports.

## CI (GitHub Actions)
- `build.yaml`: main test matrix on `ubuntu-latest` and `macos-latest` for Python `3.10`–`3.13`.
  - Installs dev requirements (CPU vs ARM variants), pyfftw, torch, then runs `pytest`.
- `build-mkl.yaml`: Ubuntu-only tests using Intel oneMKL via conda (Python `3.11`–`3.13`).
- `buildcupy.yaml`: self-hosted GPU job running CuPy tests (`TEST_CUPY_PYLOPS=1`).
- `flake8.yaml`: flake8 linting on `pylops/` with the same ignores/max line length as `setup.cfg`.
- `codacy-coverage-reporter.yaml`: runs tests with coverage and uploads `coverage.xml` to Codacy.
- `deploy.yaml`: builds and publishes the package to PyPI on GitHub release publish.

## Code style and typing guidelines
- Follow `flake8` rules from `setup.cfg`:
  - Max line length is 88.
  - Ignored rules: `E203`, `E501`, `W503`, `E402`.
  - `__init__.py` allows unused imports (`F401`, `F403`, `F405`).
- Prefer clear, PEP 8–style Python, keeping imports at the top unless there is a strong reason.
- Type checking uses `mypy` with `numpy.typing` plugin enabled.
  - `mypy` runs on `pylops/` only via `make typeannot`.
  - Several third-party modules are configured with `ignore_missing_imports = True` in `setup.cfg`.
- When changing APIs or adding new modules, add or update tests under `pytests/`.

## Notes for agents
- Use Makefile targets rather than invoking tools directly where possible.
- Keep changes focused; avoid editing build artifacts or generated docs.
