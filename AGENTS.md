# Repository Guidelines

## Project Structure & Module Organization

KANLux is a Julia/Lux implementation of Kolmogorov-Arnold Networks, ported from PyKAN.

- `src/` contains the package source. `src/KANLux.jl` loads the implementation modules in phase order: `bspline.jl`, `symbolic_lib.jl`, `data.jl`, `KANLayer.jl`, `SymbolicKANLayer.jl`, `MultKAN.jl`, `ops_internal.jl`, and `ops.jl`.
- `test/` contains `runtests.jl`, one `test_<module>.jl` file per component, shared helpers in `test_infra.jl`, and PyKAN-generated `.npy` fixtures under `test/reference/`.
- `Project.toml` and `Manifest.toml` define dependencies, compatibility, and the lockfile. `README.md` documents installation, the public API, and quick-start examples.

## Build, Test, and Development Commands

Run these commands from this directory using Julia 1.9 or newer.

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` installs and resolves dependencies.
- `julia --project=. -e 'using Pkg; Pkg.test()'` runs the complete test suite with the test-only dependencies.
- `julia --project=. -e 'using KANLux, Test; include("test/test_kanlayer.jl")'` runs a single component test.
- `julia --project=. -e 'using KANLux'` performs a basic import smoke test.

## Coding Style & Naming Conventions

- Use four-space indentation and no tabs.
- Follow Julia conventions: types and modules use `CamelCase` (`MultKAN`, `KANLayer`), functions and variables use `snake_case`, and internal helpers are prefixed with `_`.
- Add docstrings for exported types and functions. Keep comments aligned with the existing phase-based section headers.
- There is no configured formatter or linter, so match the surrounding code and avoid unrelated formatting changes.

## Testing Guidelines

- Tests use Julia's `Test` package and nested `@testset` blocks.
- Add or update tests for bug fixes and API changes, and keep numerical comparisons consistent with existing tolerances such as `1e-6`.
- `test_infra.jl` provides shared loaders, gradient checks, state round-trip checks, and the deterministic RNG.
- Regenerate `test/reference/*.npy` via `test/generate_reference.py` only when the PyKAN reference contract intentionally changes.

## Commit & Pull Request Guidelines

This working directory has no available Git history, so project-specific commit conventions cannot be extracted. Use Conventional Commits when creating history (`feat:`, `fix:`, `test:`, `docs:`). Keep pull requests focused, explain the motivation and verification performed, link related issues, and include test results or reproduction steps where relevant.
