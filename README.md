# KANLux.jl

Julia/Lux.jl implementation of Kolmogorov-Arnold Networks (KANs), migrated
from [PyKAN](https://github.com/KindXiaoming/pykan). The package provides the
B-spline mathematics, symbolic function registry, data utilities, layer
containers, and structural operations needed to build and evaluate KAN models.
Training loops, plotting, checkpointing, and symbolic auto-regression are
intentionally left to the application layer.

## Installation

Julia 1.9 or newer is required.

```julia
using Pkg
Pkg.develop(path="KANLux")
```

After the package has been registered in the General registry, install a
released version with:

```julia
using Pkg
Pkg.add("KANLux")
```

The package depends on `Lux`, `Random`, `LinearAlgebra`, and `Statistics`.
Test dependencies (`Zygote`, `FiniteDifferences`, `LuxTestUtils`, `NPZ`) are
declared only for the test target.

## Quick start

Build a two-layer KAN and evaluate it on synthetic data:

```julia
using KANLux, Lux, Random

rng = Xoshiro(42)

# width = [input, hidden, output]
model = MultKAN([2, 3, 1]; grid=3, k=3)

# Initialise trainable parameters (`ps`) and non-trainable state (`st`).
ps, st = Lux.setup(rng, model)

x = randn(rng, 16, 2)
y, st = model(x, ps, st)

println(size(y))  # (16, 1)
```

`KANLux.create_dataset` can generate train/test tensors for a scalar-valued
function:

```julia
using KANLux

f = x -> sum(x .^ 2; dims=2)
train_input, train_label, test_input, test_label =
    create_dataset(f, 2, (-1.0, 1.0), 100, 50, 1)
```

## Public API

| Export | Description |
|--------|-------------|
| `B_batch` | Cox-de Boor basis evaluation |
| `coef2curve` | Convert B-spline coefficients to curve values |
| `curve2coef` | Least-squares curve-to-coefficient conversion |
| `extend_grid` | Add uniform ghost nodes to a B-spline grid |
| `SYMBOLIC_LIB` | Registry of 27 symbolic functions |
| `create_dataset` | Synthetic dataset generator |
| `KANLayer` | Numerical B-spline activation layer |
| `SymbolicKANLayer` | Closed-form symbolic activation layer |
| `fix_symbolic!` | Assign a symbolic function to an edge |
| `MultKAN` | Composite numerical + symbolic KAN container |
| `prune` | Structural edge/node pruning |
| `refine` | B-spline grid refinement |

Layer instances implement the Lux API:

```julia
using KANLux, Lux, Random

rng = Xoshiro(0)
layer = KANLayer(2, 4, 5, 3)
ps, st = Lux.setup(rng, layer)
y, st = layer(randn(rng, 8, 2), ps, st)
```

## Running tests

```julia
using Pkg
Pkg.activate("KANLux")
Pkg.test()
```

The test suite includes forward-value checks against PyKAN-generated reference
data, layer/container state-transfer checks, gradient-path checks, edge cases,
and global Zygote-vs-finite-difference gradient cross-validation.
