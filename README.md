# KANLux.jl

> ⚠️ **Warning**: This package contains a large amount of AI-generated code
> that has not been manually reviewed by the repository author, who has not
> finished checking it yet. Anyone using or modifying this repository should
> exercise special caution.

[English](README.md) · [简体中文](README_Zh_CN.md)

Julia/Lux.jl implementation of Kolmogorov-Arnold Networks (KANs), migrated
from [PyKAN](https://github.com/KindXiaoming/pykan). The package provides the
B-spline mathematics, symbolic function registry, data utilities, layer
containers, and structural operations needed to build and evaluate KAN models.
Training loops, plotting, checkpointing, and symbolic auto-regression are
intentionally left to the application layer.

## Installation

Julia 1.10 or newer is required. Choose one of the three methods below.

### 1. From source (development)

Use this method when you have a local checkout and want to modify the code:

```julia
using Pkg
Pkg.develop(path="KANLux")
```

### 2. From the GitHub URL

Install the current `main` branch directly from the public repository:

```julia
using Pkg
Pkg.add(url="https://github.com/PeHa393/KANLux")
```

Add `rev="main"` (or a specific commit/tag) to pin a revision:

```julia
using Pkg
Pkg.add(url="https://github.com/PeHa393/KANLux", rev="main")
```

### 3. By package name (from the General registry)

```julia
using Pkg
Pkg.add("KANLux")
```

> ⚠️ **Warning**: KANLux has **not yet been uploaded to the General registry**,
> so this method is currently **not available**. Use one of the two methods
> above until registration completes.

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

The package exports twelve symbols. Unless stated otherwise, all tensors are
`Float64` Julia arrays using the same layout as PyKAN/PyTorch (`batch` as the
leading dimension).

| Export | Description |
|--------|-------------|
| `B_batch` | Cox-de Boor basis evaluation |
| `coef2curve` | Convert B-spline coefficients to curve values |
| `curve2coef` | Least-squares curve-to-coefficient conversion |
| `extend_grid` | Add uniform ghost nodes to a B-spline grid |
| `SYMBOLIC_LIB` | Registry of 27 symbolic functions (29 keys) |
| `create_dataset` | Synthetic dataset generator |
| `KANLayer` | Numerical B-spline activation layer |
| `SymbolicKANLayer` | Closed-form symbolic activation layer |
| `fix_symbolic!` | Assign a symbolic function to an edge |
| `MultKAN` | Composite numerical + symbolic KAN container |
| `prune` | Structural edge/node pruning |
| `refine` | B-spline grid refinement |

Two further helpers are public but not exported: `KANLux.fit_symbolic_params`
(the a/b grid search used by `fix_symbolic!`) and `KANLux.silu` (the default
base function).

Every layer and container implements the Lux layer interface:

```julia
using KANLux, Lux, Random

rng = Xoshiro(0)
layer = KANLayer(2, 4, 5, 3)
ps, st = Lux.setup(rng, layer)           # ps = trainable, st = non-trainable
y, st = layer(randn(rng, 8, 2), ps, st)  # forward returns (output, new state)
```

### Array layout

With `G` grid intervals, order `k`, and an extended grid of `S = G + 2k + 1`
nodes:

| Array | Shape |
|-------|-------|
| `x`, `x_eval` (inputs) | `(batch, in_dim)` |
| `grid` | `(in_dim, S)` |
| `coef` | `(in_dim, out_dim, G + k)` |
| `y_eval` (target curves) | `(batch, in_dim, out_dim)` |
| `B_batch(...)` result | `(batch, in_dim, G + k)` |
| `coef2curve(...)` result | `(batch, in_dim, out_dim)` |
| `curve2coef(...)` result | `(in_dim, out_dim, G + k)` |

### B-spline primitives

```julia
B_batch(x, grid, k)
coef2curve(x_eval, grid, coef, k)
curve2coef(x_eval, y_eval, grid, k)
extend_grid(grid, k_extend=0)
```

`B_batch` evaluates the Cox-de Boor basis of order `k`; `coef2curve` is
PyKAN's `einsum('ijk,jlk->ijl')`. `curve2coef` solves a per-edge least-squares
problem to invert `coef2curve` — it is a structural-operation helper (used by
`refine`), **not** part of the differentiable training forward pass.
`extend_grid` appends `k_extend` uniform ghost nodes to each end of every row
using the original grid step.

```julia
grid = extend_grid(repeat(reshape(collect(range(-1.0, 1.0; length=6)), 1, :), 2, 1), 3)
x = randn(16, 2)
coef = randn(2, 3, 8)                 # (in_dim, out_dim, G + k) with G=5, k=3
y = coef2curve(x, grid, coef, 3)      # (16, 2, 3)
coef_rec = curve2coef(x, y, grid, 3)  # (2, 3, 8)
```

### `SYMBOLIC_LIB`

`SYMBOLIC_LIB` is a `Dict{String, Tuple{Function, Int, Function}}`. Each value
is `(forward_fn, complexity, singularity_fn)`; `complexity` is a non-negative
integer used for symbolic-regression preference, and `singularity_fn(x, y_th)`
returns a finite, threshold-limited value at singular points (used when
`singularity_avoiding=true`).

The 29 keys cover 27 distinct functions (`x^0.5` aliases `sqrt`, `1/x^0.5`
aliases `1/sqrt(x)`):

```text
x, x^2, x^3, x^4, x^5,
1/x, 1/x^2, 1/x^3, 1/x^4, 1/x^5,
sqrt, x^0.5, x^1.5, 1/sqrt(x), 1/x^0.5,
exp, log, abs, sin, cos, tan, tanh, sgn,
arcsin, arccos, arctan, arctanh,
gaussian, 0
```

```julia
SYMBOLIC_LIB["sin"][1](0.5)           # forward value
SYMBOLIC_LIB["sin"][2]                # complexity = 2
SYMBOLIC_LIB["tan"][3](pi / 2, 10.0)  # finite singularity-avoiding value
```

### `create_dataset`

```julia
create_dataset(f, n_var, ranges=(-1.0, 1.0),
               train_num=1000, test_num=1000, seed=0)
```

`f` receives an `(N, n_var)` matrix and returns a scalar, vector, or matrix
(vectors are reshaped to `(N, 1)`). `ranges` may be a common `(lo, hi)` range,
an `n_var × 2` matrix, or a vector of per-variable tuples/vectors. Returns
`(train_input, train_label, test_input, test_label)`.

```julia
train_x, train_y, test_x, test_y =
    create_dataset(x -> sum(x .^ 2; dims=2), 2, (-1.0, 1.0), 100, 50, 1)
```

### `KANLayer`

```julia
KANLayer(in_dim, out_dim, num, k;
         grid_eps=0.02, grid_range=(-1.0, 1.0), base_fun=silu,
         sp_trainable=true, sb_trainable=true)
```

The numerical B-spline edge layer. Each `(input, output)` edge computes
`scale_base * base_fun(x) + scale_sp * spline(x)`, masked by `st.mask`, then
sums over inputs. `ps` holds `coef` plus the trainable scales; `st` holds
`grid`, `mask`, and any non-trainable scales.

```julia
layer = KANLayer(2, 4, 5, 3)   # in=2, out=4, grid intervals=5, order=3
ps, st = Lux.setup(rng, layer)
# ps.coef       :: (2, 4, 8)      (in_dim, out_dim, G + k)
# ps.scale_base :: (2, 4)
# ps.scale_sp   :: (2, 4)
# st.grid       :: (2, 12)        (S = 5 + 2*3 + 1)
# st.mask       :: (2, 4)
y, _ = layer(randn(rng, 8, 2), ps, st)  # (8, 4)
```

With `sp_trainable=false` / `sb_trainable=false` the corresponding scale is
stored in `st` instead of `ps`.

### `SymbolicKANLayer`, `fix_symbolic!`, `fit_symbolic_params`

```julia
SymbolicKANLayer(in_dim, out_dim)

fix_symbolic!(layer, i, j, name; random=false, a_range=(-10.0, 10.0),
              b_range=(-10.0, 10.0), grid_number=21)
fix_symbolic!(layer, i, j, name, x, y; random=false, ...)

KANLux.fit_symbolic_params(x, y, fun; a_range=(-10.0, 10.0),
                           b_range=(-10.0, 10.0), grid_number=21)
```

The symbolic layer applies a closed-form edge transform `c * f(a*x + b) + d`.
`ps` holds `affine :: (out_dim, in_dim, 4)` with entries `[a, b, c, d]`; `st`
holds `mask`, `funs`, and `funs_name`. Unfixed edges emit zero.

`fix_symbolic!(layer, i, j, name)` assigns `SYMBOLIC_LIB[name]` to edge
`(input i, output j)`. Without `x`/`y` the affine is the identity
`[1, 0, 1, 0]` (or random when `random=true`); with `x`/`y` it is fitted to
`y ≈ c*f(a*x+b)+d` by `fit_symbolic_params` (a/b grid search + c/d least
squares). The layer is mutated in place; enable the edge by setting
`layer.mask[j, i] = 1.0`.

```julia
layer = SymbolicKANLayer(2, 2)
fix_symbolic!(layer, 1, 1, "sin")   # input 1 -> output 1
layer.mask[1, 1] = 1.0
ps, st = Lux.setup(rng, layer)
y, _ = layer(randn(rng, 8, 2), ps, st;
             singularity_avoiding=true, y_th=10.0)
```

### `MultKAN`

```julia
MultKAN(width; grid=3, k=3, mult_arity=2, base_fun=silu,
        symbolic_enabled=true, grid_eps=0.02, grid_range=(-1.0, 1.0),
        sp_trainable=true, sb_trainable=true)
```

Composite container stacking `depth = length(width) - 1` numerical layers,
symbolic layers, sub-node affine transforms, optional multiplication nodes, and
node affine transforms. `width` entries are integer neuron counts or
`[n_sum, n_mult]` pairs. `grid`/`k` may be scalars or per-layer vectors;
`mult_arity` may be a scalar (homogeneous) or a vector matching `width`.

```julia
model = MultKAN([2, 3, 1]; grid=3, k=3)
ps, st = Lux.setup(rng, model)
# ps.act_fun[d]      :: KANLayer parameters (per layer)
# ps.symbolic_fun[d] :: (affine=...) (per layer)
# ps.node_bias, ps.node_scale        :: (per layer)
# ps.subnode_bias, ps.subnode_scale  :: (per layer)
# st.act_fun[d].grid/.mask, st.symbolic_fun[d].mask/.funs/.funs_name
y, _ = model(randn(rng, 16, 2), ps, st)
```

Assign symbolic edges through the embedded layers:

```julia
model = MultKAN([2, 3, 1]; grid=3, k=3, symbolic_enabled=true)
fix_symbolic!(model.symbolic_fun[1], 1, 2, "x^2")  # layer 1, input 1 -> output 2
model.symbolic_fun[1].mask[2, 1] = 1.0
ps, st = Lux.setup(rng, model)
```

Multiplication nodes use `[n_sum, n_mult]` width pairs; `mult_arity` controls
how many sub-node columns are multiplied into each product node.

### Structural operations: `prune` and `refine`

```julia
prune(model, ps, st, x; node_th=1e-2, edge_th=3e-2)
prune(model, x; node_th=1e-2, edge_th=3e-2, rng=Random.default_rng())

refine(model, ps, st, x, new_grid)
refine(model, x, new_grid; rng=Random.default_rng())
```

Both return `(model=..., ps=..., st=...)` and leave the inputs unchanged.

`prune` scores hidden nodes/edges from the sample `x`, removes addition nodes
below `node_th`, and zeroes edges below `edge_th`. Input/output and
multiplication nodes are always kept. `refine` increases the B-spline grid
resolution (a single integer or a per-layer vector) while re-fitting the old
curves with `curve2coef`, preserving the learned function; symbolic layers and
affine parameters are copied unchanged.

```julia
model = MultKAN([2, 3, 1]; grid=3, k=3)
ps, st = Lux.setup(rng, model)
x = randn(rng, 32, 2)

pruned = prune(model, ps, st, x; node_th=1e-2, edge_th=3e-2)
y = pruned.model(x, pruned.ps, pruned.st)[1]

refined = refine(model, ps, st, x, 10)   # or [10, 10] per layer
```

## Differences from PyKAN

KANLux ports PyKAN's *differentiable core* and verifies it numerically against
PyKAN-generated `.npy` references. It is not a drop-in Julia mirror of the full
`MultKAN` class: PyKAN's stateful convenience methods belong to the application
layer by design.

### Ported

- B-spline mathematics (`B_batch`, `coef2curve`, `curve2coef`, `extend_grid`).
- The symbolic registry `SYMBOLIC_LIB` and its singularity protection.
- `create_dataset`, `KANLayer`, `SymbolicKANLayer`, `fix_symbolic!`, and the
  `MultKAN` container with numerical + symbolic branches and multiplication
  nodes.
- Structural operations `prune` and `refine` (PyKAN's grid-update step).

### Not ported (application-layer responsibilities)

The following PyKAN `kan/MultKAN.py` methods are intentionally absent from
`src/` and must be implemented by the application:

| PyKAN method | Application-layer equivalent |
|--------------|------------------------------|
| `fit()` | Your own Lux training loop |
| `plot()` | Makie.jl (+ GraphMakie.jl) |
| `save_ckpt` / `load_ckpt` | BSON.jl / JLD2.jl over `(ps, st)` |
| `auto_symbolic()` | Symbolic-regression search loop |
| `symbolic_formula()` | SymPy.jl / Symbolics.jl assembly |

#### Training loop

`Lux.setup` produces `(ps, st)`. Losses must be differentiated with respect to
`ps` only — `st` (`grid`, `mask`, `funs`, …) is non-trainable state. The
idiomatic loop is `Training.TrainState` + `Training.single_train_step!` with an
`Optimisers.jl` rule (`Optimisers` is already a dependency of Lux; declare it,
and Zygote or Enzyme, in your application project):

```julia
# application layer
using KANLux, Lux, Random, Zygote

rng = Xoshiro(42)
model = MultKAN([2, 3, 1]; grid=3, k=3)
ps, st = Lux.setup(rng, model)

x = randn(rng, 32, 2)
target = randn(rng, 32, 1)

loss(p) = sum(abs2, model(x, p, st)[1] .- target)
grads = Zygote.gradient(loss, ps)[1]     # differentiate w.r.t. ps only
```

PyKAN's grid-update schedule (train → `refine` → `curve2coef`) is expressed
with the exported `refine`/`curve2coef`. PyKAN-style LBFGS requires adding
`Optim.jl` or `Nonconvex.jl` at the application layer. Both AD backends are
validated: **Zygote** (full coverage) and **Enzyme** (reverse-mode smoke test),
including finite-difference cross-checks.

#### Plotting

Use `Makie.jl` (+ `GraphMakie.jl` for the KAN graph). Everything PyKAN's
`plot()` needs is already exposed:

- `st.act_fun[d].mask` — edge on/off
- `st.act_fun[d].grid` and `ps.act_fun[d].coef` — spline curves
- `st.symbolic_fun[d].funs_name` — symbolic edge labels
- `model.width` — graph structure

#### Checkpointing

Use `BSON.jl` or `JLD2.jl` to save/load `(ps, st)`. Note that
`st.symbolic_fun[d].funs::Array{Function,2}` cannot be serialized directly —
persist `funs_name` + `affine` + `mask` and rebuild the function arrays from
`SYMBOLIC_LIB` on load.

#### Symbolic auto-regression

The package already provides the PyKAN building blocks:

- `SYMBOLIC_LIB` with complexity scores,
- `KANLux.fit_symbolic_params` (a/b grid search + c/d least squares),
- `fix_symbolic!` (edge assignment).

The application layer implements `auto_symbolic` (fit every library candidate
per edge, pick by R² + complexity, prune below the R² threshold) and
`symbolic_formula` (e.g. `SymPy.jl`/`Symbolics.jl` for the final expression).

## Running tests

```julia
using Pkg
Pkg.activate("KANLux")
Pkg.test()
```

The test suite includes forward-value checks against PyKAN-generated reference
data, layer/container state-transfer checks, gradient-path checks, edge cases,
and global Zygote-vs-finite-difference gradient cross-validation.
