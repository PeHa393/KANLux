# SymbolicKANLayer.jl — symbolic function activation layer, ported from PyKAN
#
# Lux layer. `affine` is the only trainable parameter group and holds, for
# every (output, input) edge, the four numbers `[a, b, c, d]` of the edge
# transform `c * f(a * x + b) + d`. The symbolic functions themselves are
# configuration data (state) rather than parameters, matching the PyKAN
# `Symbolic_KANLayer` design.

using Lux
using Random

_symbolic_zero(x) = zero(x)

"""
    SymbolicKANLayer(in_dim, out_dim)

Symbolic activation layer. Each output/input edge can be assigned a closed-form
function from `SYMBOLIC_LIB` through [`fix_symbolic!`](@ref). Unfixed edges emit
zero.

The layer stores its parameter/state arrays directly. `initialparameters` returns
`(affine,)` and `initialstates` returns `(mask, funs, funs_name)`. `affine` has
shape `(out_dim, in_dim, 4)` with entries `[a, b, c, d]`.
"""
struct SymbolicKANLayer <: Lux.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    affine::Array{Float64,3}
    mask::Array{Float64,2}
    funs::Array{Function,2}
    funs_name::Array{String,2}
end

function SymbolicKANLayer(in_dim::Integer, out_dim::Integer)
    in_dim = Int(in_dim)
    out_dim = Int(out_dim)
    affine = zeros(Float64, out_dim, in_dim, 4)
    mask = zeros(Float64, out_dim, in_dim)
    funs = Array{Function,2}(undef, out_dim, in_dim)
    funs_name = fill("0", out_dim, in_dim)
    for j in 1:out_dim, i in 1:in_dim
        funs[j, i] = _symbolic_zero
    end
    return SymbolicKANLayer(in_dim, out_dim, affine, mask, funs, funs_name)
end

Lux.initialparameters(::AbstractRNG, l::SymbolicKANLayer) = (affine=l.affine,)

Lux.initialstates(::AbstractRNG, l::SymbolicKANLayer) =
    (mask=l.mask, funs=l.funs, funs_name=l.funs_name)

"""
    fit_symbolic_params(x, y, fun; a_range=(-10, 10), b_range=(-10, 10), grid_number=21)

Fit the affine coefficients `[a, b, c, d]` for `y ≈ c * fun(a * x + b) + d`.

`a` and `b` are selected by a coarse grid search (the same approach used by
PyKAN's `fit_params`, without iterative zooming), and `c` and `d` are then
closed-form linear least squares.
"""
function fit_symbolic_params(x::AbstractVector, y::AbstractVector, fun::Function;
                             a_range=(-10.0, 10.0), b_range=(-10.0, 10.0),
                             grid_number::Int=21)
    @assert length(x) == length(y) "x and y must have the same length"
    n = length(x)
    n == 0 && return (1.0, 0.0, 1.0, 0.0)

    as = range(a_range[1], a_range[2]; length=grid_number)
    bs = range(b_range[1], b_range[2]; length=grid_number)
    ymean = sum(y) / n
    ss_tot = sum(abs2, y .- ymean)

    best = (a=1.0, b=0.0, c=1.0, d=0.0, r2=-Inf)

    for a in as, b in bs
        z = [fun(a * x[k] + b) for k in 1:n]
        all(isfinite, z) || continue
        zmean = sum(z) / n
        denom = sum(abs2, z .- zmean)
        denom == 0 && continue
        c = sum((z .- zmean) .* (y .- ymean)) / denom
        d = ymean - c * zmean
        resid = y .- (c .* z .+ d)
        r2 = 1 - sum(abs2, resid) / (ss_tot + eps(Float64))
        if r2 > best.r2
            best = (a=a, b=b, c=c, d=d, r2=r2)
        end
    end

    return (best.a, best.b, best.c, best.d)
end

function _fix_symbolic!(l::SymbolicKANLayer, i::Int, j::Int, name::String,
                        x, y; random::Bool, a_range, b_range, grid_number::Int)
    @assert 1 <= i <= l.in_dim "input index i out of range"
    @assert 1 <= j <= l.out_dim "output index j out of range"
    @assert haskey(SYMBOLIC_LIB, name) "unknown symbolic function name: $name"

    fun, _, _ = SYMBOLIC_LIB[name]
    l.funs[j, i] = fun
    l.funs_name[j, i] = name

    if x === nothing || y === nothing
        if random
            l.affine[j, i, 1:4] = rand(Random.default_rng(), 4) .* 2.0 .- 1.0
        else
            l.affine[j, i, 1] = 1.0
            l.affine[j, i, 2] = 0.0
            l.affine[j, i, 3] = 1.0
            l.affine[j, i, 4] = 0.0
        end
    else
        a, b, c, d = fit_symbolic_params(x, y, fun;
                                         a_range=a_range, b_range=b_range,
                                         grid_number=grid_number)
        l.affine[j, i, 1] = a
        l.affine[j, i, 2] = b
        l.affine[j, i, 3] = c
        l.affine[j, i, 4] = d
    end
    return nothing
end

"""
    fix_symbolic!(layer, i, j, name; x=nothing, y=nothing, random=false, ...)
    fix_symbolic!(layer, i, j, name, x, y; random=false, ...)

Assign the symbolic function `name` (a key of `SYMBOLIC_LIB`) to edge
`(input i, output j)` and set its affine coefficients. With `x` and `y` absent
the transform is the identity `[1, 0, 1, 0]`; with `x` and `y` provided they are
fitted to `y ≈ c * f(a * x + b) + d`.
"""
function fix_symbolic!(l::SymbolicKANLayer, i::Integer, j::Integer,
                       name::AbstractString, x, y;
                       random::Bool=false, a_range=(-10.0, 10.0),
                       b_range=(-10.0, 10.0), grid_number::Int=21)
    return _fix_symbolic!(l, Int(i), Int(j), String(name), x, y;
                          random=random, a_range=a_range, b_range=b_range,
                          grid_number=grid_number)
end

function fix_symbolic!(l::SymbolicKANLayer, i::Integer, j::Integer,
                       name::AbstractString; random::Bool=false,
                       a_range=(-10.0, 10.0), b_range=(-10.0, 10.0),
                       grid_number::Int=21)
    return _fix_symbolic!(l, Int(i), Int(j), String(name), nothing, nothing;
                          random=random, a_range=a_range, b_range=b_range,
                          grid_number=grid_number)
end

"""
    (layer::SymbolicKANLayer)(x, ps, st; singularity_avoiding=false, y_th=10.0)

Forward pass. `x` has shape `(batch, in_dim)` and the result has shape
`(batch, out_dim)`. The loop is explicit over input/output dimensions (scalar
per-edge work), mirroring the non-vectorized PyKAN reference.
"""
function (l::SymbolicKANLayer)(x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                               singularity_avoiding::Bool=false, y_th::Real=10.0)
    batch = size(x, 1)
    postacts = [
        begin
            a_ = ps.affine[j, i, 1]
            b_ = ps.affine[j, i, 2]
            c_ = ps.affine[j, i, 3]
            d_ = ps.affine[j, i, 4]
            pre = a_ * x[b, i] + b_
            z = if singularity_avoiding
                SYMBOLIC_LIB[st.funs_name[j, i]][3](pre, y_th)
            else
                st.funs[j, i](pre)
            end
            st.mask[j, i] * (c_ * z + d_)
        end
        for b in 1:batch, j in 1:l.out_dim, i in 1:l.in_dim
    ]
    y = reshape(sum(postacts; dims=3), batch, l.out_dim)
    return y, st
end
