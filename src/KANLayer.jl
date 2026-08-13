# KANLayer.jl — B-spline edge activation layer, ported from PyKAN KANLayer.py
#
# Lux layer implementation. Trainable parameters are stored in `ps`, while
# `grid` and `mask` are non-trainable state. When a scale parameter is marked
# non-trainable, it is placed in `st` instead of `ps`, matching Lux's convention.

using Lux
using Random

_sigmoid(x) = 1 ./ (1 .+ exp.(-x))
silu(x) = x .* _sigmoid(x)

"""
    KANLayer(in_dim, out_dim, num, k; ...)

B-spline KAN activation layer. Each input/output edge carries a masked
combination of a base activation and a learnable B-spline.
"""
struct KANLayer{F} <: Lux.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    num::Int
    k::Int
    grid_eps::Float64
    grid_range::Tuple{Float64,Float64}
    base_fun::F
    sp_trainable::Bool
    sb_trainable::Bool
end

function KANLayer(in_dim::Integer, out_dim::Integer, num::Integer, k::Integer;
                  grid_eps::Real=0.02, grid_range=(-1.0, 1.0), base_fun=silu,
                  sp_trainable::Bool=true, sb_trainable::Bool=true)
    return KANLayer{typeof(base_fun)}(
        Int(in_dim), Int(out_dim), Int(num), Int(k), Float64(grid_eps),
        (Float64(grid_range[1]), Float64(grid_range[2])), base_fun,
        sp_trainable, sb_trainable,
    )
end

_initial_grid(l::KANLayer) = begin
    base = reshape(range(l.grid_range[1], l.grid_range[2]; length=l.num + 1), 1, :)
    grid = repeat(base, l.in_dim, 1)
    extend_grid(grid, l.k)
end

_initial_scale_base(rng::AbstractRNG, l::KANLayer) =
    ((rand(rng, l.in_dim, l.out_dim) .* 2.0) .- 1.0) ./ sqrt(l.in_dim)

_initial_scale_sp(l::KANLayer) = ones(l.in_dim, l.out_dim) ./ sqrt(l.in_dim)

_initial_coef(rng::AbstractRNG, l::KANLayer) = begin
    grid = _initial_grid(l)
    S = size(grid, 2)
    x_eval = permutedims(grid[:, (l.k + 1):(S - l.k)])          # (num+1, in_dim)
    noises = (rand(rng, l.num + 1, l.in_dim, l.out_dim) .- 0.5) .* (0.5 / l.num)
    curve2coef(x_eval, noises, grid, l.k)
end

function Lux.initialparameters(rng::AbstractRNG, l::KANLayer)
    coef = _initial_coef(rng, l)
    sb = _initial_scale_base(rng, l)
    sp = _initial_scale_sp(l)

    if l.sb_trainable && l.sp_trainable
        return (coef=coef, scale_base=sb, scale_sp=sp)
    elseif l.sb_trainable
        return (coef=coef, scale_base=sb)
    elseif l.sp_trainable
        return (coef=coef, scale_sp=sp)
    end
    return (coef=coef,)
end

function Lux.initialstates(rng::AbstractRNG, l::KANLayer)
    st = (grid=_initial_grid(l), mask=ones(l.in_dim, l.out_dim))
    if !l.sb_trainable
        st = merge(st, (scale_base=_initial_scale_base(rng, l),))
    end
    if !l.sp_trainable
        st = merge(st, (scale_sp=_initial_scale_sp(l),))
    end
    return st
end

@inline function _scale_from(key::Symbol, ps::NamedTuple, st::NamedTuple)
    return hasproperty(ps, key) ? getproperty(ps, key) : getproperty(st, key)
end

function (l::KANLayer)(x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    batch = size(x, 1)
    base = l.base_fun(x)                                    # (batch, in_dim)
    spline = coef2curve(x, st.grid, ps.coef, l.k)            # (batch, in_dim, out_dim)

    scale_base = _scale_from(:scale_base, ps, st)
    scale_sp = _scale_from(:scale_sp, ps, st)

    y = reshape(scale_base, 1, l.in_dim, l.out_dim) .* reshape(base, batch, l.in_dim, 1) .+
        reshape(scale_sp, 1, l.in_dim, l.out_dim) .* spline
    y = reshape(st.mask, 1, l.in_dim, l.out_dim) .* y
    y = reshape(sum(y; dims=2), batch, l.out_dim)
    return y, st
end
