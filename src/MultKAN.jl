# MultKAN.jl — composite KAN container, ported from PyKAN MultKAN.py
#
# A MultKAN stacks `depth` numerical `KANLayer`s and `depth` symbolic
# `SymbolicKANLayer`s. Each layer is followed by a subnode affine transform,
# optional multiplication nodes, and a node affine transform.

using Lux

"""
    MultKAN(width; grid=3, k=3, mult_arity=2, base_fun=silu,
            symbolic_enabled=true, ...)

Composite KAN model.

`width` can be a plain list of neuron counts (e.g. `[2, 5, 1]`) or a list of
`[n_sum, n_mult]` pairs (e.g. `[[2, 0], [5, 0], [1, 0]]`) when multiplication
nodes are present.
"""
struct MultKAN <: Lux.AbstractLuxContainerLayer{(:act_fun, :symbolic_fun)}
    act_fun::Tuple{Vararg{KANLayer}}
    symbolic_fun::Tuple{Vararg{SymbolicKANLayer}}
    width::Vector{Vector{Int}}
    mult_arity::Any
    depth::Int
    width_in::Vector{Int}
    width_out::Vector{Int}
    mult_homo::Bool
    symbolic_enabled::Bool
    grid::Any
    k::Any
end

function _normalize_width(width)
    out = Vector{Vector{Int}}()
    for w in width
        if w isa Integer
            push!(out, [Int(w), 0])
        elseif w isa AbstractVector{<:Integer} && length(w) == 2
            push!(out, [Int(w[1]), Int(w[2])])
        else
            throw(ArgumentError("width entries must be integers or [n_sum, n_mult] pairs"))
        end
    end
    length(out) >= 2 || throw(ArgumentError("width must have at least two layers"))
    return out
end

function MultKAN(width; grid=3, k=3, mult_arity=2, base_fun=silu,
                 symbolic_enabled::Bool=true, grid_eps::Real=0.02,
                 grid_range=(-1.0, 1.0), sp_trainable::Bool=true,
                 sb_trainable::Bool=true)
    width_norm = _normalize_width(width)
    depth = length(width_norm) - 1

    mult_homo = mult_arity isa Integer
    width_in = [width_norm[d][1] + width_norm[d][2] for d in 1:(depth + 1)]

    if mult_homo
        arity = Int(mult_arity)
        width_out = [width_norm[d][1] + arity * width_norm[d][2] for d in 1:(depth + 1)]
    else
        @assert length(mult_arity) == depth + 1 "mult_arity must match width length"
        width_out = [width_norm[d][1] + sum(Int.(mult_arity[d])) for d in 1:(depth + 1)]
    end

    act_layers = Vector{KANLayer}(undef, depth)
    sym_layers = Vector{SymbolicKANLayer}(undef, depth)
    for d in 1:depth
        grid_d = grid isa AbstractVector ? grid[d] : grid
        k_d = k isa AbstractVector ? k[d] : k
        act_layers[d] = KANLayer(width_in[d], width_out[d + 1], grid_d, k_d;
                                 grid_eps=grid_eps, grid_range=grid_range,
                                 base_fun=base_fun, sp_trainable=sp_trainable,
                                 sb_trainable=sb_trainable)
        sym_layers[d] = SymbolicKANLayer(width_in[d], width_out[d + 1])
    end

    return MultKAN(Tuple(act_layers), Tuple(sym_layers), width_norm, mult_arity,
                   depth, width_in, width_out, mult_homo, symbolic_enabled, grid, k)
end

function Lux.initialparameters(rng::AbstractRNG, m::MultKAN)
    act_ps = map(l -> Lux.initialparameters(rng, l), m.act_fun)
    sym_ps = map(l -> Lux.initialparameters(rng, l), m.symbolic_fun)
    node_bias = ntuple(d -> zeros(Float64, m.width_in[d + 1]), m.depth)
    node_scale = ntuple(d -> ones(Float64, m.width_in[d + 1]), m.depth)
    subnode_bias = ntuple(d -> zeros(Float64, m.width_out[d + 1]), m.depth)
    subnode_scale = ntuple(d -> ones(Float64, m.width_out[d + 1]), m.depth)
    return (act_fun=act_ps, symbolic_fun=sym_ps,
            node_bias=node_bias, node_scale=node_scale,
            subnode_bias=subnode_bias, subnode_scale=subnode_scale)
end

function Lux.initialstates(rng::AbstractRNG, m::MultKAN)
    act_st = map(l -> Lux.initialstates(rng, l), m.act_fun)
    sym_st = map(l -> Lux.initialstates(rng, l), m.symbolic_fun)
    return (act_fun=act_st, symbolic_fun=sym_st)
end

"""
    _multiply_subnodes(x, n_sum, n_mult, mult_arity, mult_homo)

Combine subnode columns into multiplication-node columns. `x` has
`n_sum + sum(arities)` columns; the first `n_sum` columns are addition nodes and
the remaining columns are grouped (contiguously for non-homogeneous arities, or
in fixed-size blocks for homogeneous arity) into `n_mult` multiplication nodes.
"""
function _multiply_subnodes(x::AbstractMatrix, n_sum::Int, n_mult::Int,
                            mult_arity, mult_homo::Bool)
    batch = size(x, 1)
    n_mult == 0 && return zeros(Float64, batch, 0)

    if mult_homo
        arity = Int(mult_arity)
        x_mult = x[:, (n_sum + 1):arity:size(x, 2)]
        for k in 2:arity
            x_mult = x_mult .* x[:, (n_sum + k):arity:size(x, 2)]
        end
        return x_mult
    end

    out = zeros(Float64, batch, n_mult)
    acml = n_sum
    for j in 1:n_mult
        ar = Int(mult_arity[j])
        col = x[:, acml + 1]
        for k in 2:ar
            col = col .* x[:, acml + k]
        end
        out[:, j] = col
        acml += ar
    end
    return out
end

"""
    (m::MultKAN)(x, ps, st; singularity_avoiding=false, y_th=10.0)

Forward pass. The numerical and symbolic branches are summed, then each layer's
subnodes receive an affine transform, multiplication nodes (if any) are formed,
and finally node affine transforms are applied.
"""
function (m::MultKAN)(x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                      singularity_avoiding::Bool=false, y_th::Real=10.0)
    x_cur = x
    for d in 1:m.depth
        x_num = m.act_fun[d](x_cur, ps.act_fun[d], st.act_fun[d])[1]
        if m.symbolic_enabled
            x_sym = m.symbolic_fun[d](x_cur, ps.symbolic_fun[d], st.symbolic_fun[d];
                                     singularity_avoiding=singularity_avoiding,
                                     y_th=y_th)[1]
        else
            x_sym = zeros(Float64, size(x_num))
        end

        x_pre = x_num .+ x_sym
        sub_scale = ps.subnode_scale[d]
        sub_bias = ps.subnode_bias[d]
        x_pre = x_pre .* reshape(sub_scale, 1, :) .+ reshape(sub_bias, 1, :)

        n_sum = m.width[d + 1][1]
        n_mult = m.width[d + 1][2]
        if n_mult > 0
            arity = m.mult_homo ? m.mult_arity : m.mult_arity[d + 1]
            x_mult = _multiply_subnodes(x_pre, n_sum, n_mult, arity, m.mult_homo)
            x_cur = hcat(x_pre[:, 1:n_sum], x_mult)
        else
            x_cur = x_pre[:, 1:n_sum]
        end

        node_scale = ps.node_scale[d]
        node_bias = ps.node_bias[d]
        x_cur = x_cur .* reshape(node_scale, 1, :) .+ reshape(node_bias, 1, :)
    end
    return x_cur, st
end
