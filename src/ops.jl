# ops.jl — structural prune and grid refine operations

using Lux
using Random

function _prune_layer(m::MultKAN, ps::NamedTuple, st::NamedTuple,
                      scores::Vector{Matrix{Float64}},
                      active::Vector{<:AbstractVector{Bool}}, d::Int, edge_th::Real)
    in_id = findall(active[d])
    out_id = _subnode_indices(m, d + 1, active[d + 1])

    nl, nps, nst = _subset_kanlayer(m.act_fun[d], ps.act_fun[d], st.act_fun[d],
                                    in_id, out_id)
    sl, sps, sst = _subset_symbolic(m.symbolic_fun[d], ps.symbolic_fun[d],
                                    st.symbolic_fun[d], in_id, out_id)

    if edge_th > 0
        keep = scores[d][in_id, out_id] .> Float64(edge_th)
        nst = merge(nst, (mask=nst.mask .* keep,))
    end

    return (
        act_layer=nl, act_ps=nps, act_st=nst,
        sym_layer=sl, sym_ps=sps, sym_st=sst,
        node_bias=ps.node_bias[d][active[d + 1]],
        node_scale=ps.node_scale[d][active[d + 1]],
        subnode_bias=ps.subnode_bias[d][out_id],
        subnode_scale=ps.subnode_scale[d][out_id],
    )
end

function _refine_layer(m::MultKAN, ps::NamedTuple, st::NamedTuple,
                       inputs::Vector{Matrix{Float64}}, grids::Vector{Int}, d::Int)
    layer = m.act_fun[d]
    lps = ps.act_fun[d]
    lst = st.act_fun[d]
    num_new = grids[d]

    nl = KANLayer(layer.in_dim, layer.out_dim, num_new, layer.k;
                  grid_eps=layer.grid_eps, grid_range=layer.grid_range,
                  base_fun=layer.base_fun, sp_trainable=layer.sp_trainable,
                  sb_trainable=layer.sb_trainable)
    new_grid_full = _initial_grid(nl)
    x_eval = inputs[d]
    y_eval = coef2curve(x_eval, lst.grid, lps.coef, layer.k)
    new_coef = curve2coef(x_eval, y_eval, new_grid_full, layer.k)

    nps = (coef=new_coef,)
    hasproperty(lps, :scale_base) &&
        (nps = merge(nps, (scale_base=copy(lps.scale_base),)))
    hasproperty(lps, :scale_sp) &&
        (nps = merge(nps, (scale_sp=copy(lps.scale_sp),)))
    nst = (grid=new_grid_full, mask=copy(lst.mask))
    hasproperty(lst, :scale_base) &&
        (nst = merge(nst, (scale_base=copy(lst.scale_base),)))
    hasproperty(lst, :scale_sp) &&
        (nst = merge(nst, (scale_sp=copy(lst.scale_sp),)))

    slayer = m.symbolic_fun[d]
    slps = ps.symbolic_fun[d]
    slst = st.symbolic_fun[d]
    new_slayer = SymbolicKANLayer(slayer.in_dim, slayer.out_dim,
                                  copy(slps.affine), copy(slst.mask),
                                  copy(slst.funs), copy(slst.funs_name))

    return (
        act_layer=nl, act_ps=nps, act_st=nst,
        sym_layer=new_slayer,
        sym_ps=(affine=new_slayer.affine,),
        sym_st=(mask=new_slayer.mask, funs=new_slayer.funs,
                funs_name=new_slayer.funs_name),
    )
end

"""
    prune(model, ps, st, x; node_th=1e-2, edge_th=3e-2)

Remove low-importance hidden addition nodes and optionally zero low-importance
edges. Returns `(model=..., ps=..., st=...)` with the pruned structure and
sliced parameters/state. Input and output nodes are always kept, as are
multiplication nodes.
"""
function prune(m::MultKAN, ps::NamedTuple, st::NamedTuple, x::AbstractMatrix;
               node_th::Real=1e-2, edge_th::Real=3e-2)
    inputs = _layer_inputs(m, ps, st, x)
    scores = _edge_scores(m, ps, st, inputs)
    node_scores = _node_scores(m, scores)

    active = [node_scores[d] .> Float64(node_th) for d in 1:(m.depth + 1)]
    active[1] .= true
    active[m.depth + 1] .= true

    for d in 2:m.depth
        n_sum = m.width[d][1]
        n_mult = m.width[d][2]
        if count(active[d][1:n_sum]) == 0 && n_mult == 0
            throw(ArgumentError(
                "prune would remove all hidden neurons in layer $d; " *
                "lower node_th or provide non-degenerate input"
            ))
        end
    end

    new_width = Vector{Vector{Int}}(undef, m.depth + 1)
    new_width[1] = copy(m.width[1])
    for d in 2:m.depth
        n_sum = m.width[d][1]
        n_mult = m.width[d][2]
        new_width[d] = [count(active[d][1:n_sum]), n_mult]
    end
    new_width[m.depth + 1] = copy(m.width[m.depth + 1])

    layers = ntuple(d -> _prune_layer(m, ps, st, scores, active, d, edge_th),
                    m.depth)
    new_act = ntuple(d -> layers[d].act_layer, m.depth)
    new_sym = ntuple(d -> layers[d].sym_layer, m.depth)
    new_act_ps = ntuple(d -> layers[d].act_ps, m.depth)
    new_act_st = ntuple(d -> layers[d].act_st, m.depth)
    new_sym_ps = ntuple(d -> layers[d].sym_ps, m.depth)
    new_sym_st = ntuple(d -> layers[d].sym_st, m.depth)
    node_bias_new = ntuple(d -> layers[d].node_bias, m.depth)
    node_scale_new = ntuple(d -> layers[d].node_scale, m.depth)
    subnode_bias_new = ntuple(d -> layers[d].subnode_bias, m.depth)
    subnode_scale_new = ntuple(d -> layers[d].subnode_scale, m.depth)

    new_width_in, new_width_out = _width_vectors(new_width, m.mult_arity, m.mult_homo)
    new_model = MultKAN(new_act, new_sym, new_width, m.mult_arity,
                        m.depth, new_width_in, new_width_out, m.mult_homo,
                        m.symbolic_enabled, m.grid, m.k)
    new_ps = (act_fun=new_act_ps, symbolic_fun=new_sym_ps,
              node_bias=node_bias_new, node_scale=node_scale_new,
              subnode_bias=subnode_bias_new, subnode_scale=subnode_scale_new)
    new_st = (act_fun=new_act_st, symbolic_fun=new_sym_st)
    return (model=new_model, ps=new_ps, st=new_st)
end

function prune(m::MultKAN, x::AbstractMatrix; node_th::Real=1e-2,
               edge_th::Real=3e-2, rng::AbstractRNG=Random.default_rng())
    ps, st = Lux.setup(rng, m)
    return prune(m, ps, st, x; node_th=node_th, edge_th=edge_th)
end

"""
    refine(model, ps, st, x, new_grid)

Increase the B-spline grid resolution while preserving the learned function.
For each layer the old spline is evaluated on the layer's intermediate
activations and re-fitted with `curve2coef` on the new grid. Symbolic layers and
affine parameters are copied unchanged. Returns
`(model=..., ps=..., st=...)`.
"""
function refine(m::MultKAN, ps::NamedTuple, st::NamedTuple, x::AbstractMatrix,
                new_grid)
    grids = new_grid isa AbstractVector ? collect(Int, new_grid) : fill(Int(new_grid), m.depth)
    @assert length(grids) == m.depth "new_grid must match model depth"
    inputs = _layer_inputs(m, ps, st, x)

    layers = ntuple(d -> _refine_layer(m, ps, st, inputs, grids, d), m.depth)
    new_act = ntuple(d -> layers[d].act_layer, m.depth)
    new_sym = ntuple(d -> layers[d].sym_layer, m.depth)
    new_act_ps = ntuple(d -> layers[d].act_ps, m.depth)
    new_act_st = ntuple(d -> layers[d].act_st, m.depth)
    new_sym_ps = ntuple(d -> layers[d].sym_ps, m.depth)
    new_sym_st = ntuple(d -> layers[d].sym_st, m.depth)

    new_model = MultKAN(new_act, new_sym, m.width, m.mult_arity,
                        m.depth, m.width_in, m.width_out, m.mult_homo,
                        m.symbolic_enabled, new_grid, m.k)
    new_ps = (act_fun=new_act_ps, symbolic_fun=new_sym_ps,
              node_bias=copy.(ps.node_bias), node_scale=copy.(ps.node_scale),
              subnode_bias=copy.(ps.subnode_bias), subnode_scale=copy.(ps.subnode_scale))
    new_st = (act_fun=new_act_st, symbolic_fun=new_sym_st)
    return (model=new_model, ps=new_ps, st=new_st)
end

function refine(m::MultKAN, x::AbstractMatrix, new_grid;
                rng::AbstractRNG=Random.default_rng())
    ps, st = Lux.setup(rng, m)
    return refine(m, ps, st, x, new_grid)
end
