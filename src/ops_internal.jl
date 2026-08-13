# ops_internal.jl — internal helpers for structural prune/refine

using Statistics

function _width_vectors(width::Vector{Vector{Int}}, mult_arity, mult_homo::Bool)
    n = length(width)
    width_in = [width[d][1] + width[d][2] for d in 1:n]
    if mult_homo
        ar = Int(mult_arity)
        width_out = [width[d][1] + ar * width[d][2] for d in 1:n]
    else
        width_out = [width[d][1] + sum(Int.(mult_arity[d])) for d in 1:n]
    end
    return width_in, width_out
end

function _layer_inputs(m::MultKAN, ps::NamedTuple, st::NamedTuple, x::AbstractMatrix)
    inputs = Vector{Matrix{Float64}}(undef, m.depth)
    x_cur = x
    for d in 1:m.depth
        inputs[d] = x_cur
        x_num = m.act_fun[d](x_cur, ps.act_fun[d], st.act_fun[d])[1]
        if m.symbolic_enabled
            x_sym = m.symbolic_fun[d](x_cur, ps.symbolic_fun[d], st.symbolic_fun[d])[1]
        else
            x_sym = zeros(Float64, size(x_num))
        end
        x_pre = x_num .+ x_sym
        x_pre = x_pre .* reshape(ps.subnode_scale[d], 1, :) .+
                reshape(ps.subnode_bias[d], 1, :)
        n_sum = m.width[d + 1][1]
        n_mult = m.width[d + 1][2]
        if n_mult > 0
            arity = m.mult_homo ? m.mult_arity : m.mult_arity[d + 1]
            x_mult = _multiply_subnodes(x_pre, n_sum, n_mult, arity, m.mult_homo)
            x_cur = hcat(x_pre[:, 1:n_sum], x_mult)
        else
            x_cur = x_pre[:, 1:n_sum]
        end
        x_cur = x_cur .* reshape(ps.node_scale[d], 1, :) .+
                reshape(ps.node_bias[d], 1, :)
    end
    return inputs
end

function _edge_scores(m::MultKAN, ps::NamedTuple, st::NamedTuple,
                      inputs::Vector{Matrix{Float64}})
    scores = Vector{Matrix{Float64}}(undef, m.depth)
    for d in 1:m.depth
        layer = m.act_fun[d]
        lps = ps.act_fun[d]
        lst = st.act_fun[d]
        x_cur = inputs[d]

        base = layer.base_fun(x_cur)
        spline = coef2curve(x_cur, lst.grid, lps.coef, layer.k)
        sb = hasproperty(lps, :scale_base) ? lps.scale_base : lst.scale_base
        sp = hasproperty(lps, :scale_sp) ? lps.scale_sp : lst.scale_sp
        postacts = reshape(sb, 1, layer.in_dim, layer.out_dim) .*
                   reshape(base, size(base, 1), layer.in_dim, 1) .+
                   reshape(sp, 1, layer.in_dim, layer.out_dim) .* spline
        postacts = reshape(lst.mask, 1, layer.in_dim, layer.out_dim) .* postacts

        if size(postacts, 1) > 1
            scores[d] = reshape(std(postacts; dims=1), layer.in_dim, layer.out_dim)
        else
            scores[d] = reshape(mean(abs, postacts; dims=1), layer.in_dim, layer.out_dim)
        end
    end
    return scores
end

function _node_scores(m::MultKAN, scores::Vector{Matrix{Float64}})
    out = Vector{Vector{Float64}}(undef, m.depth + 1)
    out[1] = fill(Inf, m.width_in[1])
    for d in 2:m.depth
        e = scores[d - 1]
        n_sum = m.width[d][1]
        n_mult = m.width[d][2]
        ns = Vector{Float64}(undef, n_sum + n_mult)
        for j in 1:n_sum
            ns[j] = maximum(e[:, j])
        end
        for k in 1:n_mult
            ns[n_sum + k] = Inf
        end
        out[d] = ns
    end
    out[m.depth + 1] = fill(Inf, m.width_in[m.depth + 1])
    return out
end

function _subnode_indices(m::MultKAN, d::Int, active::AbstractVector{Bool})
    n_sum = m.width[d][1]
    n_mult = m.width[d][2]
    idx = Int[]
    for j in 1:n_sum
        active[j] && push!(idx, j)
    end
    if n_mult > 0
        if m.mult_homo
            ar = Int(m.mult_arity)
            for k in 1:n_mult
                if active[n_sum + k]
                    start = n_sum + (k - 1) * ar + 1
                    append!(idx, start:(start + ar - 1))
                end
            end
        else
            arities = m.mult_arity[d]
            acml = n_sum
            for k in 1:n_mult
                ar = Int(arities[k])
                if active[n_sum + k]
                    append!(idx, (acml + 1):(acml + ar))
                end
                acml += ar
            end
        end
    end
    return idx
end

function _subset_kanlayer(layer::KANLayer, lps::NamedTuple, lst::NamedTuple,
                          in_id::AbstractVector{Int}, out_id::AbstractVector{Int})
    new_layer = KANLayer(length(in_id), length(out_id), layer.num, layer.k;
                         grid_eps=layer.grid_eps, grid_range=layer.grid_range,
                         base_fun=layer.base_fun, sp_trainable=layer.sp_trainable,
                         sb_trainable=layer.sb_trainable)

    new_ps = (coef=lps.coef[in_id, out_id, :],)
    hasproperty(lps, :scale_base) &&
        (new_ps = merge(new_ps, (scale_base=lps.scale_base[in_id, out_id],)))
    hasproperty(lps, :scale_sp) &&
        (new_ps = merge(new_ps, (scale_sp=lps.scale_sp[in_id, out_id],)))

    new_st = (grid=lst.grid[in_id, :], mask=lst.mask[in_id, out_id])
    hasproperty(lst, :scale_base) &&
        (new_st = merge(new_st, (scale_base=lst.scale_base[in_id, out_id],)))
    hasproperty(lst, :scale_sp) &&
        (new_st = merge(new_st, (scale_sp=lst.scale_sp[in_id, out_id],)))

    return new_layer, new_ps, new_st
end

function _subset_symbolic(layer::SymbolicKANLayer, lps::NamedTuple, lst::NamedTuple,
                          in_id::AbstractVector{Int}, out_id::AbstractVector{Int})
    new_affine = lps.affine[out_id, in_id, :]
    new_mask = lst.mask[out_id, in_id]
    new_funs = lst.funs[out_id, in_id]
    new_name = lst.funs_name[out_id, in_id]
    new_layer = SymbolicKANLayer(length(in_id), length(out_id), new_affine,
                                 new_mask, new_funs, new_name)
    return new_layer, (affine=new_affine,),
           (mask=new_mask, funs=new_funs, funs_name=new_name)
end
