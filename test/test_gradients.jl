# test_gradients.jl — Phase 8: global AD vs finite-difference cross-validation
#
# The tests below use a 5-point central finite-difference stencil and compare it
# against Zygote for each trainable parameter group, at the level of individual
# layers and the full MultKAN container.

include("test_infra.jl")
using Enzyme

# ── Small helper utilities ──────────────────────────────────────────────────

function _replace_tuple(t::Tuple, idx::Int, v)
    return ntuple(length(t)) do j
        j == idx ? v : t[j]
    end
end

function _fd_gradient(setter::Function, p0::AbstractArray, loss::Function; ϵ::Float64=1e-5)
    flat = vec(copy(p0))
    g_flat = similar(flat)
    for i in eachindex(flat)
        orig = flat[i]
        vals = zeros(Float64, 4)
        for (k, h) in enumerate((2ϵ, ϵ, -ϵ, -2ϵ))
            flat_i = copy(flat)
            flat_i[i] = orig + h
            p_new = reshape(flat_i, size(p0))
            vals[k] = loss(setter(p_new))
        end
        g_flat[i] = (-vals[1] + 8vals[2] - 8vals[3] + vals[4]) / (12ϵ)
    end
    return reshape(g_flat, size(p0))
end

function _grad_report(name::String, g_ad::AbstractArray, g_fd::AbstractArray)
    max_abs = maximum(abs.(g_ad .- g_fd))
    ad_max = maximum(abs.(g_ad))
    fd_max = maximum(abs.(g_fd))
    max_rel = max_abs / (ad_max + eps(Float64))
    println("  [grad] ", name, "  max_abs=", max_abs, "  max_rel=", max_rel)
    return (name=name, max_abs=max_abs, max_rel=max_rel, ad_max=ad_max, fd_max=fd_max)
end

_maxabs(x::AbstractArray) = maximum(abs.(x))
_maxabs(x::Tuple) = maximum(_maxabs(v) for v in x)
_maxabs(x::NamedTuple) = maximum(_maxabs(v) for v in values(x))

_has_nan(x::AbstractArray) = any(isnan, x)
_has_nan(x::Tuple) = any(_has_nan, v for v in x)
_has_nan(x::NamedTuple) = any(_has_nan, v for v in values(x))

reports = Any[]

# ── Layer fixtures ──────────────────────────────────────────────────────────

klayer = KANLayer(2, 2, 3, 3)
kps, kst = Lux.setup(rng, klayer)
kx = randn(rng, 8, 2)
kloss = p -> sum(klayer(kx, p, kst)[1])

slayer = SymbolicKANLayer(2, 2)
fix_symbolic!(slayer, 1, 1, "sin")
slayer.mask[1, 1] = 1.0
sps = (affine=slayer.affine,)
sst = (mask=slayer.mask, funs=slayer.funs, funs_name=slayer.funs_name)
sx = randn(rng, 8, 2)
sloss = p -> sum(slayer(sx, p, sst)[1])

model = MultKAN([2, 2, 1]; grid=3, k=3, symbolic_enabled=true)
fix_symbolic!(model.symbolic_fun[1], 1, 1, "sin")
model.symbolic_fun[1].mask[1, 1] = 1.0
fix_symbolic!(model.symbolic_fun[2], 1, 1, "sin")
model.symbolic_fun[2].mask[1, 1] = 1.0
mps, mst = Lux.setup(rng, model)
mx = randn(rng, 8, 2)
mloss = p -> sum(model(mx, p, mst)[1])

# ── Per-group helpers ───────────────────────────────────────────────────────

function _klayer_report(key::Symbol)
    g_ad = getproperty(Zygote.gradient(kloss, kps)[1], key)
    setter = p -> merge(kps, NamedTuple{(key,)}((p,)))
    g_fd = _fd_gradient(setter, getproperty(kps, key), kloss)
    return _grad_report("KANLayer.$key", g_ad, g_fd)
end

function _symbolic_report()
    g_ad = Zygote.gradient(sloss, sps)[1].affine
    setter = p -> merge(sps, (affine=p,))
    g_fd = _fd_gradient(setter, sps.affine, sloss)
    return _grad_report("SymbolicKANLayer.affine", g_ad, g_fd)
end

function _multkan_act_report(key::Symbol, idx::Int)
    g_ad = getproperty(Zygote.gradient(mloss, mps)[1].act_fun[idx], key)
    p0 = getproperty(mps.act_fun[idx], key)
    setter = p -> begin
        new_sub = merge(mps.act_fun[idx], NamedTuple{(key,)}((p,)))
        new_act = _replace_tuple(mps.act_fun, idx, new_sub)
        merge(mps, (act_fun=new_act,))
    end
    g_fd = _fd_gradient(setter, p0, mloss)
    return _grad_report("MultKAN.act_fun[$idx].$key", g_ad, g_fd)
end

function _multkan_symbolic_report(idx::Int)
    g_ad = Zygote.gradient(mloss, mps)[1].symbolic_fun[idx].affine
    p0 = mps.symbolic_fun[idx].affine
    setter = p -> begin
        new_sub = merge(mps.symbolic_fun[idx], (affine=p,))
        new_sym = _replace_tuple(mps.symbolic_fun, idx, new_sub)
        merge(mps, (symbolic_fun=new_sym,))
    end
    g_fd = _fd_gradient(setter, p0, mloss)
    return _grad_report("MultKAN.symbolic_fun[$idx].affine", g_ad, g_fd)
end

function _multkan_tuple_report(field::Symbol, idx::Int)
    g_ad = getproperty(Zygote.gradient(mloss, mps)[1], field)[idx]
    p0 = getproperty(getproperty(mps, field), idx)
    setter = p -> begin
        new_tuple = _replace_tuple(getproperty(mps, field), idx, p)
        merge(mps, NamedTuple{(field,)}((new_tuple,)))
    end
    g_fd = _fd_gradient(setter, p0, mloss)
    return _grad_report("MultKAN.$field[$idx]", g_ad, g_fd)
end

# ── G1-G6 + diagnostics ─────────────────────────────────────────────────────

@testset "Global gradient cross-validation (Phase 8)" begin
    @testset "G1: KANLayer coef" begin
        r = _klayer_report(:coef)
        push!(reports, r)
        @test r.max_abs < 1e-4
    end

    @testset "G2: KANLayer scale_sp" begin
        r = _klayer_report(:scale_sp)
        push!(reports, r)
        @test r.max_abs < 1e-4
    end

    @testset "G3: KANLayer scale_base" begin
        r = _klayer_report(:scale_base)
        push!(reports, r)
        @test r.max_abs < 1e-4
    end

    @testset "G4: SymbolicKANLayer affine" begin
        r = _symbolic_report()
        push!(reports, r)
        @test r.max_abs < 1e-4
    end

    @testset "G5: MultKAN full propagation" begin
        rs = [
            _multkan_act_report(:coef, 1),
            _multkan_act_report(:scale_sp, 1),
            _multkan_act_report(:scale_base, 1),
            _multkan_act_report(:coef, 2),
            _multkan_act_report(:scale_sp, 2),
            _multkan_act_report(:scale_base, 2),
            _multkan_symbolic_report(1),
            _multkan_symbolic_report(2),
        ]
        foreach(r -> push!(reports, r), rs)
        @test maximum(r.max_abs for r in rs) < 1e-3
    end

    @testset "G6: MultKAN node/subnode affine" begin
        rs = Any[]
        for field in (:node_bias, :node_scale, :subnode_bias, :subnode_scale)
            for idx in 1:model.depth
                push!(rs, _multkan_tuple_report(field, idx))
            end
        end
        foreach(r -> push!(reports, r), rs)
        @test maximum(r.max_abs for r in rs) < 1e-4
    end

    @testset "G7: Enzyme reverse-mode smoke" begin
        mode = Enzyme.set_runtime_activity(Reverse)
        gk = Enzyme.gradient(mode, kloss, kps)[1]
        gs = Enzyme.gradient(mode, sloss, sps)[1]
        gm = Enzyme.gradient(mode, mloss, mps)[1]

        @test !_has_nan(gk) && _maxabs(gk) > 0
        @test !_has_nan(gs) && _maxabs(gs) > 0
        @test !_has_nan(gm) && _maxabs(gm) > 0
    end

    @testset "D1: NaN detection" begin
        groups = [
            getproperty(Zygote.gradient(kloss, kps)[1], :coef),
            getproperty(Zygote.gradient(kloss, kps)[1], :scale_sp),
            getproperty(Zygote.gradient(kloss, kps)[1], :scale_base),
            Zygote.gradient(sloss, sps)[1].affine,
            Zygote.gradient(mloss, mps)[1],
        ]
        @test !any(_has_nan, groups)
    end

    @testset "D2: vanishing gradient detection" begin
        groups = [
            Zygote.gradient(kloss, kps)[1].coef,
            Zygote.gradient(kloss, kps)[1].scale_sp,
            Zygote.gradient(kloss, kps)[1].scale_base,
            Zygote.gradient(sloss, sps)[1].affine,
            Zygote.gradient(mloss, mps)[1].act_fun,
            Zygote.gradient(mloss, mps)[1].symbolic_fun,
            Zygote.gradient(mloss, mps)[1].node_bias,
            Zygote.gradient(mloss, mps)[1].node_scale,
            Zygote.gradient(mloss, mps)[1].subnode_bias,
            Zygote.gradient(mloss, mps)[1].subnode_scale,
        ]
        absmaxes = _maxabs.(groups)
        @test all(v -> v > 1e-8, absmaxes)
    end

    @testset "D3: per-group report" begin
        println("  [grad] report entries: ", length(reports))
        @test length(reports) >= 6
    end
end
