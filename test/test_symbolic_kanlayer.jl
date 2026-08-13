# test_symbolic_kanlayer.jl — Phase 5: SymbolicKANLayer forward, gradients, state

include("test_infra.jl")

const SKL_IN = 2
const SKL_OUT = 3
const SKL_BATCH = 100

function _build_reference_layer()
    l = SymbolicKANLayer(SKL_IN, SKL_OUT)
    # PyKAN's `fun_map` is indexed (output j, input i). Julia uses 1-based
    # indices, so convert each pair.
    fix_symbolic!(l, 1, 1, "sin")
    fix_symbolic!(l, 2, 1, "x^2")
    fix_symbolic!(l, 1, 2, "exp")
    fix_symbolic!(l, 2, 2, "x")
    fix_symbolic!(l, 1, 3, "abs")
    fix_symbolic!(l, 2, 3, "tanh")
    return l
end

function _symbolic_fd_maxdiff(layer, x, ps, st; ϵ=1e-5)
    key = :affine
    g_ad = getfield(Zygote.gradient(p -> sum(layer(x, p, st)[1]), ps)[1], key)
    p = getfield(ps, key)
    p_flat = vec(copy(p))
    g_flat = similar(p_flat)

    for idx in eachindex(p_flat)
        f_vals = zeros(4)
        for (k, h) in enumerate((2ϵ, ϵ, -ϵ, -2ϵ))
            p_flat[idx] = p_flat[idx] + h
            p_new = reshape(p_flat, size(p))
            ps_new = merge(ps, NamedTuple{(key,)}((p_new,)))
            f_vals[k] = sum(layer(x, ps_new, st)[1])
            p_flat[idx] = p_flat[idx] - h
        end
        g_flat[idx] = (-f_vals[1] + 8f_vals[2] - 8f_vals[3] + f_vals[4]) / (12ϵ)
    end

    return maximum(abs.(g_ad .- reshape(g_flat, size(g_ad))))
end

@testset "SymbolicKANLayer (Phase 5)" begin
    @testset "A) forward values" begin
        @testset "A1: output shape" begin
            l = _build_reference_layer()
            ps = (affine=l.affine,)
            st = (mask=ones(SKL_OUT, SKL_IN), funs=l.funs, funs_name=l.funs_name)
            y = l(randn(rng, SKL_BATCH, SKL_IN), ps, st)[1]
            @test size(y) == (SKL_BATCH, SKL_OUT)
        end

        @testset "A2: forward vs PyKAN reference" begin
            x_ref = load_reference("symbolic_kanlayer_input")
            y_ref = load_reference("symbolic_kanlayer_output")
            affine_ref = load_reference("symbolic_kanlayer_affine")
            mask_ref = load_reference("symbolic_kanlayer_mask")
            l = _build_reference_layer()
            ps = (affine=affine_ref,)
            st = (mask=mask_ref, funs=l.funs, funs_name=l.funs_name)
            y = l(x_ref, ps, st)[1]
            @test maximum(abs.(y .- y_ref)) < 1e-6
        end

        @testset "A3: activation function correctness" begin
            l = SymbolicKANLayer(2, 2)
            fix_symbolic!(l, 1, 1, "sin")
            mask = zeros(2, 2)
            mask[1, 1] = 1.0
            ps = (affine=l.affine,)
            st = (mask=mask, funs=l.funs, funs_name=l.funs_name)
            x = randn(rng, 32, 2)
            y = l(x, ps, st)[1]
            @test isapprox(y[:, 1], sin.(x[:, 1]); atol=1e-12) &&
                  all(iszero, y[:, 2])
        end
    end

    grad_layer = SymbolicKANLayer(2, 2)
    fix_symbolic!(grad_layer, 1, 1, "sin")
    ps_g = (affine=grad_layer.affine,)
    mask_g = zeros(2, 2)
    mask_g[1, 1] = 1.0
    st_g = (mask=mask_g, funs=grad_layer.funs, funs_name=grad_layer.funs_name)
    x_g = rand(rng, 32, 2)

    @testset "B) gradient paths" begin
        @testset "B1: affine a non-zero" begin
            g = Zygote.gradient(p -> sum(grad_layer(x_g, p, st_g)[1]), ps_g)[1].affine
            @test g[1, 1, 1] != 0.0 && !any(isnan, g)
        end

        @testset "B2: affine b non-zero" begin
            g = Zygote.gradient(p -> sum(grad_layer(x_g, p, st_g)[1]), ps_g)[1].affine
            @test g[1, 1, 2] != 0.0 && !any(isnan, g)
        end

        @testset "B3: affine c non-zero" begin
            g = Zygote.gradient(p -> sum(grad_layer(x_g, p, st_g)[1]), ps_g)[1].affine
            @test g[1, 1, 3] != 0.0 && !any(isnan, g)
        end

        @testset "B4: affine d non-zero" begin
            g = Zygote.gradient(p -> sum(grad_layer(x_g, p, st_g)[1]), ps_g)[1].affine
            @test g[1, 1, 4] != 0.0 && !any(isnan, g)
        end

        @testset "B5: mask has no gradient" begin
            grads = Zygote.gradient(p -> sum(grad_layer(x_g, p, st_g)[1]), ps_g)[1]
            @test !(:mask in propertynames(grads))
        end

        @testset "B6: affine vs finite differences" begin
            @test _symbolic_fd_maxdiff(grad_layer, x_g, ps_g, st_g) < 1e-4
        end
    end

    @testset "C) state transfer" begin
        l = _build_reference_layer()
        ps = (affine=l.affine,)
        st = (mask=ones(SKL_OUT, SKL_IN), funs=l.funs, funs_name=l.funs_name)
        x = randn(rng, SKL_BATCH, SKL_IN)
        _, st2 = l(x, ps, st)

        @testset "C1: same state keys" begin
            @test sort(collect(propertynames(st2))) == sort(collect(propertynames(st)))
        end

        @testset "C2: mask unchanged" begin
            @test st2.mask === st.mask
        end

        @testset "C3: funs reference unchanged" begin
            @test st2.funs === st.funs
        end
    end

    @testset "D) edge cases" begin
        @testset "D1: all-zero mask" begin
            l = _build_reference_layer()
            ps = (affine=l.affine,)
            st = (mask=zeros(SKL_OUT, SKL_IN), funs=l.funs, funs_name=l.funs_name)
            y = l(randn(rng, 16, SKL_IN), ps, st)[1]
            @test all(iszero, y) && !any(isnan, y)
        end

        @testset "D2: singularity-avoiding mode" begin
            l = SymbolicKANLayer(1, 1)
            fix_symbolic!(l, 1, 1, "1/x")
            mask = ones(1, 1)
            ps = (affine=l.affine,)
            st = (mask=mask, funs=l.funs, funs_name=l.funs_name)
            x = reshape([0.0, 0.5, 1.0], 3, 1)
            y = l(x, ps, st; singularity_avoiding=true, y_th=10.0)[1]
            @test all(isfinite, y)
        end

        @testset "D3: unfixed edge emits zero" begin
            l = SymbolicKANLayer(2, 2)
            ps = (affine=l.affine,)
            st = (mask=ones(2, 2), funs=l.funs, funs_name=l.funs_name)
            y = l(randn(rng, 16, 2), ps, st)[1]
            @test all(iszero, y) && !any(isnan, y)
        end

        @testset "D4: mixed fixed/unfixed edges" begin
            l = SymbolicKANLayer(2, 2)
            fix_symbolic!(l, 1, 1, "x^2")
            mask = zeros(2, 2)
            mask[1, 1] = 1.0   # fixed edge contributes
            mask[1, 2] = 1.0   # unfixed edge contributes zero
            ps = (affine=l.affine,)
            st = (mask=mask, funs=l.funs, funs_name=l.funs_name)
            x = randn(rng, 16, 2)
            y = l(x, ps, st)[1]
            @test isapprox(y[:, 1], x[:, 1] .^ 2; atol=1e-12) &&
                  all(iszero, y[:, 2])
        end
    end
end
