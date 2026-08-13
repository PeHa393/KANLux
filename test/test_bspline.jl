# test_bspline.jl — Phase 1: B-spline math primitives (12 spec items)
#
# Covers:
#   A) forward values (6): partition-of-unity, B_batch vs PyKAN, coef2curve vs
#      PyKAN, curve2coef round-trip, extend_grid node count + spacing
#   B) gradient paths (4): coef grad non-zero, grid non-trainable, B_batch grad
#      to x, curve2coef absent from training path
#   D) edge cases (4): k=0, k=5, degenerate grid NaN->0, x beyond grid range

include("test_infra.jl")

@testset "B-spline core (Phase 1)" begin
    IN_DIM = 2
    OUT_DIM = 3
    NUM = 5        # grid intervals
    K = 3          # cubic B-spline
    BATCH = 100

    # ── shared reference data (matching generate_reference.py) ──
    x_ref = load_reference("b_batch_input_x")        # (100, 2)
    grid_ref = load_reference("b_batch_grid")        # (2, 12)
    out_bb_ref = load_reference("b_batch_output")    # (100, 2, 8)

    @testset "A) forward values" begin
        @testset "A1: partition of unity" begin
            b = B_batch(x_ref, grid_ref, K)
            sums = dropdims(sum(b; dims=3); dims=3)  # (100, 2)
            @test all(isapprox.(sums, 1.0; atol=1e-10))
        end

        @testset "A2: B_batch vs PyKAN" begin
            b = B_batch(x_ref, grid_ref, K)
            @test size(b) == (BATCH, IN_DIM, NUM + K)
            @test maximum(abs.(b .- out_bb_ref)) < 1e-6
        end

        @testset "A3: coef2curve vs PyKAN" begin
            x = load_reference("coef2curve_x_eval")       # (100, 2)
            g = load_reference("coef2curve_grid")         # (2, 12)
            c = load_reference("coef2curve_coef")         # (2, 3, 8)
            y_ref = load_reference("coef2curve_output")   # (100, 2, 3)
            y = coef2curve(x, g, c, K)
            @test size(y) == (BATCH, IN_DIM, OUT_DIM)
            @test maximum(abs.(y .- y_ref)) < 1e-6
        end

        @testset "A4: curve2coef round-trip" begin
            # Well-posed round-trip: x spans the full grid range so every
            # basis function is active (rank-full least squares). The reference
            # x (torch.rand in [0,1)) sits far inside the extended grid and
            # leaves some basis columns identically zero, making recovery
            # ill-posed for those columns.
            g = load_reference("coef2curve_grid")          # (2, 12)
            c = load_reference("coef2curve_coef")          # (2, 3, 8)
            x_full = rand(rng, BATCH, IN_DIM) .* 2.0 .- 1.0  # uniform in [-1, 1]
            y = coef2curve(x_full, g, c, K)
            c_rec = curve2coef(x_full, y, g, K)
            @test size(c_rec) == (IN_DIM, OUT_DIM, NUM + K)
            @test maximum(abs.(c_rec .- c)) < 1e-4
        end

        @testset "A5: extend_grid node count" begin
            g0 = load_reference("extend_grid_input")       # (2, 6)
            ge = extend_grid(g0, K)
            @test size(ge) == (IN_DIM, NUM + 1 + 2 * K)
        end

        @testset "A6: extend_grid spacing + vs PyKAN" begin
            g0 = load_reference("extend_grid_input")
            ge_ref = load_reference("extend_grid_output")  # (2, 12)
            ge = extend_grid(g0, K)
            @test maximum(abs.(ge .- ge_ref)) < 1e-12
            h = (g0[:, end] .- g0[:, 1]) ./ (size(g0, 2) - 1)
            diffs = ge[:, 2:end] .- ge[:, 1:end-1]
            @test all(isapprox.(diffs, repeat(h, 1, size(ge, 2) - 1); atol=1e-12))
        end
    end

    @testset "B) gradient paths" begin
        @testset "B1: coef gradient non-zero" begin
            x = load_reference("coef2curve_x_eval")
            g = load_reference("coef2curve_grid")
            c = load_reference("coef2curve_coef")
            grad = Zygote.gradient(coef -> sum(coef2curve(x, g, coef, K)), c)[1]
            @test !all(iszero, grad)
            @test !any(isnan, grad)
        end

        @testset "B2: grid is non-trainable (no gradient tracked)" begin
            x = load_reference("coef2curve_x_eval")
            g = load_reference("coef2curve_grid")
            c = load_reference("coef2curve_coef")
            # grid is fixed structural state (stored in st, not ps, in the KAN
            # design): a plain constant that is never differentiated.
            @test g isa Matrix{Float64}
            grad = Zygote.gradient(coef -> sum(coef2curve(x, g, coef, K)), c)[1]
            @test !all(iszero, grad)
        end

        @testset "B3: B_batch gradient flows to x" begin
            x = load_reference("b_batch_input_x")
            g = load_reference("b_batch_grid")
            grad = Zygote.gradient(xx -> sum(B_batch(xx, g, K)), x)[1]
            @test !all(iszero, grad)
            @test !any(isnan, grad)
        end

        @testset "B4: curve2coef absent from training gradient path" begin
            x = load_reference("coef2curve_x_eval")
            g = load_reference("coef2curve_grid")
            c = load_reference("coef2curve_coef")
            # coef2curve's gradient w.r.t. coef is the pure B-spline basis
            # (a linear map), with no lstsq/curve2coef involved — proving
            # curve2coef is a refine-time op, absent from the differentiable
            # forward / training path.
            b = B_batch(x, g, K)                              # (batch, in_dim, n)
            grad = Zygote.gradient(coef -> sum(coef2curve(x, g, coef, K)), c)[1]
            expected = dropdims(sum(b; dims=1); dims=1)       # (in_dim, n)
            @test all(o -> isapprox(grad[:, o, :], expected; atol=1e-8), 1:OUT_DIM)
        end
    end

    @testset "D) edge cases" begin
        @testset "D1: k=0 constant spline" begin
            b0 = B_batch(x_ref, grid_ref, 0)
            @test size(b0) == (BATCH, IN_DIM, size(grid_ref, 2) - 1)
            @test all(v -> v == 0.0 || v == 1.0, b0)
            sums = dropdims(sum(b0; dims=3); dims=3)
            @test all(isapprox.(sums, 1.0; atol=1e-12))
        end

        @testset "D2: k=5 high-order spline" begin
            base = collect(range(-1.0, 1.0; length=6))
            g0 = repeat(reshape(base, 1, :), IN_DIM, 1)       # (2, 6)
            ge = extend_grid(g0, 5)                           # (2, 16)
            x5 = rand(rng, BATCH, IN_DIM) .* 2.0 .- 1.0       # in [-1, 1]
            b5 = B_batch(x5, ge, 5)
            @test size(b5) == (BATCH, IN_DIM, size(ge, 2) - 1 - 5)
            @test all(isfinite, b5)
            sums = dropdims(sum(b5; dims=3); dims=3)
            @test all(isapprox.(sums, 1.0; atol=1e-8))
        end

        @testset "D3: degenerate grid NaN -> 0" begin
            base = [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]      # repeated knots
            g_degen = repeat(reshape(base, 1, :), IN_DIM, 1)  # (2, 7)
            xd = rand(rng, BATCH, IN_DIM) .* 2.0 .- 1.0
            b = B_batch(xd, g_degen, 3)
            @test all(v -> !isnan(v), b)
            @test all(isfinite, b)
        end

        @testset "D4: x beyond grid range does not crash" begin
            x_out = [-10.0 10.0; -10.0 10.0]                  # (2, 2), far outside
            b = B_batch(x_out, grid_ref, K)
            @test size(b) == (2, IN_DIM, NUM + K)
            @test all(v -> !isnan(v), b)
            @test all(isapprox.(b, 0.0; atol=1e-12))
        end
    end

    @testset "Type stability (@inferred)" begin
        x = load_reference("b_batch_input_x")
        g = load_reference("b_batch_grid")
        c = load_reference("coef2curve_coef")
        y = coef2curve(x, g, c, K)
        @inferred B_batch(x, g, K)
        @inferred coef2curve(x, g, c, K)
        @inferred curve2coef(x, y, g, K)
        @inferred extend_grid(g, K)
    end
end
