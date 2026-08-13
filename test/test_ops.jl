# test_ops.jl — Phase 7: structural prune and grid refine

include("test_infra.jl")

@testset "Structural operations (Phase 7)" begin
    @testset "A) prune" begin
        @testset "A1: shape shrink" begin
            m = MultKAN([2, 3, 1]; grid=3, k=3, symbolic_enabled=false)
            ps, st = Lux.setup(rng, m)
            ps.act_fun[1].coef[:, 3, :] .= 0.0
            ps.act_fun[1].scale_base[:, 3] .= 0.0
            ps.act_fun[1].scale_sp[:, 3] .= 0.0
            st.act_fun[1].mask[:, 3] .= 0.0
            x = randn(rng, 32, 2)
            res = prune(m, ps, st, x; node_th=0.0, edge_th=0.0)
            @test res.model.width[2] == [2, 0]
        end

        @testset "A2: pruned forward runnable" begin
            m = MultKAN([2, 3, 1]; grid=3, k=3, symbolic_enabled=false)
            ps, st = Lux.setup(rng, m)
            ps.act_fun[1].coef[:, 3, :] .= 0.0
            ps.act_fun[1].scale_base[:, 3] .= 0.0
            ps.act_fun[1].scale_sp[:, 3] .= 0.0
            st.act_fun[1].mask[:, 3] .= 0.0
            x = randn(rng, 16, 2)
            res = prune(m, ps, st, x; node_th=0.0, edge_th=0.0)
            y = res.model(x, res.ps, res.st)[1]
            @test size(y) == (16, 1) && all(isfinite, y)
        end

        @testset "A3: threshold=0 keeps all" begin
            m = MultKAN([2, 3, 1]; grid=3, k=3, symbolic_enabled=false)
            ps, st = Lux.setup(rng, m)
            x = randn(rng, 32, 2)
            res = prune(m, ps, st, x; node_th=0.0, edge_th=0.0)
            @test res.model.width == m.width
        end

        @testset "A4: parameter slicing correct" begin
            m = MultKAN([2, 3, 1]; grid=3, k=3, symbolic_enabled=false)
            ps, st = Lux.setup(rng, m)
            ps.act_fun[1].coef[:, 3, :] .= 0.0
            ps.act_fun[1].scale_base[:, 3] .= 0.0
            ps.act_fun[1].scale_sp[:, 3] .= 0.0
            st.act_fun[1].mask[:, 3] .= 0.0
            x = randn(rng, 32, 2)
            res = prune(m, ps, st, x; node_th=0.0, edge_th=0.0)

            ok = res.ps.act_fun[1].coef == ps.act_fun[1].coef[:, [1, 2], :] &&
                 res.ps.act_fun[1].scale_base == ps.act_fun[1].scale_base[:, [1, 2]] &&
                 res.ps.act_fun[1].scale_sp == ps.act_fun[1].scale_sp[:, [1, 2]] &&
                 res.st.act_fun[1].mask == st.act_fun[1].mask[:, [1, 2]] &&
                 res.ps.node_bias[1] == ps.node_bias[1][[1, 2]] &&
                 res.ps.subnode_scale[1] == ps.subnode_scale[1][[1, 2]]
            @test ok
        end

        @testset "A5: pruned state can initialize" begin
            m = MultKAN([2, 3, 1]; grid=3, k=3, symbolic_enabled=false)
            ps, st = Lux.setup(rng, m)
            ps.act_fun[1].coef[:, 3, :] .= 0.0
            ps.act_fun[1].scale_base[:, 3] .= 0.0
            ps.act_fun[1].scale_sp[:, 3] .= 0.0
            st.act_fun[1].mask[:, 3] .= 0.0
            x = randn(rng, 16, 2)
            res = prune(m, ps, st, x; node_th=0.0, edge_th=0.0)
            ps2, st2 = Lux.setup(rng, res.model)
            @test ps2 isa NamedTuple && st2 isa NamedTuple
        end

        @testset "A6: symbolic layers pruned in sync" begin
            m = MultKAN([2, 3, 1]; grid=3, k=3, symbolic_enabled=true)
            fix_symbolic!(m.symbolic_fun[1], 2, 3, "sin")
            m.symbolic_fun[1].mask[3, 2] = 1.0
            ps, st = Lux.setup(rng, m)
            ps.act_fun[1].coef[:, 3, :] .= 0.0
            ps.act_fun[1].scale_base[:, 3] .= 0.0
            ps.act_fun[1].scale_sp[:, 3] .= 0.0
            st.act_fun[1].mask[:, 3] .= 0.0
            x = randn(rng, 16, 2)
            res = prune(m, ps, st, x; node_th=0.0, edge_th=0.0)
            ok = res.st.symbolic_fun[1].mask == st.symbolic_fun[1].mask[[1, 2], :] &&
                 res.st.symbolic_fun[1].funs_name == st.symbolic_fun[1].funs_name[[1, 2], :] &&
                 res.ps.symbolic_fun[1].affine == ps.symbolic_fun[1].affine[[1, 2], :, :]
            @test ok
        end

        @testset "A7: multiplication nodes preserved" begin
            m = MultKAN([[2, 0], [1, 2], [1, 0]]; grid=3, k=3, mult_arity=2,
                        symbolic_enabled=false)
            ps, st = Lux.setup(rng, m)
            x = randn(rng, 16, 2)
            res = prune(m, ps, st, x; node_th=0.0, edge_th=0.0)
            y = res.model(x, res.ps, res.st)[1]
            @test res.model.width[2][2] == 2 && size(y) == (16, 1) && all(isfinite, y)
        end
    end

    @testset "B) refine" begin
        m_ref = MultKAN([2, 5, 1]; grid=3, k=3, symbolic_enabled=false)
        ps_ref, st_ref = Lux.setup(rng, m_ref)
        x_fit = (rand(rng, 64, 2) .- 0.5)
        res_ref = refine(m_ref, ps_ref, st_ref, x_fit, 10)

        @testset "B1: grid extended" begin
            @test res_ref.model.grid == 10 &&
                  size(res_ref.st.act_fun[1].grid, 2) == 17
        end

        @testset "B2: function preserved" begin
            x_test = (rand(rng, 64, 2) .- 0.5)
            y_old = m_ref(x_test, ps_ref, st_ref)[1]
            y_new = res_ref.model(x_test, res_ref.ps, res_ref.st)[1]
            @test maximum(abs.(y_old .- y_new)) < 1e-3
        end

        @testset "B3: coef dimension correct" begin
            @test size(res_ref.ps.act_fun[1].coef) == (2, 5, 13)
        end

        @testset "B4: symbolic layers preserved" begin
            m = MultKAN([2, 2, 1]; grid=3, k=3, symbolic_enabled=true)
            fix_symbolic!(m.symbolic_fun[1], 1, 1, "x^2")
            m.symbolic_fun[1].mask[1, 1] = 1.0
            ps, st = Lux.setup(rng, m)
            x = (rand(rng, 32, 2) .- 0.5)
            res = refine(m, ps, st, x, 6)
            ok = res.ps.symbolic_fun[1].affine == ps.symbolic_fun[1].affine &&
                 res.st.symbolic_fun[1].mask == st.symbolic_fun[1].mask &&
                 res.st.symbolic_fun[1].funs_name == st.symbolic_fun[1].funs_name
            @test ok
        end

        @testset "B5: multiple refine" begin
            res2 = refine(res_ref.model, res_ref.ps, res_ref.st, x_fit, 20)
            x_test = (rand(rng, 32, 2) .- 0.5)
            y = res2.model(x_test, res2.ps, res2.st)[1]
            @test res2.model.grid == 20 && size(y) == (32, 1) && all(isfinite, y)
        end

        @testset "B6: refined state can initialize" begin
            ps2, st2 = Lux.setup(rng, res_ref.model)
            @test ps2 isa NamedTuple && st2 isa NamedTuple
        end
    end

    @testset "C) general invariants" begin
        @testset "C1: original model unchanged" begin
            m = MultKAN([2, 3, 1]; grid=3, k=3, symbolic_enabled=false)
            ps, st = Lux.setup(rng, m)
            coef_before = copy(ps.act_fun[1].coef)
            grid_before = copy(st.act_fun[1].grid)
            x = randn(rng, 16, 2)
            prune(m, ps, st, x; node_th=0.0, edge_th=0.0)
            @test ps.act_fun[1].coef == coef_before && st.act_fun[1].grid == grid_before
        end

        @testset "C2: input dimension preserved" begin
            m = MultKAN([3, 4, 1]; grid=3, k=3, symbolic_enabled=false)
            ps, st = Lux.setup(rng, m)
            x = randn(rng, 16, 3)
            res = prune(m, ps, st, x; node_th=0.0, edge_th=0.0)
            @test res.model.width_in[1] == 3
        end

        @testset "C3: output dimension preserved" begin
            m = MultKAN([2, 3, 2]; grid=3, k=3, symbolic_enabled=false)
            ps, st = Lux.setup(rng, m)
            x = randn(rng, 16, 2)
            res = refine(m, ps, st, x, 8)
            @test res.model.width_in[end] == 2
        end
    end
end
