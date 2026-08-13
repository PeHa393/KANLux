# test_multkan.jl — Phase 6: MultKAN container forward, gradients, state, edges

include("test_infra.jl")

function _multkan_reference(m)
    ref_act_ps = (
        (coef=load_reference("multkan_act0_coef"),
         scale_base=load_reference("multkan_act0_scale_base"),
         scale_sp=load_reference("multkan_act0_scale_sp")),
        (coef=load_reference("multkan_act1_coef"),
         scale_base=load_reference("multkan_act1_scale_base"),
         scale_sp=load_reference("multkan_act1_scale_sp")),
    )
    ref_act_st = (
        (grid=load_reference("multkan_act0_grid"),
         mask=load_reference("multkan_act0_mask")),
        (grid=load_reference("multkan_act1_grid"),
         mask=load_reference("multkan_act1_mask")),
    )
    ps0, st0 = Lux.setup(rng, m)
    ps = merge(ps0, (
        act_fun=ref_act_ps,
        node_bias=(load_reference("multkan_node_bias0"), load_reference("multkan_node_bias1")),
        node_scale=(load_reference("multkan_node_scale0"), load_reference("multkan_node_scale1")),
        subnode_bias=(load_reference("multkan_subnode_bias0"), load_reference("multkan_subnode_bias1")),
        subnode_scale=(load_reference("multkan_subnode_scale0"), load_reference("multkan_subnode_scale1")),
    ))
    st = merge(st0, (act_fun=ref_act_st,))
    return ps, st
end

@testset "MultKAN (Phase 6)" begin
    @testset "A) forward values" begin
        x_ref = load_reference("multkan_input")

        @testset "A1: output shape" begin
            m = MultKAN([2, 5, 1]; grid=5, k=3)
            ps, st = _multkan_reference(m)
            y = m(x_ref, ps, st)[1]
            @test size(y) == (100, 1)
        end

        @testset "A2: forward vs PyKAN (symbolic enabled)" begin
            m = MultKAN([2, 5, 1]; grid=5, k=3, symbolic_enabled=true)
            ps, st = _multkan_reference(m)
            y_ref = load_reference("multkan_output_full")
            y = m(x_ref, ps, st)[1]
            @test maximum(abs.(y .- y_ref)) < 1e-5
        end

        @testset "A3: forward vs PyKAN (symbolic disabled)" begin
            m = MultKAN([2, 5, 1]; grid=5, k=3, symbolic_enabled=false)
            ps, st = _multkan_reference(m)
            y_ref = load_reference("multkan_output_numerical")
            y = m(x_ref, ps, st)[1]
            @test maximum(abs.(y .- y_ref)) < 1e-5
        end

        @testset "A4: multiplication nodes" begin
            x = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0]
            y_h = KANLux._multiply_subnodes(x, 0, 2, 2, true)
            ok_h = y_h == [2.0 12.0; 30.0 56.0]

            x2 = [1.0 2.0 3.0 4.0 5.0 6.0; 7.0 8.0 9.0 10.0 11.0 12.0]
            y_n = KANLux._multiply_subnodes(x2, 1, 2, [2, 3], false)
            ok_n = y_n == [6.0 120.0; 72.0 1320.0]
            @test ok_h && ok_n
        end

        @testset "A5: depth=1" begin
            m = MultKAN([2, 1]; grid=3, k=3)
            ps, st = Lux.setup(rng, m)
            y = m(randn(rng, 16, 2), ps, st)[1]
            @test size(y) == (16, 1) && all(isfinite, y)
        end
    end

    gm = MultKAN([2, 1]; grid=3, k=3, symbolic_enabled=true)
    fix_symbolic!(gm.symbolic_fun[1], 1, 1, "sin")
    gm.symbolic_fun[1].mask[1, 1] = 1.0
    ps_g, st_g = Lux.setup(rng, gm)
    x_g = randn(rng, 32, 2)
    g = Zygote.gradient(p -> sum(gm(x_g, p, st_g)[1]), ps_g)[1]

    @testset "B) gradient paths" begin
        @testset "B1: act_fun.coef" begin
            @test !all(iszero, g.act_fun[1].coef) && !any(isnan, g.act_fun[1].coef)
        end

        @testset "B2: act_fun.scale_sp" begin
            @test !all(iszero, g.act_fun[1].scale_sp) && !any(isnan, g.act_fun[1].scale_sp)
        end

        @testset "B3: node_bias" begin
            @test !all(iszero, g.node_bias[1]) && !any(isnan, g.node_bias[1])
        end

        @testset "B4: node_scale" begin
            @test !all(iszero, g.node_scale[1]) && !any(isnan, g.node_scale[1])
        end

        @testset "B5: subnode_bias" begin
            @test !all(iszero, g.subnode_bias[1]) && !any(isnan, g.subnode_bias[1])
        end

        @testset "B6: subnode_scale" begin
            @test !all(iszero, g.subnode_scale[1]) && !any(isnan, g.subnode_scale[1])
        end

        @testset "B7: symbolic_fun.affine" begin
            @test !all(iszero, g.symbolic_fun[1].affine) && !any(isnan, g.symbolic_fun[1].affine)
        end

        @testset "B8: grid/mask have no gradient" begin
            @test !(:grid in propertynames(g.act_fun[1])) &&
                  !(:mask in propertynames(g.act_fun[1])) &&
                  !(:mask in propertynames(g.symbolic_fun[1]))
        end
    end

    @testset "C) state transfer" begin
        m = MultKAN([2, 1]; grid=3, k=3)
        ps, st = Lux.setup(rng, m)
        x = randn(rng, 16, 2)
        y1, st2 = m(x, ps, st)

        @testset "C1: nested structure preserved" begin
            @test propertynames(st2) == propertynames(st) &&
                  st2.act_fun isa Tuple && st2.symbolic_fun isa Tuple
        end

        @testset "C2: sublayer states independent" begin
            @test st2.act_fun[1].grid === st.act_fun[1].grid &&
                  st2.act_fun[1].mask === st.act_fun[1].mask &&
                  st2.symbolic_fun[1].mask === st.symbolic_fun[1].mask
        end

        @testset "C3: state reusable" begin
            y2 = m(x, ps, st)[1]
            @test y2 == y1
        end
    end

    @testset "D) edge cases" begin
        @testset "D1: depth=5" begin
            m = MultKAN([2, 2, 2, 2, 2, 1]; grid=3, k=3)
            ps, st = Lux.setup(rng, m)
            y = m(randn(rng, 8, 2), ps, st)[1]
            @test size(y) == (8, 1) && all(isfinite, y)
        end

        @testset "D2: width in full list format" begin
            m = MultKAN([[2, 0], [2, 0], [1, 0]]; grid=3, k=3)
            ps, st = Lux.setup(rng, m)
            y = m(randn(rng, 8, 2), ps, st)[1]
            @test size(y) == (8, 1) && all(isfinite, y)
        end

        @testset "D3: batch=1" begin
            m = MultKAN([2, 5, 1]; grid=3, k=3)
            ps, st = Lux.setup(rng, m)
            y = m(randn(rng, 1, 2), ps, st)[1]
            @test size(y) == (1, 1) && all(isfinite, y)
        end

        @testset "D4: zero-width hidden layer rejected" begin
            @test_throws ArgumentError MultKAN([2, 0, 1]; grid=3, k=3)
        end

        @testset "D5: forward type stability" begin
            for symbolic_enabled in (false, true)
                m = MultKAN([2, 1]; grid=3, k=3,
                            symbolic_enabled=symbolic_enabled)
                ps, st = Lux.setup(rng, m)
                x = randn(rng, 8, 2)
                @test begin
                    @inferred m(x, ps, st)
                    true
                end
            end
        end

        @testset "D6: multiplication-node type stability" begin
            m = MultKAN([[2, 0], [1, 2], [1, 0]]; grid=3, k=3,
                        mult_arity=2, symbolic_enabled=true)
            ps, st = Lux.setup(rng, m)
            x = randn(rng, 8, 2)
            @test begin
                @inferred m(x, ps, st)
                true
            end
        end
    end
end
