# test_data.jl — Phase 3: dataset creation utility
#
# Covers:
#   A) forward values (3): shapes, same-seed determinism, different seeds
#   D) edge cases (3): n_var=1, test_num=0, custom ranges
#   R) reference loading (3): file existence, correct load, determinism

include("test_infra.jl")

@testset "Data utilities (Phase 3)" begin
    @testset "A) forward values" begin
        @testset "A1: shapes" begin
            train_x, train_y, test_x, test_y =
                create_dataset(x -> x .^ 2, 3, (-1.0, 1.0), 100, 50)
            @test size(train_x) == (100, 3)
            @test size(train_y) == (100, 3)
            @test size(test_x) == (50, 3)
            @test size(test_y) == (50, 3)
        end

        @testset "A2: same-seed determinism" begin
            a = create_dataset(x -> x .^ 2, 3, (-1.0, 1.0), 100, 50, 7)
            b = create_dataset(x -> x .^ 2, 3, (-1.0, 1.0), 100, 50, 7)
            @test all(a[i] == b[i] for i in 1:4)
        end

        @testset "A3: different seeds differ" begin
            train_x0, _, _, _ = create_dataset(x -> x .^ 2, 3, (-1.0, 1.0), 100, 50, 0)
            train_x1, _, _, _ = create_dataset(x -> x .^ 2, 3, (-1.0, 1.0), 100, 50, 1)
            @test train_x0 != train_x1
        end
    end

    @testset "D) edge cases" begin
        @testset "D1: n_var=1" begin
            train_x, train_y, _, _ = create_dataset(x -> x .^ 2, 1, (-1.0, 1.0), 20, 0)
            @test size(train_x) == (20, 1)
            @test size(train_y) == (20, 1)
        end

        @testset "D2: test_num=0" begin
            _, _, test_x, test_y = create_dataset(x -> x .^ 2, 3, (-1.0, 1.0), 100, 0)
            @test isempty(test_x)
            @test isempty(test_y)
            @test size(test_x) == (0, 3)
            @test size(test_y) == (0, 3)
        end

        @testset "D3: custom ranges" begin
            train_x, _, test_x, _ = create_dataset(x -> x .^ 2, 2, [0.0, 10.0], 100, 50)
            @test all(0.0 .<= train_x .<= 10.0)
            @test all(0.0 .<= test_x .<= 10.0)
        end

        @testset "D4: scalar label rejected for multi-sample input" begin
            @test_throws DimensionMismatch create_dataset(
                x -> 1.0, 2, (-1.0, 1.0), 10, 2
            )
        end

        @testset "D5: scalar label allowed for empty/single-sample output" begin
            train_x, train_y, test_x, test_y =
                create_dataset(x -> 1.0, 2, (-1.0, 1.0), 1, 0)
            @test size(train_y) == (1, 1)
            @test size(test_y) == (0, 1)
        end
    end

    @testset "R) reference loading" begin
        @testset "R1: .npy files exist" begin
            refs = readdir(joinpath(@__DIR__, "reference"))
            npy_files = filter(f -> endswith(f, ".npy"), refs)
            @test length(npy_files) >= 10
        end

        @testset "R2: correct load" begin
            arr = load_reference("b_batch_output")
            @test size(arr) == (100, 2, 8)
            @test eltype(arr) == Float64
        end

        @testset "R3: reference determinism" begin
            a = load_reference("b_batch_output")
            b = load_reference("b_batch_output")
            @test a == b
        end
    end
end
