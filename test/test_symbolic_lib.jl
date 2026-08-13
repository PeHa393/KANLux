# test_symbolic_lib.jl — Phase 2: symbolic function registry
#
# Covers:
#   A) 27 distinct functions x 5 points, matching PyKAN/PyTorch semantics
#   D) 9 singularity-protection checks at singular boundaries

include("test_infra.jl")

# Julia's built-in inverse/trig functions throw DomainError where PyTorch
# returns NaN/Inf. These small references mirror PyKAN's torch behaviour so
# the regular function field can be tested on the full requested grid.
_ref_sqrt(x) = x >= 0 ? sqrt(x) : NaN
_ref_log(x) = x > 0 ? log(x) : (x == 0 ? -Inf : NaN)
_ref_asin(x) = abs(x) <= 1 ? asin(x) : NaN
_ref_acos(x) = abs(x) <= 1 ? acos(x) : NaN
_ref_atanh(x) = abs(x) < 1 ? atanh(x) : (abs(x) == 1 ? sign(x) * Inf : NaN)

const TEST_POINTS = (-2.0, -1.0, 0.0, 1.0, 2.0)

const TEST_NAMES = (
    "x", "x^2", "x^3", "x^4", "x^5",
    "1/x", "1/x^2", "1/x^3", "1/x^4", "1/x^5",
    "sqrt", "x^1.5", "1/sqrt(x)",
    "exp", "log", "abs", "sin", "cos", "tan", "tanh", "sgn",
    "arcsin", "arccos", "arctan", "arctanh",
    "gaussian", "0",
)

const REFS = Dict{String,Function}(
    "x" => (x -> x),
    "x^2" => (x -> x^2),
    "x^3" => (x -> x^3),
    "x^4" => (x -> x^4),
    "x^5" => (x -> x^5),
    "1/x" => (x -> 1 / x),
    "1/x^2" => (x -> 1 / x^2),
    "1/x^3" => (x -> 1 / x^3),
    "1/x^4" => (x -> 1 / x^4),
    "1/x^5" => (x -> 1 / x^5),
    "sqrt" => _ref_sqrt,
    "x^1.5" => (x -> _ref_sqrt(x)^3),
    "1/sqrt(x)" => (x -> 1 / _ref_sqrt(x)),
    "exp" => exp,
    "log" => _ref_log,
    "abs" => abs,
    "sin" => sin,
    "cos" => cos,
    "tan" => tan,
    "tanh" => tanh,
    "sgn" => sign,
    "arcsin" => _ref_asin,
    "arccos" => _ref_acos,
    "arctan" => atan,
    "arctanh" => _ref_atanh,
    "gaussian" => (x -> exp(-x^2)),
    "0" => (x -> 0.0),
)

_values_match(a, b) = (isnan(a) && isnan(b)) || (a == b) || isapprox(a, b; atol=1e-12)

@testset "Symbolic function library (Phase 2)" begin
    @testset "A) 27-function correctness" begin
        @testset "$name" for name in TEST_NAMES
            fn = SYMBOLIC_LIB[name][1]
            ref = REFS[name]
            @test all(_values_match(fn(x), ref(x)) for x in TEST_POINTS)
        end
    end

    @testset "D) singularity safety" begin
        @testset "D1: 1/x at x=0" begin
            y = SYMBOLIC_LIB["1/x"][3](0.0, 10.0)
            @test isfinite(y)
        end

        @testset "D2: 1/x^2 at x=0" begin
            y = SYMBOLIC_LIB["1/x^2"][3](0.0, 10.0)
            @test isfinite(y)
        end

        @testset "D3: sqrt at x=0" begin
            y = SYMBOLIC_LIB["sqrt"][3](0.0, 10.0)
            @test isfinite(y)
        end

        @testset "D4: log at x=0" begin
            y = SYMBOLIC_LIB["log"][3](0.0, 10.0)
            @test isfinite(y) && isapprox(y, -10.0; atol=1e-12)
        end

        @testset "D5: exp at large x" begin
            y = SYMBOLIC_LIB["exp"][3](100.0, 10.0)
            @test isfinite(y) && isapprox(y, 10.0; atol=1e-12)
        end

        @testset "D6: tan at pi/2" begin
            y = SYMBOLIC_LIB["tan"][3](pi / 2, 10.0)
            @test isfinite(y)
        end

        @testset "D7: arctanh at x=1" begin
            y = SYMBOLIC_LIB["arctanh"][3](1.0, 10.0)
            @test isfinite(y)
        end

        @testset "D8: arcsin at x=1.1" begin
            y = SYMBOLIC_LIB["arcsin"][3](1.1, 10.0)
            @test isfinite(y)
        end

        @testset "D9: arccos at x=1.1" begin
            y = SYMBOLIC_LIB["arccos"][3](1.1, 10.0)
            @test isfinite(y)
        end
    end
end
