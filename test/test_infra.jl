# test_infra.jl — Shared test helper functions for all KANLux test files
#
# Include this file at the top of each test script:
#   include("test_infra.jl")
#
# Provides:
#   - load_reference(name)        : load .npy reference tensor from reference/
#   - check_gradient(...)         : verify a ps key has non-zero gradient
#   - check_zero_gradient(...)    : verify a st key has no gradient
#   - check_st_roundtrip(...)     : verify state structure is preserved after forward
#   - check_type_stability(...)   : verify @inferred passes on forward pass
#   - rng                         : shared RNG (Xoshiro(42)) for deterministic tests

using KANLux
using Test
using Random
using Lux
using Zygote
using NPZ
using Statistics

# ── Shared RNG ──
const rng = Xoshiro(42)

# ── Reference data loader ──

"""
    load_reference(name::String)

Load a .npy reference file from `test/reference/<name>.npy`.
Returns a Julia Array (Float64 by default, matching PyKAN's torch.float64).
"""
function load_reference(name::String)
    filepath = joinpath(@__DIR__, "reference", "$name.npy")
    @assert isfile(filepath) "Reference file not found: $filepath"
    arr = NPZ.npzread(filepath)
    return arr
end

# ── Gradient checks ──

"""
    check_gradient(ps_key::Symbol, layer, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                   msg::String = "")

Verify that the parameter field named `ps_key` has a non-zero gradient w.r.t.
`sum(layer(x, ps, st)[1])`. Fails if gradient is all-zero or contains NaN.
"""
function check_gradient(ps_key::Symbol, layer, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                        msg::String = "")
    grads = Zygote.gradient(ps -> sum(layer(x, ps, st)[1]), ps)[1]
    g = getfield(grads, ps_key)

    @test !all(iszero, g)
    @test !any(isnan, g)
    return g
end

"""
    check_zero_gradient(st_key::Symbol, layer, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                        msg::String = "")

Verify that the state field named `st_key` either has no gradient (nothing)
or its gradient is all-zero. State parameters should not participate in
gradient computation.
"""
function check_zero_gradient(st_key::Symbol, layer, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                             msg::String = "")
    # We only take gradient w.r.t. ps — st is not differentiated by Lux design
    grads = Zygote.gradient(ps -> sum(layer(x, ps, st)[1]), ps)[1]

    # The st key shouldn't appear in ps gradients
    @test !(st_key in propertynames(grads))
end

"""
    check_st_roundtrip(layer, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                       msg::String = "")

Verify that after a forward pass, the state structure is preserved:
- st_new has the same keys as st
- Non-trainable state values (grid, mask) are unchanged
"""
function check_st_roundtrip(layer, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                            msg::String = "")
    _, st_new = layer(x, ps, st)

    # Same keys
    @test sort(collect(propertynames(st_new))) == sort(collect(propertynames(st)))

    # Each non-trainable field should be unchanged (value equality)
    for key in propertynames(st)
        old_val = getfield(st, key)
        new_val = getfield(st_new, key)
        if old_val isa AbstractArray
            @test old_val == new_val
        end
    end
end

"""
    check_type_stability(layer, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                         msg::String = "")

Verify that the forward pass is type-stable by checking that @inferred succeeds.
"""
function check_type_stability(layer, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                              msg::String = "")
    @inferred layer(x, ps, st)
    @test true  # reached without error
end

# ── Utility: finite-difference gradient cross-check ──

"""
    check_fd_gradient(ps_key::Symbol, layer, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                      atol::Float64 = 1e-4, msg::String = "")

Cross-validate Zygote gradient against 5-point central finite differences.
Returns the max absolute difference.
"""
function check_fd_gradient(ps_key::Symbol, layer, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple;
                           atol::Float64 = 1e-4, msg::String = "")
    g_ad = getfield(Zygote.gradient(ps -> sum(layer(x, ps, st)[1]), ps)[1], ps_key)

    # Finite difference approximation
    ϵ = 1e-5
    p = getfield(ps, ps_key)
    p_flat = vec(copy(p))
    g_flat = similar(p_flat)

    for i in eachindex(p_flat)
        orig = p_flat[i]
        # 5-point stencil: (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
        f_vals = zeros(4)
        for (j, h) in enumerate([2ϵ, ϵ, -ϵ, -2ϵ])
            p_flat[i] = orig + h
            p_new = reshape(p_flat, size(p))
            ps_new = setproperties(ps, (ps_key => p_new,))
            f_vals[j] = sum(layer(x, ps_new, st)[1])
        end
        g_flat[i] = (-f_vals[1] + 8f_vals[2] - 8f_vals[3] + f_vals[4]) / (12ϵ)
        p_flat[i] = orig  # restore
    end

    g_fd = reshape(g_flat, size(g_ad))
    max_diff = maximum(abs.(g_ad .- g_fd))
    @test max_diff < atol
    return max_diff
end

# ── Utility: small deterministic test input ──

"""
    test_input(in_dim::Int, batch_size::Int = 16)

Generate a small deterministic input matrix for testing.
Uses a fixed seed to ensure reproducibility.
"""
function test_input(in_dim::Int, batch_size::Int = 16)
    return randn(Xoshiro(123), Float64, in_dim, batch_size)
end
