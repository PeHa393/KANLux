# symbolic_lib.jl — symbolic function registry, ported from PyKAN kan/utils.py
#
# Mirrors `SYMBOLIC_LIB` from PyKAN: each entry is a 3-tuple
#
#     (julia_fn, complexity, singularity_fn)
#
# where `julia_fn` is the regular forward function and `singularity_fn`
# receives `(x, y_th)` and returns a finite, threshold-limited value used
# by the symbolic layer in singularity-avoiding mode.
#
# The registry keeps PyKAN's public aliases (`x^0.5`, `1/x^0.5`) while the
# test suite checks the 27 distinct functions in one group.

"""
    _torch_nan_to_num(x)

Match PyTorch's `torch.nan_to_num` for real scalar/array inputs:
NaN -> 0, +Inf -> +floatmax, -Inf -> -floatmax.
"""
@inline function _torch_nan_to_num(x::Real)
    isnan(x) && return zero(x)
    x == Inf && return floatmax(x)
    x == -Inf && return -floatmax(x)
    return x
end

_torch_nan_to_num(x::AbstractArray) = map(_torch_nan_to_num, x)

# PyTorch-compatible domain handling. Julia's `sqrt`, `log`, `asin`, etc.
# throw `DomainError` for arguments where PyTorch returns NaN/Inf. These
# wrappers preserve the PyTorch behaviour so the symbolic layer matches the
# Python reference implementation.
_torch_sqrt(x) = ifelse.(x .>= 0, sqrt.(max.(x, zero(x))), NaN)

_torch_log(x) = ifelse.(
    x .> 0,
    log.(max.(x, floatmin(Float64))),
    ifelse.(x .== 0, -Inf, NaN),
)

_torch_asin(x) = ifelse.(abs.(x) .<= 1, asin.(clamp.(x, -1.0, 1.0)), NaN)
_torch_acos(x) = ifelse.(abs.(x) .<= 1, acos.(clamp.(x, -1.0, 1.0)), NaN)

_torch_atanh(x) = ifelse.(
    abs.(x) .< 1,
    atanh.(clamp.(x, -0.999999999999, 0.999999999999)),
    ifelse.(abs.(x) .== 1, sign.(x) * Inf, NaN),
)

_inv(x) = 1 ./ x
_inv2(x) = 1 ./ (x .^ 2)
_inv3(x) = 1 ./ (x .^ 3)
_inv4(x) = 1 ./ (x .^ 4)
_inv5(x) = 1 ./ (x .^ 5)
_pow1d5(x) = _torch_sqrt(x) .^ 3
_invsqrt(x) = 1 ./ _torch_sqrt(x)
_gaussian(x) = exp.(-(x .^ 2))
_zero(x) = x .* 0

# ── Singularity-protected variants ──

_f_inv(x, y_th) = begin
    x_th = 1 / y_th
    ifelse.(abs.(x) .< x_th, y_th / x_th .* x, _torch_nan_to_num(_inv(x)))
end

_f_inv2(x, y_th) = begin
    x_th = 1 / sqrt(y_th)
    ifelse.(abs.(x) .< x_th, y_th, _torch_nan_to_num(_inv2(x)))
end

_f_inv3(x, y_th) = begin
    x_th = 1 / cbrt(y_th)
    ifelse.(abs.(x) .< x_th, y_th / x_th .* x, _torch_nan_to_num(_inv3(x)))
end

_f_inv4(x, y_th) = begin
    x_th = 1 / y_th^(1 / 4)
    ifelse.(abs.(x) .< x_th, y_th, _torch_nan_to_num(_inv4(x)))
end

_f_inv5(x, y_th) = begin
    x_th = 1 / y_th^(1 / 5)
    ifelse.(abs.(x) .< x_th, y_th / x_th .* x, _torch_nan_to_num(_inv5(x)))
end

_f_sqrt(x, y_th) = begin
    x_th = 1 / y_th^2
    ifelse.(
        abs.(x) .< x_th,
        x_th / y_th .* x,
        _torch_nan_to_num(sqrt.(max.(abs.(x), 0)) .* sign.(x)),
    )
end

_f_power1d5(x, y_th) = abs.(x) .^ 1.5

_f_invsqrt(x, y_th) = begin
    x_th = 1 / y_th^2
    ifelse.(
        abs.(x) .< x_th,
        y_th,
        _torch_nan_to_num(1 ./ sqrt.(max.(abs.(x), floatmin(Float64)))),
    )
end

_f_log(x, y_th) = begin
    x_th = exp(-y_th)
    ifelse.(
        abs.(x) .< x_th,
        -y_th,
        _torch_nan_to_num(log.(max.(abs.(x), floatmin(Float64)))),
    )
end

_f_tan(x, y_th) = begin
    clip = mod.(x, pi)
    delta = pi / 2 - atan(y_th)
    ifelse.(
        abs.(clip .- pi / 2) .< delta,
        -y_th / delta .* (clip .- pi / 2),
        _torch_nan_to_num(tan.(clip)),
    )
end

_f_arctanh(x, y_th) = begin
    delta = 1 - tanh(y_th) + 1e-4
    ifelse.(
        abs.(x) .> 1 - delta,
        y_th .* sign.(x),
        _torch_nan_to_num(atanh.(clamp.(x, -1 + delta, 1 - delta))),
    )
end

_f_arcsin(x, y_th) = ifelse.(
    abs.(x) .> 1,
    pi / 2 .* sign.(x),
    _torch_nan_to_num(asin.(clamp.(x, -1.0, 1.0))),
)

_f_arccos(x, y_th) = ifelse.(
    abs.(x) .> 1,
    pi / 2 .* (1 .- sign.(x)),
    _torch_nan_to_num(acos.(clamp.(x, -1.0, 1.0))),
)

_f_exp(x, y_th) = begin
    x_th = log(y_th)
    ifelse.(x .> x_th, y_th, exp.(x))
end

"""
    SYMBOLIC_LIB

Registry of symbolic activation functions available to `SymbolicKANLayer`.

The values are `(julia_fn, complexity, singularity_fn)`, matching PyKAN's
`SYMBOLIC_LIB`. Complexity is a non-negative integer used for symbolic
regression preference, not for runtime behaviour.
"""
const SYMBOLIC_LIB = Dict{String,Tuple{Function,Int,Function}}(
    # identity and monomials
    "x" => (x -> x, 1, (x, y_th) -> x),
    "x^2" => (x -> x .^ 2, 2, (x, y_th) -> x .^ 2),
    "x^3" => (x -> x .^ 3, 3, (x, y_th) -> x .^ 3),
    "x^4" => (x -> x .^ 4, 3, (x, y_th) -> x .^ 4),
    "x^5" => (x -> x .^ 5, 3, (x, y_th) -> x .^ 5),

    # reciprocal powers
    "1/x" => (_inv, 2, _f_inv),
    "1/x^2" => (_inv2, 2, _f_inv2),
    "1/x^3" => (_inv3, 3, _f_inv3),
    "1/x^4" => (_inv4, 4, _f_inv4),
    "1/x^5" => (_inv5, 5, _f_inv5),

    # roots and related powers
    "sqrt" => (_torch_sqrt, 2, _f_sqrt),
    "x^0.5" => (_torch_sqrt, 2, _f_sqrt),
    "x^1.5" => (_pow1d5, 4, _f_power1d5),
    "1/sqrt(x)" => (_invsqrt, 2, _f_invsqrt),
    "1/x^0.5" => (_invsqrt, 2, _f_invsqrt),

    # exponential and logarithm
    "exp" => (exp, 2, _f_exp),
    "log" => (_torch_log, 2, _f_log),

    # elementary functions
    "abs" => (abs, 3, (x, y_th) -> abs.(x)),
    "sin" => (sin, 2, (x, y_th) -> sin.(x)),
    "cos" => (cos, 2, (x, y_th) -> cos.(x)),
    "tan" => (tan, 3, _f_tan),
    "tanh" => (tanh, 3, (x, y_th) -> tanh.(x)),
    "sgn" => (sign, 3, (x, y_th) -> sign.(x)),

    # inverse trigonometric functions
    "arcsin" => (_torch_asin, 4, _f_arcsin),
    "arccos" => (_torch_acos, 4, _f_arccos),
    "arctan" => (atan, 4, (x, y_th) -> atan.(x)),
    "arctanh" => (_torch_atanh, 4, _f_arctanh),

    # miscellaneous symbolic functions
    "gaussian" => (_gaussian, 3, (x, y_th) -> _gaussian(x)),
    "0" => (_zero, 0, (x, y_th) -> _zero(x)),
)
