# data.jl — dataset creation utility, ported from PyKAN kan/utils.py
#
# `create_dataset` generates deterministic train/test inputs and applies the
# user-provided function in the same column-major convention as PyKAN:
# `f` receives an `(N, n_var)` matrix and returns a matrix or vector.
# Vector labels are reshaped to `(N, 1)` so the result is always matrix-like.

using Random

"""
    create_dataset(f, n_var::Int, ranges=(-1.0, 1.0),
                   train_num::Int=1000, test_num::Int=1000, seed::Int=0)

Generate a synthetic regression dataset.

# Arguments
- `f`: function mapping an `(N, n_var)` input matrix to labels.
- `n_var`: number of input variables.
- `ranges`: either a length-2 scalar range used for every variable, or a
  per-variable range collection (`n_var × 2` matrix or vector of tuples).
- `train_num`, `test_num`: number of train/test samples.
- `seed`: RNG seed for deterministic generation.

# Returns
- `(train_input, train_label, test_input, test_label)`, all Float64 arrays.
"""
function create_dataset(f, n_var::Int, ranges=(-1.0, 1.0),
                        train_num::Int=1000, test_num::Int=1000, seed::Int=0)
    ranges_norm = _normalize_ranges(ranges, n_var)
    rng = Xoshiro(seed)

    train_input = Matrix{Float64}(undef, train_num, n_var)
    test_input = Matrix{Float64}(undef, test_num, n_var)

    for i in 1:n_var
        lo, hi = ranges_norm[i]
        train_input[:, i] .= lo .+ (hi - lo) .* rand(rng, train_num)
        test_input[:, i] .= lo .+ (hi - lo) .* rand(rng, test_num)
    end

    train_label = _as_label_matrix(f(train_input), train_num)
    test_label = _as_label_matrix(f(test_input), test_num)

    return train_input, train_label, test_input, test_label
end

"""
    _normalize_ranges(ranges, n_var)

Convert `ranges` into a vector of `(lo, hi)` tuples, one per input variable.
"""
function _normalize_ranges(ranges, n_var::Int)
    # Common range for all variables: (-1, 1), [0, 10], etc.
    if ranges isa Tuple && length(ranges) == 2 &&
       !(ranges[1] isa Tuple || ranges[1] isa AbstractVector)
        return [(float(ranges[1]), float(ranges[2])) for _ in 1:n_var]
    end

    if ranges isa AbstractVector && length(ranges) == 2 &&
       !(ranges[1] isa Tuple || ranges[1] isa AbstractVector)
        return [(float(ranges[1]), float(ranges[2])) for _ in 1:n_var]
    end

    # Per-variable matrix: n_var x 2.
    if ranges isa AbstractMatrix && size(ranges) == (n_var, 2)
        return [(float(ranges[i, 1]), float(ranges[i, 2])) for i in 1:n_var]
    end

    # Per-variable collection: vector of tuples/vectors.
    if ranges isa AbstractVector && length(ranges) == n_var &&
       all(r -> (r isa Tuple && length(r) == 2) ||
                (r isa AbstractVector && length(r) == 2), ranges)
        return [(float(r[1]), float(r[2])) for r in ranges]
    end

    throw(ArgumentError("ranges must be a length-2 scalar range or an n_var-length per-variable range collection"))
end

function _as_label_matrix(y, n::Int)
    if y isa Number
        n == 0 && return zeros(Float64, 0, 1)
        n == 1 || throw(DimensionMismatch(
            "label is a scalar but expected $n rows; return a vector or matrix instead"
        ))
        return reshape([float(y)], 1, 1)
    elseif y isa AbstractVector
        length(y) == n || throw(DimensionMismatch("label vector has length $(length(y)); expected $n"))
        return reshape(float.(y), n, 1)
    elseif y isa AbstractMatrix
        size(y, 1) == n || throw(DimensionMismatch("label matrix has $(size(y, 1)) rows; expected $n"))
        return float.(y)
    end

    throw(ArgumentError("dataset function must return a scalar, vector, or matrix"))
end
