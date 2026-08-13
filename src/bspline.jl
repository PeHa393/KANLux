# bspline.jl — B-spline math primitives, ported from PyKAN kan/spline.py
#
# All functions are vectorized and type-stable, operating on Float64 arrays
# (KANs need double precision for convergence). Array layouts match PyKAN
# exactly so results can be compared against the .npy reference data:
#
#   x     : (batch, in_dim)
#   grid  : (in_dim, S)  with S = G + 2k + 1   (G = number of grid intervals)
#   coef  : (in_dim, out_dim, G + k)
#
#   B_batch     : Cox-de Boor recursion        -> (batch, in_dim, G + k)
#   coef2curve  : einsum('ijk,jlk->ijl')       -> (batch, in_dim, out_dim)
#   curve2coef  : least-squares inverse        -> (in_dim, out_dim, G + k)
#   extend_grid : uniform ghost-node extension -> (in_dim, S + 2 * k_extend)

"""
    _nan_to_num(v)

Match `torch.nan_to_num` semantics: NaN -> 0, ±Inf -> ±floatmax.
This is the degenerate-grid protection applied at every Cox-de Boor level.
"""
@inline function _nan_to_num(v::AbstractArray{T}) where {T<:Real}
    v = ifelse.(isnan.(v), zero(T), v)
    posinf = convert(T, Inf)
    v = ifelse.(v .== posinf, floatmax(T), v)
    v = ifelse.(v .== -posinf, -floatmax(T), v)
    return v
end

"""
    B_batch(x::AbstractMatrix, grid::AbstractMatrix, k::Int)

Evaluate `x` on the B-spline bases of order `k` via the Cox-de Boor recursion.

* `x`    : (batch, in_dim)
* `grid` : (in_dim, S), already extended by `extend_grid`
* returns : (batch, in_dim, S - 1 - k)  =  (batch, in_dim, G + k)
"""
function B_batch(x::AbstractMatrix{T}, grid::AbstractMatrix{T}, k::Int) where {T<:Real}
    x3 = reshape(x, size(x, 1), size(x, 2), 1)          # (batch, in_dim, 1)
    g3 = reshape(grid, 1, size(grid, 1), size(grid, 2)) # (1, in_dim, S)
    return _B_recursive(x3, g3, k)
end

function _B_recursive(x::AbstractArray{T,3}, grid::AbstractArray{T,3}, k::Int) where {T<:Real}
    S = size(grid, 3)
    if k == 0
        # indicator on half-open intervals [grid_i, grid_{i+1})
        val = (x .>= grid[:, :, 1:S-1]) .& (x .< grid[:, :, 2:S])
        return float.(val)  # (batch, in_dim, S-1) of 0.0 / 1.0
    else
        B_km1 = _B_recursive(x, grid, k - 1)            # (batch, in_dim, S-k)
        # Cox-de Boor recurrence, 1-based slicing of PyKAN's:
        #   term1 = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}
        #   term2 = (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}
        num1 = x .- grid[:, :, 1:S-(k+1)]                 # (batch, in_dim, S-k-1)
        den1 = grid[:, :, k+1:S-1] .- grid[:, :, 1:S-(k+1)]
        term1 = (num1 ./ den1) .* B_km1[:, :, 1:S-k-1]

        num2 = grid[:, :, k+2:S] .- x                     # (batch, in_dim, S-k-1)
        den2 = grid[:, :, k+2:S] .- grid[:, :, 2:S-k]
        term2 = (num2 ./ den2) .* B_km1[:, :, 2:S-k]

        return _nan_to_num(term1 .+ term2)                # (batch, in_dim, S-k-1)
    end
end

"""
    coef2curve(x_eval, grid, coef, k)

Convert B-spline coefficients to curve values by summing basis functions
weighted by coefficients (equivalent to PyKAN's `einsum('ijk,jlk->ijl')`).

* `x_eval` : (batch, in_dim)
* `grid`   : (in_dim, S)
* `coef`   : (in_dim, out_dim, G+k)
* returns  : (batch, in_dim, out_dim)
"""
function coef2curve(x_eval::AbstractMatrix{T}, grid::AbstractMatrix{T},
                    coef::AbstractArray{T,3}, k::Int) where {T<:Real}
    b = B_batch(x_eval, grid, k)                          # (batch, in_dim, n)
    batch, in_dim, n = size(b)
    out_dim = size(coef, 2)
    b4 = reshape(b, batch, in_dim, 1, n)
    c4 = reshape(coef, 1, in_dim, out_dim, n)
    return reshape(sum(b4 .* c4; dims=4), batch, in_dim, out_dim)
end

"""
    curve2coef(x_eval, y_eval, grid, k)

Convert B-spline curves back to coefficients by solving a least-squares
problem per (in_dim, out_dim) pair. This is the inverse of `coef2curve` and
is used only by structural operations (grid refinement), never in the
differentiable training forward path.

* `x_eval` : (batch, in_dim)
* `y_eval` : (batch, in_dim, out_dim)
* `grid`   : (in_dim, S)
* returns  : (in_dim, out_dim, G+k)
"""
function curve2coef(x_eval::AbstractMatrix{T}, y_eval::AbstractArray{T,3},
                    grid::AbstractMatrix{T}, k::Int) where {T<:Real}
    batch = size(x_eval, 1)
    in_dim = size(x_eval, 2)
    out_dim = size(y_eval, 3)
    n_coef = size(grid, 2) - k - 1
    mat = B_batch(x_eval, grid, k)                        # (batch, in_dim, n_coef)
    coef = Array{T,3}(undef, in_dim, out_dim, n_coef)
    for o in 1:out_dim, i in 1:in_dim
        coef[i, o, :] = mat[:, i, :] \ y_eval[:, i, o]
    end
    return coef
end

"""
    extend_grid(grid, k_extend)

Uniformly extend a grid by `k_extend` ghost nodes on each end, using the
original grid's step `h = (last - first) / (n - 1)` (computed once).

* `grid` : (in_dim, S)
* returns : (in_dim, S + 2*k_extend)
"""
function extend_grid(grid::AbstractMatrix{T}, k_extend::Int=0) where {T<:Real}
    h = (grid[:, end] .- grid[:, 1]) ./ (size(grid, 2) - 1)  # (in_dim,), fixed
    g = grid
    for _ in 1:k_extend
        g = hcat(g[:, 1] .- h, g)
        g = hcat(g, g[:, end] .+ h)
    end
    return g
end
