"""
    squeeze(A; dims)

Wrapper around `dropdims` to handle both numbers (return just the number) and arrays
(return `dropdims(A; dims)`).
"""
squeeze(A::Number; dims) = A
squeeze(A::AbstractArray; dims) = dropdims(A, dims=dims)

function _dot(M, V; dim=1)
    size(V, dim) == size(M, 1) || throw(DomainError(
        "Size of V along dim must be same as size of M"))
    n = ndims(V)
    dim ≤ n || throw(DomainError("Cannot multiply along dimension $dim of $n-D array"))
    dimsV = ntuple(identity, n)
    dimsM = (0, dim)
    dimsY = (1:(dim - 1)..., 0, ((dim + 1):n)...)
    code = EinCode((dimsV, dimsM), dimsY)
    return einsum(code, (V, M))
end
function _dot(M, V::AbstractVector; dim=1)
    size(V, dim) == size(M, 1) || throw(DomainError(
        "Size of V along dim must be same as size of M"))
    dim == 1 || throw(DomainError("Cannot multiply along dimension $dim of 1-D array"))
    return M * V
end
function _dot(M, V, s; dim=1)
    return if length(M) < length(V)
        _dot(M .* s, V; dim=dim)
    else
        _dot(M, V .* s; dim=dim)
    end
end

"""
    _dot!(out, M, V; dim=1)

Matrix-vector multiplication along specific dimension of array `V`, storing result in `out`.

This is equivalent to iterating over all dimensions of `V` other than `dim` and applying the
matrix-vector multiplication `M * v[..., :, ...]`, but works by reshaping the array if
necessary to take advantage of faster matrix-matrix multiplication. If `dim==1`, `dot!` is
fastest and allocation-free.
"""
function _dot!(out, M, V::AbstractVector; dim=1)
    size(V, dim) == size(M, 1) || throw(DomainError(
        "Size of V along dim must be same as size of M"))
    size(out) == size(V) || throw(DomainError("Input and output arrays must have same size"))
    dim == 1 || throw(DomainError("Cannot multiply along dimension $dim of 1-D array"))
    return mul!(out, M, V)
end
_dot!(out, M, V::AbstractVector, s; dim=1) = _dot!(out, M, V .* s; dim=dim)
_dot!(out, M, V; dim=1) = copyto!(out, _dot(M, V; dim=dim))
_dot!(out, M, V, s; dim=1) = copyto!(out, _dot(M, V, s; dim=dim))

"""
    dimdot(v, A; dim=1)

Calculate the dot product between vector `v` and one dimension of array `A`, iterating over
all other dimensions.
"""
function dimdot(v, A; dim=1)
    n = ndims(A)
    dimsv = (dim, 0)
    dimsA = ntuple(identity, n)
    dimsY = (1:(dim - 1)..., 0, ((dim + 1):n)...)
    code = EinCode((dimsv, dimsA), dimsY)
    return einsum(code, (reshape(v, length(v), 1), A))
end
dimdot(v, A::AbstractVector; dim=1) = dot(v, A)

"""
    sphbesselj_scale(n)

Return the normalization factor for the (hyper)spherical Bessel function
([`sphbesselj`](@ref)) with spherical dimension ``n``, given as
``c_n = \\sqrt{\\frac{π}{2}}`` for even ``n`` and 1 otherwise.

After:

[1] J. S. Avery, J. E. Avery. Hyperspherical Harmonics And Their Physical Applications.
    Singapore: World Scientific Publishing Company, 2017.
"""
sphbesselj_scale(n) = isodd(n) ? 1.0 : √(π / 2)

"""
    sphbesselj(p, n, x)

(Hyper)spherical Bessel function of order ``p`` and spherical dimension ``n``.
The hyperspherical Bessel function generalizes the cylindrical and spherical Bessel
functions to the ``n``-sphere (embedded in ``ℝ^{n+1}``). It is given as

```math
j_p^{n}(x) = c_n x^{-(n-1)/2} J_{p + (n-1)/2}(x),
```

where ``c_n`` is a normalization factor defined by [`sphbesselj_scale`](@ref). Note that
``n`` is not an exponent here.

It has as its special cases:
- Cylindrical Bessel function (``n=1``): ``j_p^{1}(x) = J_p(x)``
- Spherical Bessel function (``n=2``): ``j_p^{2}(x) = j_p(x)``

After:

[1] J. S. Avery, J. E. Avery. Hyperspherical Harmonics And Their Physical Applications.
    Singapore: World Scientific Publishing Company, 2017.
"""
function sphbesselj(p, n, x)
    n == 1 && return besselj(p, x)
    α = (n - 1) / 2
    Jppa = besselj(p + α, x)
    cn = sphbesselj_scale(n)
    if abs(x) ≤ sqrt(eps(real(zero(Jppa))))
        if p == 0
            J0pa = cn / gamma(α + 1) / 2^α
            return convert(typeof(Jppa), J0pa)
        else
            return zero(Jppa)
        end
    end
    jp = cn * Jppa / x^α
    return convert(typeof(Jppa), jp)
end

"""
    sphbesselj_zero(p, n, m)

Get the ``m``th zero of the (hyper)spherical Bessel function of order ``p`` and spherical
dimension ``n``.

See [`sphbesselj`](@ref).
"""
sphbesselj_zero(p, n, m) = besselj_zero(p + (n - 1) / 2, m)
