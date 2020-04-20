# Rules for automatic differentiation

## rules for fwd/rev QDHT
function _mul_back(ΔY, Q, A)
    ∂A = similar(ΔY)
    dot!(∂A, Q.T', ΔY, dim = Q.dim)
    ∂A .*= Q.scaleRK
    return ∂A
end

function _ldiv_back(ΔY, Q, A)
    ∂A = similar(ΔY)
    dot!(∂A, Q.T', ΔY, dim = Q.dim)
    ∂A ./= Q.scaleRK
    return ∂A
end

function ChainRulesCore.rrule(::typeof(*), Q::AbstractQDHT, A)
    Y = Q * A
    function mul_pullback(ΔY)
        return NO_FIELDS, DoesNotExist(), @thunk _mul_back(ΔY, Q, A)
    end
    return Y, mul_pullback
end

function ChainRulesCore.rrule(::typeof(\), Q::AbstractQDHT, A)
    Y = Q \ A
    function ldiv_pullback(ΔY)
        return NO_FIELDS, DoesNotExist(), @thunk _ldiv_back(ΔY, Q, A)
    end
    return Y, ldiv_pullback
end

## rules for dimdot, makes integrateR/K autodiffable
function _dimdot_back(ΔΩ, v, A; dim = 1, dims = Tuple(collect(size(A))))
    T = Base.promote_eltype(v, ΔΩ)
    ∂A = similar(A, T, dims)
    idxlo = CartesianIndices(dims[1:(dim - 1)])
    idxhi = CartesianIndices(dims[(dim + 1):end])
    for lo in idxlo, hi in idxhi
        ∂A[lo, :, hi] .= ΔΩ[lo, 1, hi] .* v
    end
    return ∂A
end
_dimdot_back(ΔΩ, v, A::AbstractVector; dim = 1) = ΔΩ .* v

function ChainRulesCore.rrule(::typeof(dimdot), v, A; dim = 1)
    Ω = dimdot(v, A; dim = dim)
    function dimdot_pullback(ΔΩ)
        return NO_FIELDS, DoesNotExist(), @thunk _dimdot_back(ΔΩ, v, A; dim = dim)
    end
    return Ω, dimdot_pullback
end
