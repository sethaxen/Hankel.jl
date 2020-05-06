# Rules for automatic differentiation

## rules for fwd/rev transform
function ChainRulesCore.rrule(::typeof(*), Q::QDHT, A)
    Y = Q * A
    function mul_pullback(ΔY)
        ∂Q = DoesNotExist()
        ∂A = @thunk _mul_back(ΔY, Q, A, Q.scaleRK)
        return NO_FIELDS, ∂Q, ∂A
    end
    return Y, mul_pullback
end

function ChainRulesCore.rrule(::typeof(\), Q::QDHT, A)
    Y = Q \ A
    function ldiv_pullback(ΔY)
        ∂Q = DoesNotExist()
        ∂A = @thunk _mul_back(ΔY, Q, A, inv(Q.scaleRK))
        return NO_FIELDS, ∂Q, ∂A
    end
    return Y, ldiv_pullback
end

function _mul_back(ΔY, Q, A, s)
    ∂A = similar(ΔY)
    dot!(∂A, Q.T', ΔY, dim = Q.dim)
    ∂A .*= s
    return ∂A
end

## rules for integrateR/integrateK
function ChainRulesCore.rrule(::typeof(integrateR), A, Q::QDHT; dim = 1)
    function integrateR_pullback(ΔΩ)
        ∂A = @thunk _integrateRK_back(ΔΩ, A, Q.scaleR; dim = dim)
        return NO_FIELDS, ∂A, DoesNotExist()
    end
    return integrateR(A, Q; dim = dim), integrateR_pullback
end

function ChainRulesCore.rrule(::typeof(integrateK), A, Q::QDHT; dim = 1)
    function integrateK_pullback(ΔΩ)
        ∂A = @thunk _integrateRK_back(ΔΩ, A, Q.scaleK; dim = dim)
        return NO_FIELDS, ∂A, DoesNotExist()
    end
    return integrateK(A, Q; dim = dim), integrateK_pullback
end

_integrateRK_back(ΔΩ, A::AbstractVector, scale; dim = 1) = ΔΩ .* conj.(scale)
function _integrateRK_back(ΔΩ, A::AbstractMatrix, scale; dim = 1)
    return dim == 1 ? conj.(scale) .* ΔΩ : ΔΩ * scale'
end
function _integrateRK_back(ΔΩ, A, scale; dim = 1)
    dims = Tuple(collect(size(A)))
    n = ndims(A)
    scalearray = reshape(
        scale,
        ntuple(_ -> 1, dim - 1)...,
        dims[dim],
        ntuple(_ -> 1, n - dim)...,
    )
    ∂A = ΔΩ .* conj.(scalearray)
    return ∂A
end
