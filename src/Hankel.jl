module Hankel
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj, gamma
import LinearAlgebra: mul!, ldiv!, dot
import Base: *, \
using ZygoteRules: @adjoint

export QDHT, QDSHT, integrateK, integrateR, onaxis, symmetric, Rsymmetric

const J₀₀ = besselj(0, 0)

include("utils.jl")
include("plan.jl")
include("qdht.jl")
include("qdsht.jl")
include("diffrules.jl")

end
