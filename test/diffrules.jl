# adapted from ChainRulesTestUtils.rrule_test
function rrule_test(
    ::Val{:ChainRules},
    f,
    ȳ,
    xx̄s::Tuple{Any,Any}...;
    rtol = 1e-9,
    atol = 1e-9,
    fkwargs = NamedTuple(),
    fdm = central_fdm(5, 1),
    kwargs...,
)
    # Check correctness of evaluation.
    xs, x̄s = collect(zip(xx̄s...))
    y, pullback = rrule(f, xs...; fkwargs...)
    @test f(xs...; fkwargs...) == y

    @assert !(isa(ȳ, Thunk))
    ∂s = pullback(ȳ)
    ∂self = ∂s[1]
    x̄s_ad = ∂s[2:end]
    @test ∂self === NO_FIELDS

    # Correctness testing via finite differencing.
    return for (i, dx_ad) in enumerate(x̄s_ad)
        if x̄s[i] === nothing
            @test dx_ad isa DoesNotExist
        else
            x̄_fd = j′vp(
                fdm,
                x -> f(xs[1:(i - 1)]..., x, xs[(i + 1):end]...; fkwargs...),
                ȳ,
                xs[i],
            )[1]
            x̄_ad = unthunk(dx_ad)
            @test isapprox(x̄_ad, x̄_fd; rtol = rtol, atol = atol, kwargs...)
        end
    end
end

# adapted from ChainRulesTestUtils.rrule_test
function rrule_test(
    ::Val{:Zygote},
    f,
    ȳ,
    xx̄s::Tuple{Any,Any}...;
    rtol = 1e-9,
    atol = 1e-9,
    fkwargs = NamedTuple(),
    fdm = central_fdm(5, 1),
    kwargs...,
)
    # Check correctness of evaluation.
    xs, x̄s = collect(zip(xx̄s...))
    y, pullback = Zygote.pullback((xs...) -> f(xs...; fkwargs...), xs...)
    @test f(xs...; fkwargs...) == y

    x̄s_ad = pullback(ȳ)

    # Correctness testing via finite differencing.
    return for (i, x̄_ad) in enumerate(x̄s_ad)
        if x̄s[i] === nothing
            @test x̄_ad === nothing
        else
            x̄_fd = j′vp(
                fdm,
                x -> f(xs[1:(i - 1)]..., x, xs[(i + 1):end]...; fkwargs...),
                ȳ,
                xs[i],
            )[1]
            @test isapprox(x̄_ad, x̄_fd; rtol = rtol, atol = atol, kwargs...)
        end
    end
end

@testset "autodiff rules" begin
    @testset "$method" for method in (:ChainRules, :Zygote)
        mval = Val(method)
        @testset "*(::AbstractQDHT, ::Array)" begin
            rng = MersenneTwister(86)
            @testset "Vector" begin
                N = 64
                q = Hankel.QDSHT(1, 2, 10, N)
                fr, f̄r, f̄k = randn(rng, N), randn(rng, N), randn(rng, N)
                rrule_test(mval, *, f̄k, (q, nothing), (fr, f̄r))
            end

            @testset "Matrix" begin
                N = 64
                M = 5
                @testset for dim in (1, 2)
                    q = Hankel.QDSHT(1, 2, 10, N; dim = dim)
                    s = dim == 1 ? (N, M) : (M, N)
                    fr, f̄r, f̄k = randn(rng, s), randn(rng, s), randn(rng, s)
                    rrule_test(mval, *, f̄k, (q, nothing), (fr, f̄r))
                end
            end

            @testset "Array{T,3}" begin
                N = 64
                M = 5
                K = 10
                q = Hankel.QDSHT(1, 2, 10, N; dim = 2)
                s = (M, N, K)
                fr, f̄r, f̄k = randn(rng, s), randn(rng, s), randn(rng, s)
                rrule_test(mval, *, f̄k, (q, nothing), (fr, f̄r))
            end
        end

        @testset "\\(::AbstractQDHT, ::Array)" begin
            rng = MersenneTwister(62)
            @testset "Vector" begin
                N = 64
                q = Hankel.QDSHT(1, 2, 10, N)
                fr, f̄r, f̄k = randn(rng, N), randn(rng, N), randn(rng, N)
                rrule_test(mval, \, f̄k, (q, nothing), (fr, f̄r))
            end

            @testset "Matrix" begin
                N = 64
                M = 5
                @testset for dim in (1, 2)
                    q = Hankel.QDSHT(1, 2, 10, N; dim = dim)
                    s = dim == 1 ? (N, M) : (M, N)
                    fr, f̄r, f̄k = randn(rng, s), randn(rng, s), randn(rng, s)
                    rrule_test(mval, \, f̄k, (q, nothing), (fr, f̄r))
                end
            end

            @testset "Array{T,3}" begin
                N = 64
                M = 5
                K = 10
                q = Hankel.QDSHT(1, 2, 10, N; dim = 2)
                s = (M, N, K)
                fr, f̄r, f̄k = randn(rng, s), randn(rng, s), randn(rng, s)
                rrule_test(mval, \, f̄k, (q, nothing), (fr, f̄r))
            end
        end

        @testset "dimdot(::AbstractMatrix, ::Array)" begin
            rng = MersenneTwister(27)
            @testset "Vector" begin
                N = 64
                v, A, Ā, ȳ = randn(rng, N), randn(rng, N), randn(rng, N), randn(rng)
                rrule_test(mval, Hankel.dimdot, ȳ, (v, nothing), (A, Ā))
            end

            @testset "Matrix" begin
                N = 64
                M = 5
                @testset for dim in (1, 2)
                    s = dim == 1 ? (N, M) : (M, N)
                    sy = dim == 1 ? (1, M) : (M, 1)
                    v, A, Ā, ȳ =
                        randn(rng, N), randn(rng, s), randn(rng, s), randn(rng, sy)
                    rrule_test(
                        mval,
                        Hankel.dimdot,
                        ȳ,
                        (v, nothing),
                        (A, Ā);
                        fkwargs = (; dim = dim),
                    )
                end
            end

            @testset "Array{T,3}" begin
                N = 64
                M = 5
                K = 10
                s = (M, N, K)
                sy = (M, 1, K)
                v, A, Ā, ȳ = randn(rng, N), randn(rng, s), randn(rng, s), randn(rng, sy)
                rrule_test(
                    mval,
                    Hankel.dimdot,
                    ȳ,
                    (v, nothing),
                    (A, Ā);
                    fkwargs = (; dim = 2),
                )
            end
        end
    end
end
