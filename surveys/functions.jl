"""
Submodules each implement three functions.
    each of which must share the same optional keyword arguments,
    and return a callable accepting a vector of floats, of any length L,
    and return a type given below:
- `lossfunction`: Float
- `gradient`: Vector{Float}(L)
- `hessian`: Matrix{Float}(L,L)

The loss function for any given number of parameters
    should have a global minimum of 0.0,
    located at [1.0, 1.0, ...].
The loss function evaluated at [1.0, 1.0, ...] should have a value of 1.0.

The outputs of `gradient` and `hessian` should be related
    to that of `lossfunction` in the obvious way. Obviously.

Other than that, anything goes. :)

"""
module TestFunctions
    export noisy, analytic
    export PowerQuad, Rosenbrock

    function noisy(rng, fn, σ)
        return x -> fn(x) + σ*randn(rng)
    end

    function analytic(Submodule; kwargs...)
        return (
            f = Submodule.lossfunction(kwargs...),
            g = Submodule.gradient(kwargs...),
            H = Submodule.Hessian(kwargs...),
        )
    end

    """ The prototypical "hard" function. """
    module Rosenbrock
        function lossfunction(a=1.0, b=100.0)
            return x -> (
                N = length(x);
                total = 0.0;
                for i in 1:N-1;
                    total += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2;
                end;
                total / (N-1)
            )
        end

        function gradient(a=1.0, b=100.0)
            return x -> (
                N = length(x);
                g = zeros(N);
                for k in eachindex(g);
                    k ≠ 1 && ( g[k] += 2b * (x[k] - x[k-1]^2) );
                    k ≠ N && ( g[k] -= 2 * (a - x[k]) + 4b * (x[k+1] - x[k]^2) * x[k] );
                end;
                g ./ (N-1)
            )
        end

        function hessian(a=1.0, b=100.0)
            return x -> (
                N = length(x);
                H = zeros(N,N);
                for k in axes(H,1);
                    # MAIN DIAGONAL
                    k ≠ 1 && ( H[k,k] += 2b )
                    k ≠ N && ( H[k,k] += 2 + 8b * x[k]^2 - 4b * (x[k+1] - x[k]^2) )
                    # OFF DIAGONAL
                    k ≠ 1 && ( H[k,k-1] -= 4b * x[k-1] )
                    k ≠ N && ( H[k,k+1] -= 4b * x[k] )
                end;
                H ./ (N-1)
            )
        end
    end

    """ Explicitly quadratic function, such that Newton converges in one step. """
    module PowerQuad
        powerlaw(c, A, γ, k) = c / (A + k + 1)^γ
        partition(c, A, γ, N) = (
            total = 0.0;
            for i in 1:N;
            for j in 1:N;
                total += powerlaw(c,A,γ,abs(i-j));
            end; end;
            total
        )

        function lossfunction(a=1.0, c=100.0, A=0.0, γ=1.0)
            return x -> (
                N = length(x);
                total = 0.0;
                for i in 1:N;
                for j in 1:N;
                    total += powerlaw(c,A,γ,abs(i-j)) * (a - x[i]) * (a - x[j]);
                end; end;
                total / partition(c,A,γ,N)
            )
        end

        function gradient(a=1.0, c=100.0, A=0.0, γ=1.0)
            return x -> (
                N = length(x);
                g = zeros(N);
                for k in 1:N;
                for i in 1:N;
                    g[k] -= 2 * powerlaw(c,A,γ,abs(k-i)) * (a - x[i])
                end; end;
                g ./ partition(c,A,γ,N)
            )
        end

        function hessian(a=1.0, c=100.0, A=0.0, γ=1.0)
            return x -> (
                N = length(x);
                H = zeros(N,N);
                for k in 1:N;
                for l in 1:N;
                    H[k,l] += 2 * powerlaw(c,A,γ,abs(k-l))
                end; end;
                H ./ partition(c,A,γ,N)
            )
        end
    end
end