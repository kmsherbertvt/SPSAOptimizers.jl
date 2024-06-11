"""

Wikipedia mentions two ways of generalizing to more than two dimensions;
    I've implemented the second way.

"""
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