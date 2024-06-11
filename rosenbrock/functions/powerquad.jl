"""

I need an explicitly quadratic function (to verify 2nd-order methods converge immediately)
    which sort of looks at first glance as complicated as a Rosenbrock function.

"""
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