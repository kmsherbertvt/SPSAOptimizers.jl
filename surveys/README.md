The goal here is to just try a whole lot of trajectories for any given (fn, x0),
    given a total budget M and noise sigma,
    and to try to visualize them all in a consistent way.


order, n, (K constrained by these three), A/K, a0, c0, alpha, gamma, kpL, tolerance, trust (1.0 or Inf)







Substitute different fn and x0.

fn can be rosenbrock or powerquad or noisy versions of either with various noise.

Also try to engineer something with a geometry similar to ctrl-VQE, somehow. Maybe expectation value of an n-level evolving under an anharmonic device with a sequence of impulse drives informed by the parameter vector. Engineer it so that 0,0.. is the "Hartree-Fock" state and 1,1.. prepares the ground-state of the observable. This doesn't seem easy...

x0 can be 0, or perturbation around 1, or totally crazy.


