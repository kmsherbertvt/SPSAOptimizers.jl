This sub-package was mostly a sandbox to debug the `SPSAOptimizers.jl` package,
    and to try and build intuition for how to use SPSA.

It's all a bit of a mess. Consider the scripts starting with `contrast_` to be experimental,
    and those with `contrast.` to be more finalized.

The most important part was comparing results directly to trajectories generated with `PyCall.jl` and qiskit's implementation.
This was done in the `contrast.qiskit_interface.jl` script.
I couldn't sync the random number generators, so nothing is identical, but everything seems plausibly close. ^_^

After that was finished, I started trying to assess how sensitive SPSA is to its various hyperparameters.
Alas, the answer to that is VERY.
More importantly, the ideal hyperparameters change with the problem AND with the total function allocation budget.
Probably the most important lesson I've learned from these tests is that Stochastic Approximation algorithms
    are not meant to be continued indefinitely;
    they should be run with a fixed budget in mind from the beginning
Of course pausing and resuming within that budget is a great thing,
    design-wise, which is the main motivation for this package.
But, given the mercurial nature of each quantum computer,
    probably it makes sense to set the budget according to what one can do within a single session.
