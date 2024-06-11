# SPSAOptimizers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kmsherbertvt.github.io/SPSAOptimizers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kmsherbertvt.github.io/SPSAOptimizers.jl/dev/)
[![Build Status](https://github.com/kmsherbertvt/SPSAOptimizers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kmsherbertvt/SPSAOptimizers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kmsherbertvt/SPSAOptimizers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kmsherbertvt/SPSAOptimizers.jl)




Simultaneous Perturbation Stochastic Approximation (SPSA) is a scheme to optimize a multivariate function using a stochastic gradient descent that minimizes cost function evalutions and provides some robustness against noise.

## Motivation

In the context of variational quantum algorithms, one wishes to optimize a cost-function which is the result of stochastic measurements on a quantum computer, after applying some quantum circuit.
Applying a quantum circuit is, as of yet, quite expensive, so one wishes to minimize cost function evaluations.
And stochastic measurements inevitably incur some sampling noise (*even* if the quantum computer itself is "perfect").
Thus, the SPSA algorithm is particularly attractive.
Indeed, IBM's `qiskit` framework provides an interface into several competing optimizers, but the online documentation includes this note:

> SPSA can be used in the presence of noise, and it is therefore indicated in situations involving measurement uncertainty on a quantum computation when finding a minimum. If you are executing a variational algorithm using a Quantum ASseMbly Language (QASM) simulator or a real device, SPSA would be the most recommended choice among the optimizers provided here.

This package is written with the following goals in mind:
- A pure Julia implementation of SPSA, suitable for high-performance quantum simulation software written in pure Julia.
- The optimization state (including that of the pseudo-random number generator used for stochastic gradient desent) can be saved and resumed.
- Potential integration into an existing pure-Julia optimization library such as `Optim.jl`.



## References
1. [J. C. Spall (1998). An Overview of the Simultaneous Perturbation Method for Efficient Optimization, Johns Hopkins APL Technical Digest, 19(4), 482–492.](https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF)
2. [J. C. Spall (1998). Implementation of the Simultaneous Perturbation Algorithm for Stochastic Optimization, Johns Hopkins APL Technical Digest, 34(3), 817-823.](https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF)
2. [J. C. Spall (1997). A One-Measurement Form of Simultaneous Perturbation Stochastic Approximation, Automatica 33, 109–112.](https://doi.org/10.1016/S0005-1098(96)00149-5)
3. [J. C. Spall (1997). Accelerated second-order stochastic optimization using only function measurements, Proceedings of the 36th IEEE Conference on Decision and Control, 1417-1424 vol.2.](https://ieeexplore.ieee.org/document/657661)

This implementation is largely inspired by the Python implemention in `qiskit`.
1. [Online Documentation](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.optimizers.SPSA.html)
2. [GitHub code](https://github.com/qiskit-community/qiskit-algorithms/blob/main/qiskit_algorithms/optimizers/spsa.py)
