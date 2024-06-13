This is sorta a sandbox, primarily to compare this implementation of SPSA directly against that in qiskit,
    but also to try and get a feel for what it takes to use SPSA, um, usefully.

In `contrast.qiskit_interface`, we compare the two implementations pretty directly, and they match to my satisfaction,
    and they perform more or less as well as I might expect on an explicitly quadratic function.
So, that's done.


Now we can use pure Julia and really start playing with things.
- Do a scan over a0 to see what the best 1SPSA is, and to convince yourself whether 2SPSA really is best with 1.0.
- Resampling a whole lot in the early stages. I have my head irrationally fixed on a L-L scheme.
- Using the exact initial Hessian (as though sampled heavily) with a large weight (e.g. L*L again).
- Running 1SPSA as best you can, to get as best you can into a concave region, then 2SPSA as above.
- Same as above but measure the Hessian as you go for the first trajectory, and use THAT to start 2SPSA. (Practically, this is running 2SPSA but with warmup=maxiter.)

