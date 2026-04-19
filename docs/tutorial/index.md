# Tutorial

A guided introduction to inverse kinematics and to the algorithm IKFast uses internally. The tutorial assumes basic linear algebra; everything else is built up from scratch.

!!! note "Under construction"
    Tutorial chapters land as the underlying implementation lands. Issue [#16](https://github.com/siddhss5/ikfastpy/issues/16) tracks chapters 1–4 (foundational); issue [#17](https://github.com/siddhss5/ikfastpy/issues/17) tracks chapters 5–7 (deep dive into IKFast and EAIK).

## Planned chapters

1. **The IK Problem** — forward vs inverse kinematics; why closed-form solutions matter for planning.
2. **Kinematics Primer** — rigid transforms, Denavit–Hartenberg parameters, Product-of-Exponentials.
3. **Numerical IK** — Jacobian inverse, damped least squares, when these methods fail.
4. **Classical Closed-Form** — Pieper's three-intersecting-axes condition, Paden–Kahan subproblems.
5. **Algebraic IK and How IKFast Works** — resultants, elimination, the symbolic decomposition strategy.
6. **How EAIK Works** — kinematic family detection, comparison to IKFast.
7. **Practical Guide** — picking a solver, troubleshooting, performance tuning.
