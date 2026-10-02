# Task 7 / Sec. 6.4 — physical interpretation of trim trends

As γ decreases from helicopter toward airplane mode, the rotor thrust direction rotates progressively into the body-forward direction. The wing therefore carries a larger fraction of the required vertical force at higher airspeed, while rotor collective and pitch attitude shift to satisfy the longitudinal and vertical force balance. Longitudinal cyclic is available in the model to create a first-harmonic hub pitching response; tail elevator supplies the remaining aircraft pitching-moment trim. The exact values depend on the low-order wing/tail model and the rigid-disk rotor approximation.

A numerically converged state is not automatically a feasible state: rotor stall, reverse flow, Mach, wing-stall, power and control bounds are checked after convergence. This distinction is used in the conversion map and mission planner.
