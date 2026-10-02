# Task 8 / Sec. 7 — corridor summary

Resolved grid: 19 nacelle angles × 18 airspeeds = 342 states. All cases use h=500 m, gross mass 3000 kg, ISA and the same aircraft/rotor configuration.

- feasible: 5 states
- excessive trim residual: 126 states
- control saturation: 0 states
- wing stall: 0 states
- power limitation: 1 states
- advancing-tip Mach: 0 states
- reverse flow: 28 states
- rotor stall: 182 states

The map is deliberately labeled as a candidate-control feasibility map: the controls are seeded from the required 3×3 trim matrix and each grid point is checked with the aircraft model. A point labeled feasible is therefore a model-feasible interpolated control state, not an independently converged trim solution.
