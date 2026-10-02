# Integrated Milestone 2 assessment

Edgewise flight adds azimuthal blade-loading asymmetry, retreating-side reverse-flow onset, advancing-tip Mach growth and cyclic-control-induced hub moments. The M2 solver carries the validated M1 axial inflow as its zero-edgewise limit and adds the azimuthal correction/kinematics needed for conversion.

The 3×3 trim matrix contains 6/9 numerically converged states and 4/9 states that also satisfy the adopted physical-feasibility checks. The dense Task-8 corridor contains 5/342 candidate-control feasible grid states; the map is intentionally labeled as a seeded/candidate-control map rather than 342 independent nonlinear trim solves.

The explicit wing/empennage model allows rotor lift sharing to change with nacelle angle. The trim and conversion analyses separate numerical trim from physical feasibility, so a mathematically balanced state can still be rejected for rotor stall, reverse flow, Mach, wing stall, power or control saturation.

The central design trade-off is preserving hover authority while allowing the rotor to unload progressively into the wing during airplane-mode conversion. Lower RPM reduces tip Mach but also changes the available thrust/power characteristics, so the RPM schedule is an explicit transition input rather than an implicit constant.
