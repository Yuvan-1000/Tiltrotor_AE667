# Task 10 design-refinement rationale

The M2 design retains the M1 rotor geometry so edgewise-flight physics can be isolated without mixing geometry changes and model changes. The important refinements are the explicit airframe lift/drag model, continuous nacelle and RPM schedules, full pilot-control limits, rotor-to-body load transformations, and a single mass/fuel source of truth.

The low-order nature of the airframe and rotor model remains a known limitation; wake/wing interference, realistic post-stall polars, flapping dynamics and a higher-fidelity propulsion map are reserved for Milestone 3.
