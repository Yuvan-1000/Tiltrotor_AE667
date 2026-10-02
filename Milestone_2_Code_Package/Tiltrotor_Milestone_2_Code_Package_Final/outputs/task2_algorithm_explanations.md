# Task 2 algorithm explanation

## 2.1 Edgewise estimator
The estimator starts with user inputs and the ISA atmosphere, then resolves the aircraft forward speed into through-disk and in-plane components from the nacelle angle. The validated M1 axial BEMT provides a per-station induced-inflow baseline. The M2 handout relation is applied using lambda_G = lambda_bar_i + V_axial/(Omega R), where lambda_G is total inflow ratio and the corrected quantity is the induced component. Each (r,psi) element then uses azimuth-dependent tangential velocity, non-uniform induced velocity and the course cyclic-pitch law. Sectional lift/drag is resolved into thrust and tangential force, reverse-flow/stall/Mach checks are applied, and the loads are integrated over a revolution before being transformed to body axes.

## 2.2 Trim solver
The steady solver treats the flight condition as fixed and adjusts seven bounded pilot/attitude variables. Six residuals are evaluated directly from the aircraft model. A finite-difference Jacobian gives a Gauss-Newton step, with clipping and backtracking to keep the iteration inside the adopted bounds. Numerical success and physical feasibility are reported separately.

## 2.3 Mission Planner v2
The planner is a generic segment/time-step engine. All mission schedules come from JSON input rather than hard-coded mission logic. At each time step it calls the same aircraft model, burns fuel from shaft-power demand, reduces gross mass, and checks the adopted aerodynamic, power and control constraints. Both outbound and inbound demonstrations use the same aircraft configuration.
