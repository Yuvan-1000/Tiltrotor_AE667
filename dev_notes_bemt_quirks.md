# Development notes: things we hit while building this code package

These are honest notes from building this milestone's code. They are not
required reading to run the scripts, but they will save your team time if
you change any geometry/twist/collective/RPM numbers, and they are good
material for "model limitations" style discussion if you want to cite them.

## 1. Large twist + small collective range can silently flip the sign of
   thrust

`theta(r) = collective + twist(r)`. If your twist distribution has a large
negative washout (e.g. -30 to -40 deg, as real full-scale tiltrotor blades
use) but your collective range is a "helicopter-like" 2-20 deg, the blade
tip's local pitch becomes strongly negative (e.g. 10 deg collective - 32 deg
washout = -22 deg tip pitch). Because `dT` scales with `r^2`-ish weighting,
a large negative-lift region near the tip can dominate the (smaller, less
r-weighted) positive-lift region inboard, and the *integrated* thrust comes
out **negative** even though nothing "crashed" -- the solver converges fine
and just reports a small/negative/nonsensical T. `bemt_solver.py` does not
and cannot know this is unintended; it will not warn you (there is no
physical law against a negative-thrust operating point -- windmilling rotors
do this on purpose).

**Takeaway / self-check your team should run on ANY new geometry:** after
defining a `RotorGeometry`, print `res["alpha_deg"]` (or at least
`res["alpha_deg"][-1]`, the tip value) across your intended collective range
and confirm it stays in a physically sensible band (very roughly -5 to
+12 deg for an efficient rotor) before trusting T, Q, P, or FM from that
run. We added exactly this check when sizing `task5_tiltrotor_design.py`
and that is *why* the shipped design uses -12 deg (not a full -35/-40 deg)
of washout -- see that file's module docstring.

## 2. Bracket warnings near the tip at high collective + high twist
   combinations

At some of the more extreme collective/twist combinations explored in
`task4_design_variable_study.py`'s twist trim study, you will see
`bemt_solver.py` print warnings like:

```
BEMT: could not bracket a sign change in the induced-velocity residual at N
station(s) (r/R = [0.9xx, ...]); induced velocity set to 0 there.
```

This means the per-station bisection window (`window_lo`/`window_hi` in
`BEMTSolver.solve`) didn't contain the true root for a handful of outboard
stations at that specific operating point. The solver falls back to
`vi=0` at just those stations, which has only a small effect on the
*integrated* CT/CQ (few stations, near the edge, small `dr` weight each),
but it does mean the very-tip-most points in a plotted radial distribution
at those specific operating points are not fully trustworthy. If your team
pushes into more extreme operating points than we did here, consider
widening the window in `BEMTSolver.solve` (search for `window_lo =` /
`window_hi =`) or refining `n_grid`.

## 3. Sign convention reminder

`FlightCondition.V_climb` and `FlightCondition.V_axial` are both defined
positive along the rotor axis in the "thrust" direction (i.e. positive
`V_climb` = climbing away from the disk in helicopter mode; positive
`V_axial` = axial flight with freestream flowing *into* the rotor disk, the
normal "propeller advancing into the wind" case used in Task 7). Do not mix
sign conventions between the two if you ever combine them.
