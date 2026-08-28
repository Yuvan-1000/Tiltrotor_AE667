# Tiltrotor Milestone 1 -- Code Package

Rotor-performance (BEMT) tool and Mission Planner v1 for the twin-tiltrotor
course project. Python 3, dependencies: `numpy`, `scipy`, `matplotlib`.

## How to run (reproduces every graded plot/table in this package)
```bash
pip install numpy scipy matplotlib          # or: pip install -r requirements.txt
python3 task3_validation.py                 # Task 3  -- validation
python3 task4_design_variable_study.py      # Task 4  -- design-variable study
python3 task5_tiltrotor_design.py           # Task 5  -- rotor/aircraft design table
python3 task6_hover_assessment.py           # Task 6  -- hover performance maps
python3 task7_forward_flight_assessment.py  # Task 7  -- axial/propeller maps
python3 task8_benchmarking.py               # Task 8  -- comparable-rotor benchmarking
python3 task9_mission_planner_demo.py       # Task 9  -- mission planner verification + plots
python3 task10_feasibility_demo.py          # Task 10 -- feasible + infeasible mission demo
```
Each script is self-contained and writes its own figures to `figures/` and
tables to `outputs/` (both created automatically). Run them in the order
above the first time, since Tasks 6-10 import the rotor/aircraft definition
from `task5_tiltrotor_design.py`.

## File / folder purpose (paste this table onto Slide 36)
| File | Purpose | How to run / reproduce |
|---|---|---|
| `bemt_solver.py` | Core BEMT solver: atmosphere, geometry, airfoil models, iterative inflow solve, stall/tip-Mach checks (Tasks 1-2) | imported by every other script; `python3 bemt_solver.py` runs its own built-in demo |
| `data/knight_hefner_table1.csv` | Digitized Knight & Hefner (1937) NACA TN 626 Table I validation data | read by `task3_validation.py` |
| `data/README_validation_data.md` | Provenance / justification for the validation dataset | reference only |
| `task3_validation.py` | Task 3: BEMT vs experiment, error metrics | `python3 task3_validation.py` |
| `task4_design_variable_study.py` | Task 4: solidity/blade-number, taper, twist sweeps | `python3 task4_design_variable_study.py` |
| `task5_tiltrotor_design.py` | Task 5: the ONE aircraft/rotor definition used by every later task | `python3 task5_tiltrotor_design.py` |
| `task6_hover_assessment.py` | Task 6: hover maps, hover ceiling, max hover weight | `python3 task6_hover_assessment.py` |
| `task7_forward_flight_assessment.py` | Task 7: propeller-mode maps, cruise-point search | `python3 task7_forward_flight_assessment.py` |
| `task8_benchmarking.py` | Task 8: nondimensional comparison vs XV-15/V-22 | `python3 task8_benchmarking.py` |
| `mission_planner.py` | Task 9: core mission-planner library (segments, feasibility checks) | imported by task9/task10 scripts |
| `task9_mission_planner_demo.py` | Task 9 demo: implementation verification + Sec 7.2-7.4 plots | `python3 task9_mission_planner_demo.py` |
| `task10_feasibility_demo.py` | Task 10: one feasible + one deliberately infeasible mission | `python3 task10_feasibility_demo.py` |
| `dev_notes_bemt_quirks.md` | Numerical pitfalls found during development (twist/collective sign issues, bracket edge cases) | reference only |
| `TECHNICAL_ENTITIES.md` | Slide-by-slide test-case values for the report template | reference only |
| `REFERENCES.md` | Consolidated reference list for Sec 8.2 | reference only |
| `figures/` | All generated plots (created on first run) | -- |
| `outputs/` | All generated tables/logs/notes (created on first run) | -- |

## Convergence / nonphysical-input checks already in place
- `bemt_solver.py`'s bisection solve reports whether it converged and warns
  (with station indices) if it could not bracket a root at any station.
- Every root-find for a target thrust (`_collective_for_thrust` in
  `mission_planner.py`, `collective_for_thrust` in `task6`) explicitly
  checks the sign of the residual at both ends of the allowed collective
  range before calling `brentq`, and returns `None`/raises rather than
  silently extrapolating if the target is unreachable.
- `MissionPlanner` runs a fixed, ordered set of feasibility checks (fuel,
  power margin, stall fraction, tip Mach, collective bounds) every time
  step and stops immediately with a segment/time/reason on the first
  violation (Task 10).

## Known limitations (see `dev_notes_bemt_quirks.md` and the docstrings in
`task5_tiltrotor_design.py` / `task7_forward_flight_assessment.py` /
`outputs/task9_notes_cruise_trim.md` for the full, honest list)
- The designed rotor's cruise speed (~27.5 m/s trimmed) falls well short of
  the 110 m/s mission requirement -- flagged explicitly as a Milestone-2
  action item, not silently hidden.
- The airfoil model for the designed rotor is a documented linear-Cl
  assumption, not a real digitized polar (none was provided in the
  handout beyond the 1930s NACA 0015 validation airfoil).
