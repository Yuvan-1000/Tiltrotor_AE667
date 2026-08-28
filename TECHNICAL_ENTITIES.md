# Technical entities / test cases for Milestone_1_Student_Template.pptx

This maps every slide in the template that has checkable numeric content to
the exact value(s) this code package produces. Use these as **workflow test
cases**: if you re-run the same script with the same config and get numbers
in this neighborhood, your pipeline is wired correctly end-to-end. If your
team changes the design (different R, B, twist, GW, ...), your numbers
*should* differ from these -- that's expected and correct. Slides that need
an original diagram/schematic/prose are marked "NOT AUTO-GENERATED" and are
left for your team, as they should be.

Run order to regenerate everything from scratch:
```
python3 task3_validation.py
python3 task4_design_variable_study.py
python3 task5_tiltrotor_design.py
python3 task6_hover_assessment.py
python3 task7_forward_flight_assessment.py
python3 task8_benchmarking.py
python3 task9_mission_planner_demo.py
python3 task10_feasibility_demo.py
```

---
## Slide 4 -- 1.1 Physics Assumptions and Data
Source: `bemt_solver.py` docstring (assumptions A1-A8), `task5_tiltrotor_design.py` docstring.
- Induced-flow model: local (per-station) combined blade-element/momentum theory, iterative bisection solve for vi(r)
- Tip loss: Prandtl tip-loss factor F, **enabled** (`use_tip_loss=True`)
- Root loss: **disabled** by default (`use_root_loss=False`) -- root cutout handled by integration limits only
- Compressibility: optional Prandtl-Glauert correction on Cl (`use_compressibility`, off by default in Tasks 3-9)
- Stall criterion: validation airfoil ±14 deg; designed-rotor airfoil ±12 deg (flag-only, linear Cl-alpha does not bend over)
- Adopted design stall-margin limit (Tasks 6, 7, 9, 10): **no more than 5% of blade span stalled**
- Adopted mission-level tip-Mach limit (Task 9/10): **M_tip ≤ 0.72**

## Slide 5 -- 1.2 Environmental Assumptions and Data
- Atmosphere model: ISA (`bemt_solver.Atmosphere`), with a `dT_isa` offset input
- Altitudes exercised across this package: 0 m (sea level) through 3000-8000 m (search bounds)
- Sea-level density used: 1.225 kg/m^3; speed of sound 340.3 m/s
- ISA+15 hot-day condition used explicitly in Task 6 (1500 m, ISA+15)

## Slide 6 -- 1.3 Vehicle and Propulsion Assumptions and Data
Source: `task5_tiltrotor_design.py`, `AIRCRAFT` dict.
| Quantity | Value |
|---|---|
| n_rotors | 2 |
| Gross weight (MTOW) | 3000 kg |
| Empty weight | 1950 kg |
| Max payload | 550 kg |
| Fuel capacity | 500 kg |
| Reserve fuel fraction | 10% (= 50 kg) |
| Drivetrain efficiency | 0.95 |
| n_engines | 2 |
| Sea-level power per engine | 400 kW |
| Cruise SFC | 0.30 kg/kWh |
| Cruise L/D (assumption A3, mission_planner.py) | 8.5 |

## Slide 8 -- 2.1 BEMT flow diagram
**NOT AUTO-GENERATED** (draw your own). Reference for content: user inputs ->
atmosphere (`Atmosphere`) -> geometry (`RotorGeometry`) -> airfoil lookup
(`LinearAirfoil`/`TableAirfoil`) -> per-station bisection solve for vi ->
sectional Cl/Cd/dT/dQ -> radial (Simpson) integration -> CT/CQ/CP/FM + stall
& tip-Mach flags -> convergence check -> `results` dict. See
`bemt_solver.py`'s own module docstring and `BEMTSolver.solve()`.

## Slide 9 -- 2.2 Mission Planner flow diagram
**NOT AUTO-GENERATED** (draw your own). Reference for content: segment list
-> per-segment time loop -> per-step thrust target -> `_collective_for_thrust`
(root-find via BEMT) -> aerodynamic power -> shaft power (÷ drivetrain
efficiency) -> power-available check -> fuel burn (SFC × P × dt) -> mass
update -> Task-10 feasibility checks -> next step or `MissionFailure`. See
`mission_planner.py`'s module docstring and `MissionPlanner._step()`.

## Slide 11 -- 3.1 Validation Rotor and Data Preparation
Source: `data/README_validation_data.md`, `task3_validation.py`.
| Parameter | Value |
|---|---|
| Source | Knight & Hefner (1937), NACA TN 626, Table I |
| Radius R | 0.762 m |
| Root cutout | 0.125 m |
| Blades B | 2 |
| Chord | 0.0508 m |
| Solidity sigma | 0.0424 |
| Airfoil model | Cl=5.75·alpha, Cd=0.0113+1.25·alpha^2 |
| Test RPM | 960 |
| Condition | sea level, standard day |

## Slide 12 -- 3.2 Thrust Comparison
Figure: `figures/task3_2_thrust_validation.png`
Test case: at theta0 = 8.0 deg, **CT_bemt = 0.00315**, CT_exp = 0.00650.

## Slide 13 -- 3.3 Torque/Power Comparison (+ FM)
Figures: `figures/task3_3_torque_validation.png`, `figures/task3_4_figure_of_merit.png`
Test case: at theta0 = 8.0 deg, **CQ_bemt = 0.000246**, CQ_exp = 0.000494.

## Slide 14 -- 3.4 Observations, Errors, Model Limitations
Table (`outputs/task3_error_metrics.csv`):
| Coefficient | RMSE | MAE | MAPE |
|---|---|---|---|
| CT | 1.725e-3 | 1.216e-3 | 47.2% |
| CQ | 1.293e-4 | 1.069e-4 | 46.3% |
Discussion seed material: `outputs/task3_notes_for_section_3_4.md` (cites
Knight & Hefner's own conclusion that neglected slipstream contraction is
the most likely source of their theory-experiment gap too).

## Slides 16-18 -- 4.1/4.2/4.3 Design-Variable Study
Figures: `task4_1a_blade_number.png` / `task4_1b_solidity_continuous.png` /
`task4_1c_rpm_bonus.png` (bonus) / `task4_2_taper_ratio.png` / `task4_3_twist.png`.
Test case (blade-number sweep, `outputs/task4_1a_blade_number.csv`), at the
representative point (10 deg collective, 960 RPM, sea level -- NOTE: this
uses the *validation* rotor's baseline scale, R=0.762 m, not the designed
tiltrotor):
| B | sigma | T [N] | FM |
|---|---|---|---|
| 2 | 0.0355 | 54.5 | 0.535 |
| 3 | 0.0532 | 73.8 | 0.571 |
| 4 | 0.0710 | 90.4 | 0.593 |
| 5 | 0.0887 | 105.0 | 0.607 |
Twist study uses a **thrust-trimmed** method (see file header) -- FM
increases from 0.535 (untwisted) toward ~0.58 at -16 deg tip twist, at the
*same* thrust: this monotonic trend is your correctness check.

## Slide 21 -- 5.2 Aircraft Schematic
**NOT AUTO-GENERATED** -- draw your own twin-tiltrotor concept sketch.

## Slide 22 -- 5.3 Aircraft Mass and Design Requirements
Same table as Slide 6 above, plus: design range 550 km, design cruise speed
(mission requirement) 110 m/s / 214 kt, service ceiling 6000 m.

## Slide 23 -- 5.4 Rotor Design Table
Full table: `outputs/task5_rotor_design_table.csv`. Key entries:
| Parameter | Value |
|---|---|
| Radius R | 2.60 m |
| Root cutout | 0.35 m |
| Blades B | 3 |
| Root chord / taper | 0.34 m / 0.80 |
| Twist | 0.0 to -12.0 deg |
| Solidity sigma | **0.0973** |
| RPM (hover / cruise) | 750 / 650 |
| Tip speed (hover / cruise) | 204.2 / 177.0 m/s |
| M_tip (hover / cruise) | **0.600 / 0.520** |
| Collective range (hover / cruise) | 2-26 deg / 5-35 deg |
| Disk loading (per rotor, MTOW) | **70.6 kg/m^2** |

## Slide 24 -- 5.5 Rotor-Design Rationale and Trade-Offs
Evidence to cite: Task 3 validation error bands; Task 4 twist-trim trend
(FM 0.535->0.58); Task 6 hover FM ≈0.75-0.76 at MTOW; Task 7/9 cruise-speed
shortfall (see `outputs/task9_notes_cruise_trim.md`) -- the central,
honest trade-off finding of this design: -12 deg twist gives a numerically
robust, efficient HOVER rotor (FM 0.756, matches XV-15-class hover
performance) but caps trimmed cruise speed at **~27.5 m/s**, well under the
110 m/s mission target. State this explicitly; it is exactly the kind of
finding Sec 5.5 wants.

## Slide 26 -- 6.1 Hover Performance Maps
Figures: `task6_1_hover_performance_maps.png`, `task6_1_max_hover_weight_vs_altitude.png`.
Test cases (`outputs/task6_1_max_hover_weight_vs_altitude.csv`):
| Altitude | Max hover GW | Binding constraint |
|---|---|---|
| 0 m | **3205 kg** | power-limited |
| 1500 m | 2836 kg | power-limited |
| 3000 m | 2500 kg | power-limited |
**Hover ceiling at MTOW (3000 kg) = 820 m** (`outputs/task6_1_hover_ceiling.txt`).

## Slide 27 -- 6.2 Axial Forward-Flight / Propeller Maps
Figures: `task7_forward_flight_maps.png`, `task7_blade_aoa_distribution.png`.
Selected cruise point test case (`outputs/task7_selected_cruise_point.csv`,
an *untrimmed* best-efficiency grid point -- see Slide 24 note above for the
trimmed value used in Section 7):
collective=34 deg, J=0.65, V=36.6 m/s, T=6651 N, P=328 kW, **eta_p=0.742**,
stall_fraction=0.05.

## Slide 28 -- 6.3 Comparison with Comparable Rotors
Table: `outputs/task8_benchmarking_table.csv`, figure `task8_benchmarking_bars.png`.
| Rotor | R [m] | B | sigma | DL [kg/m^2] | M_tip | FM |
|---|---|---|---|---|---|---|
| This design | 2.60 | 3 | 0.097 | 70.6 | 0.600 | 0.756 |
| XV-15 | 3.81 | 3 | 0.089 | 69.6 | 0.691 | 0.75 (approx) |
| V-22 | 5.79 | 3 | 0.105 (approx) | 130.3 | 0.594 | 0.808 |

## Slide 30 -- 7.1 Mission-Planner Implementation Verification
Table: `outputs/task9_1_verification_table.csv` -- **10/10 checks pass**.
Feasible/infeasible mission test evidence: `outputs/task10_summary.md`,
`figures/task10_missionA_feasible.png`, `figures/task10_missionB_infeasible.png`.
Test case: Mission B's first violated constraint is
**segment='climb', t=1853.6 s (30.9 min), "power required 925.2 kW exceeds
power available 757.5 kW"** after an oversized +900 kg mid-mission pickup.

## Slide 31 -- 7.2 Fuel-Burn Rate vs Gross Weight
Figure: `figures/task9_2_fuel_burn_vs_gw.png`. Test case
(`outputs/task9_2_fuel_burn_vs_gw.csv`): at GW=3000 kg, hover fuel-burn rate
= **206.5 kg/hr** at sea level (2900 kg -> 196.4 kg/hr, 3100 kg -> 216.9 kg/hr).

## Slide 32 -- 7.3 Hover Endurance vs Takeoff Weight
Figure: `figures/task9_3_hover_endurance_vs_takeoff_weight.png`. Test cases
(`outputs/task9_3_hover_endurance.csv`): 2200 kg -> 4.00 hr (search cap, not
fuel-limited); 3000 kg -> **2.47 hr** (fuel-limited, reserve reached).

## Slide 33 -- 7.4 Cruise Range vs Cruise Speed
Figure: `figures/task9_4_cruise_range_vs_speed.png`. Test cases
(`outputs/task9_4_cruise_range.csv`): V=20 m/s -> 996 km; V=25 m/s ->
**1050 km** (both reserve-limited, at 3000 m altitude, MTOW).

## Slide 34 -- 7.5 Overall Design Observations
Write your own, but the two load-bearing facts from this run are:
(1) hover is comfortably feasible at MTOW (power margin ~9.5% at sea level,
hover ceiling 820 m); (2) cruise speed is the dominant shortfall (~27.5 m/s
trimmed vs 110 m/s required) -- driven by the moderated -12 deg twist choice
documented in `task5_tiltrotor_design.py`. State the Milestone-2 fix
direction: redesign with more washout (XV-15-class, -35/-40 deg) paired
with a correspondingly higher hover collective schedule, then re-validate
Tasks 6-9 don't regress.

## Slide 36 -- 8.1 Code Organization and Reproducibility
See `README.md` at the package root for the full file/purpose/how-to-run
table -- paste that table directly onto this slide.

## Slide 37 -- 8.2 References
See `REFERENCES.md` at the package root for a consolidated, ready-to-paste
numbered reference list covering every external source used in this code
(Knight & Hefner 1937/NACA TN 626, XV-15 and V-22 sources, etc.).

## Slide 38 -- 8.3 Acknowledgement and Assistance Disclosure
**NOT AUTO-GENERATED** -- your team must write this, but factually: this
code package (bemt_solver.py's completion in Tasks 3-10, mission_planner.py,
and this reference document) was produced with Claude (Anthropic) assistance.
Course policy explicitly permits disclosed generative-AI assistance (see the
handout's Sec 8.3 description) -- disclose it here, and independently verify
the numbers above by re-running the scripts yourselves before submitting.
