"""
task5_tiltrotor_design.py
================================================================================
Milestone-1 Task 5 / Report Sections 5.1-5.5: "Tiltrotor Aircraft Definition
and Rotor Design".

THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR THE DESIGNED AIRCRAFT.
Tasks 6, 7, 8, 9, and 10 all `import` the objects defined here so that the
whole report is internally consistent (one rotor, one aircraft, everywhere).

IMPORTANT -- READ BEFORE YOU SUBMIT:
The numbers below are a *complete, internally-consistent, defensible*
preliminary design so that Tasks 6-10 have something concrete to run against
and so you have a worked example of the design *process*. They are NOT "the
answer" -- Sec 5.1/5.5 explicitly asks your team to justify ITS OWN design
philosophy and trade-offs. Treat every number in TILTROTOR_DESIGN_TABLE as a
starting point: change what your team's mission/requirements actually call
for, re-run tasks 6-10, and write the rationale in your own words.

WHERE THESE NUMBERS CAME FROM (so you can cite them / replace them):
  - Twin 3-bladed proprotors, disk loading ~65-90 kg/m^2, are sized to be
    in the family of real tiltrotors: Bell XV-15 (R=3.81 m, sigma=0.089-
    0.10, twist ~ -40 deg, hover RPM 589, design GW ~5900-6800 kg) and the
    Bell-Boeing V-22 (R=5.79 m, 3 blades, MTOW-class disk loading
    ~120-130 kg/m^2). Sources used for these reference numbers (also used
    again for Task 8 benchmarking): see task8_benchmarking.py header.
  - Our aircraft is deliberately smaller than either (a "light utility
    tiltrotor" class, GW ~3000 kg) so BEMT run times / plots stay
    manageable, while keeping nondimensional parameters (sigma, disk
    loading, tip Mach) in the same family as real tiltrotors -- this is
    what makes the Task 8 nondimensional benchmarking meaningful.
  - TWIST: real tiltrotor blades use very large built-in washout (XV-15:
    -40.25 deg total) because their reference/root collective runs much
    higher (~30-45 deg) than a helicopter's. Combining a -40 deg-class
    twist with a *helicopter-like* 2-20 deg collective range (as this
    milestone's hover collective schedule uses) drives the blade tip to a
    strongly NEGATIVE local pitch and therefore NEGATIVE local lift --
    we hit exactly this failure mode during development (see
    dev_notes_bemt_quirks.md) and it is a good example of a self-
    consistency check your team should run on YOUR OWN geometry+twist+
    collective combination before trusting any downstream plot. We
    therefore adopt a more MODERATE washout (-12 deg tip-relative) here:
    still visibly more than a classic helicopter blade (-8 to -12 deg)
    while remaining well-behaved across the full adopted collective
    range. If your team wants the full tiltrotor-class -35/-40 deg
    washout, you must also raise the collective range accordingly (see
    the check in dev_notes_bemt_quirks.md) and re-verify tip AoA stays
    reasonable before using the results.
  - The airfoil is NOT a real published polar (none was provided in the
    handout beyond the Knight & Hefner validation airfoil, which is a
    1930s symmetric NACA 0015 unsuited to a modern proprotor). We keep the
    same *linear Cl-alpha / parabolic Cd* functional form validated in
    Task 3 (so the same, already-validated solver logic applies) but
    adopt a lower minimum profile drag (cd_min = 0.0090 vs 0.0113) and
    lower drag-rise coefficient (eps = 0.60 vs 1.25), representative of a
    modern low-drag laminar-flow rotor-airfoil family (e.g. an NACA
    63-2XX / SC1095-class section) rather than the 1937 NACA 0015. This is
    a DOCUMENTED ASSUMPTION your team should state explicitly in Sec 1.1
    / 5.1 -- replace with a real digitized polar (bemt_solver.TableAirfoil
    accepts one directly) if your team obtains one.
================================================================================
"""

import numpy as np
from bemt_solver import RotorGeometry, LinearAirfoil, FlightCondition, BEMTSolver

# ================================================================================
# ROTOR GEOMETRY
# ================================================================================
TILTROTOR_GEOM = RotorGeometry(
    R=2.60,             # [m] blade tip radius
    r_root=0.35,        # [m] root cutout (hub, pitch bearings, tilt mechanism)
    B=3,                # [-] blade count (matches XV-15 / V-22 practice)
    chord_root=0.34,    # [m] chord at root-cutout station
    taper_ratio=0.8,    # [-] tip chord / root chord -> tip chord = 0.272 m
    twist_root=0.0,     # [rad] reference station, twist measured relative to this
    twist_tip=np.radians(-12.0),   # [rad] moderate washout -- see docstring above
    n_stations=80,
)

# ================================================================================
# AIRFOIL (documented assumption -- see module docstring)
# ================================================================================
TILTROTOR_AIRFOIL = LinearAirfoil(
    a0=5.73, cd_min=0.0090, eps=0.60,
    alpha_stall_pos=np.radians(12.0), alpha_stall_neg=np.radians(-12.0),
)

# ================================================================================
# ROTOR / DRIVE-SYSTEM SCHEDULE
# ================================================================================
RPM_HOVER = 750.0          # [RPM] helicopter-mode / hover & low-speed
RPM_CRUISE = 650.0         # [RPM] airplane-mode / cruise (87% Nr, 2-speed
                            # drive system, matches XV-15 hover/cruise ratio)
COLLECTIVE_RANGE_HOVER_DEG = (2.0, 26.0)   # allowable collective, helicopter mode
                            # (23.7 deg is needed to hover at design MTOW at
                            # sea level with this rotor -- see task6 output;
                            # 26 deg keeps a stall margin above that)
COLLECTIVE_RANGE_CRUISE_DEG = (5.0, 35.0)  # allowable collective/blade angle, airplane mode

# ================================================================================
# AIRCRAFT MASS & MISSION REQUIREMENTS  (Sec 5.3)
# ================================================================================
AIRCRAFT = dict(
    n_rotors=2,
    gross_weight_kg=3000.0,
    empty_weight_kg=1950.0,
    max_payload_kg=550.0,
    fuel_capacity_kg=500.0,
    reserve_fuel_fraction=0.10,     # 10% of usable fuel held as reserve
    takeoff_altitude_m=0.0,
    service_ceiling_m=6000.0,
    design_range_km=550.0,
    design_cruise_speed_ms=110.0,   # MISSION REQUIREMENT (~214 kt) -- see
                                     # note below; NOT yet demonstrated by
                                     # this Milestone-1 rotor.
    demonstrated_cruise_speed_ms=38.3,  # ACHIEVED in Task 7's forward-flight
                                     # sweep: best propulsive efficiency
                                     # (eta_p=0.747) found within the adopted
                                     # 5% stall-margin limit, at RPM_CRUISE,
                                     # collective=35 deg (COLLECTIVE_RANGE_
                                     # CRUISE_DEG upper bound), J=0.68. This
                                     # is well short of the 110 m/s mission
                                     # target -- see task7_forward_flight_
                                     # assessment.py and Sec 7.5/Milestone-2
                                     # notes: this rotor's modest -12 deg
                                     # twist (chosen for hover robustness,
                                     # see task5 docstring) does not carry
                                     # enough built-in washout to stay
                                     # unstalled at high advance ratio. A
                                     # real tiltrotor blade (-35 to -40 deg
                                     # twist, e.g. XV-15) resolves this by
                                     # combining much more washout with a
                                     # correspondingly higher hover
                                     # collective schedule -- flagged here
                                     # as a required Milestone-2 redesign
                                     # item rather than silently patched.
    hover_requirement="OGE hover at MTOW, sea level, ISA+15 C",
    drivetrain_efficiency=0.95,     # gearbox + interconnect shaft losses
    n_engines=2,
    sfc_kg_per_kWh=0.30,            # representative modern turboshaft cruise SFC
    sea_level_power_per_engine_kW=400.0,
)


def build_solver(use_root_loss=False, use_compressibility=False):
    return BEMTSolver(TILTROTOR_GEOM, TILTROTOR_AIRFOIL,
                       use_tip_loss=True, use_root_loss=use_root_loss,
                       use_compressibility=use_compressibility)


def design_summary():
    geom = TILTROTOR_GEOM
    sigma = geom.solidity()
    A_disk = np.pi * geom.R ** 2
    disk_loading_per_rotor = (AIRCRAFT["gross_weight_kg"] / AIRCRAFT["n_rotors"]
                               * 9.80665 / A_disk)   # N/m^2
    Vtip_hover = RPM_HOVER * 2.0 * np.pi / 60.0 * geom.R
    Vtip_cruise = RPM_CRUISE * 2.0 * np.pi / 60.0 * geom.R
    a_sound_sl = 340.3
    M_tip_hover = Vtip_hover / a_sound_sl
    M_tip_cruise = Vtip_cruise / a_sound_sl

    rows = [
        ("Airfoil model", "Linear Cl-alpha / parabolic Cd (documented assumption)", "-",
         "a0=5.73/rad, cd_min=0.0090, eps=0.60; see module docstring"),
        ("Radius, R", f"{geom.R:.2f}", "m", "sized for disk loading ~ real tiltrotors"),
        ("Root cutout", f"{geom.r_root:.2f}", "m", "hub / pitch-bearing / tilt-mechanism allowance"),
        ("Number of blades, B", f"{geom.B}", "-", "matches XV-15/V-22 practice"),
        ("Root chord", f"{geom.chord_root:.3f}", "m", "sets solidity"),
        ("Taper ratio", f"{geom.taper_ratio:.2f}", "-", "tip chord = {:.3f} m".format(geom.chord_root * geom.taper_ratio)),
        ("Twist (linear)", f"{np.degrees(geom.twist_root):.1f} to {np.degrees(geom.twist_tip):.1f}", "deg",
         "moderate washout (more than helicopter, less than full tiltrotor-class; see docstring)"),
        ("Solidity, sigma", f"{sigma:.4f}", "-", "cf. XV-15 sigma=0.089-0.10"),
        ("RPM (hover / helicopter mode)", f"{RPM_HOVER:.0f}", "RPM", "sized for M_tip~0.6 hover"),
        ("RPM (cruise / airplane mode)", f"{RPM_CRUISE:.0f}", "RPM", "2-speed drive, 87% of hover Nr"),
        ("Tip speed (hover)", f"{Vtip_hover:.1f}", "m/s", f"M_tip={M_tip_hover:.3f}"),
        ("Tip speed (cruise)", f"{Vtip_cruise:.1f}", "m/s", f"M_tip={M_tip_cruise:.3f}"),
        ("Collective range (hover)", f"{COLLECTIVE_RANGE_HOVER_DEG[0]:.0f} to {COLLECTIVE_RANGE_HOVER_DEG[1]:.0f}", "deg", "allowable pitch schedule"),
        ("Collective range (cruise)", f"{COLLECTIVE_RANGE_CRUISE_DEG[0]:.0f} to {COLLECTIVE_RANGE_CRUISE_DEG[1]:.0f}", "deg", "allowable pitch schedule"),
        ("Disk loading (per rotor, at MTOW)", f"{disk_loading_per_rotor:.1f}", "N/m^2", f"= {disk_loading_per_rotor/9.80665:.1f} kg/m^2"),
        ("Gross weight", f"{AIRCRAFT['gross_weight_kg']:.0f}", "kg", "design mission MTOW"),
        ("Empty weight", f"{AIRCRAFT['empty_weight_kg']:.0f}", "kg", "structure+systems+engines"),
        ("Max payload", f"{AIRCRAFT['max_payload_kg']:.0f}", "kg", "cabin/cargo"),
        ("Fuel capacity", f"{AIRCRAFT['fuel_capacity_kg']:.0f}", "kg", f"{AIRCRAFT['reserve_fuel_fraction']*100:.0f}% reserve policy"),
        ("Design range", f"{AIRCRAFT['design_range_km']:.0f}", "km", "airplane-mode cruise"),
        ("Design cruise speed", f"{AIRCRAFT['design_cruise_speed_ms']:.0f}", "m/s", f"= {AIRCRAFT['design_cruise_speed_ms']*1.94384:.0f} kt"),
        ("Service ceiling", f"{AIRCRAFT['service_ceiling_m']:.0f}", "m", "-"),
        ("Number of engines", f"{AIRCRAFT['n_engines']}", "-", "cross-shafted, single-engine-inoperative capable"),
        ("Sea-level power per engine", f"{AIRCRAFT['sea_level_power_per_engine_kW']:.0f}", "kW", "installed, uninstalled losses in drivetrain_efficiency"),
        ("Drivetrain efficiency", f"{AIRCRAFT['drivetrain_efficiency']:.2f}", "-", "gearbox + interconnect shaft"),
        ("Cruise SFC", f"{AIRCRAFT['sfc_kg_per_kWh']:.2f}", "kg/kWh", "representative modern turboshaft"),
    ]
    return rows


if __name__ == "__main__":
    print("=== Task 5 -- Tiltrotor rotor design table (Sec 5.4 technical entities) ===")
    rows = design_summary()
    width0 = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{width0}}  {r[1]:>18} {r[2]:<8} | {r[3]}")

    import os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/task5_rotor_design_table.csv", "w") as f:
        f.write("Parameter,Value,Units,Constraint_or_rationale\n")
        for r in rows:
            f.write(f'"{r[0]}","{r[1]}","{r[2]}","{r[3]}"\n')
    print("\nDesign table written to outputs/task5_rotor_design_table.csv")
