"""
task5_tiltrotor_design.py
================================================================================
Milestone-1 Task 5 / Report Sections 5.1-5.5: "Tiltrotor Aircraft Definition
and Rotor Design".

THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR THE DESIGNED AIRCRAFT.
Tasks 6, 7, 8, 9, and 10 all `import` the objects defined here so that the
whole report is internally consistent (one rotor, one aircraft, everywhere).
================================================================================
"""

from __future__ import annotations
import os
import csv
import numpy as np

# Primary BEMT imports with fallback handling
from bemt_solver import RotorGeometry, FlightCondition, BEMTSolver, AirfoilModel

try:
    from bemt_solver import LinearAirfoil
except ImportError:
    class LinearAirfoil(AirfoilModel):
        """Fallback linear Cl / parabolic Cd airfoil model."""
        def __init__(self, a0: float = 5.73, cd_min: float = 0.0090, eps: float = 0.60, alpha_stall: float = np.radians(12.0)):
            self.a0 = a0
            self.cd_min = cd_min
            self.eps = eps
            self.alpha_stall = alpha_stall

        def get_cl_cd(self, alpha: np.ndarray):
            alpha_arr = np.asarray(alpha)
            stalled = np.abs(alpha_arr) > self.alpha_stall
            cl = np.where(~stalled, self.a0 * alpha_arr, self.a0 * self.alpha_stall * np.sign(alpha_arr))
            cd = self.cd_min + self.eps * (alpha_arr ** 2)
            return cl, cd, stalled


OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

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
    twist_tip=np.radians(-12.0),   # [rad] moderate washout
    n_stations=80,
)

# ================================================================================
# AIRFOIL (documented assumption)
# ================================================================================
TILTROTOR_AIRFOIL = LinearAirfoil(
    a0=5.73,
    cd_min=0.0090,
    eps=0.60,
    alpha_stall=np.radians(12.0),
)

# ================================================================================
# ROTOR / DRIVE-SYSTEM SCHEDULE
# ================================================================================
RPM_HOVER = 750.0          # [RPM] helicopter-mode / hover & low-speed
RPM_CRUISE = 650.0         # [RPM] airplane-mode / cruise (87% Nr, 2-speed drive)
COLLECTIVE_RANGE_HOVER_DEG = (2.0, 26.0)   # allowable collective, helicopter mode
COLLECTIVE_RANGE_CRUISE_DEG = (5.0, 35.0)  # allowable collective, airplane mode

# ================================================================================
# AIRCRAFT MASS & MISSION REQUIREMENTS (Sec 5.3)
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
    design_cruise_speed_ms=110.0,   # Mission requirement (~214 kt)
    demonstrated_cruise_speed_ms=38.3,  # Achieved in Task 7 sweep
    hover_requirement="OGE hover at MTOW, sea level, ISA+15 C",
    drivetrain_efficiency=0.95,     # gearbox + interconnect shaft losses
    n_engines=2,
    sfc_kg_per_kWh=0.30,            # representative modern turboshaft cruise SFC
    sea_level_power_per_engine_kW=400.0,
)


def build_solver(use_root_loss: bool = False, use_compressibility: bool = False) -> BEMTSolver:
    """Instantiates a BEMTSolver configured with the single-source-of-truth tiltrotor geometry."""
    kwargs = {"use_tip_loss": True}
    if "use_root_loss" in BEMTSolver.__init__.__code__.co_varnames:
        kwargs["use_root_loss"] = use_root_loss
    if "use_compressibility" in BEMTSolver.__init__.__code__.co_varnames:
        kwargs["use_compressibility"] = use_compressibility

    return BEMTSolver(TILTROTOR_GEOM, TILTROTOR_AIRFOIL, **kwargs)


def design_summary() -> list[tuple[str, str, str, str]]:
    """Generates an itemized summary table of the tiltrotor design parameters."""
    geom = TILTROTOR_GEOM
    sigma = geom.solidity() if callable(getattr(geom, "solidity", None)) else getattr(geom, "solidity", 0.0)
    A_disk = np.pi * (geom.R ** 2)
    disk_loading_per_rotor = (AIRCRAFT["gross_weight_kg"] / AIRCRAFT["n_rotors"] * 9.80665) / A_disk   # N/m^2

    Vtip_hover = RPM_HOVER * 2.0 * np.pi / 60.0 * geom.R
    Vtip_cruise = RPM_CRUISE * 2.0 * np.pi / 60.0 * geom.R
    a_sound_sl = 340.3
    M_tip_hover = Vtip_hover / a_sound_sl
    M_tip_cruise = Vtip_cruise / a_sound_sl

    rows = [
        ("Airfoil model", "Linear Cl-alpha / parabolic Cd", "-", "a0=5.73/rad, cd_min=0.0090, eps=0.60; see module docstring"),
        ("Radius, R", f"{geom.R:.2f}", "m", "sized for disk loading ~ real tiltrotors"),
        ("Root cutout", f"{geom.r_root:.2f}", "m", "hub / pitch-bearing / tilt-mechanism allowance"),
        ("Number of blades, B", f"{geom.B}", "-", "matches XV-15/V-22 practice"),
        ("Root chord", f"{geom.chord_root:.3f}", "m", "sets solidity"),
        ("Taper ratio", f"{geom.taper_ratio:.2f}", "-", f"tip chord = {geom.chord_root * geom.taper_ratio:.3f} m"),
        ("Twist (linear)", f"{np.degrees(geom.twist_root):.1f} to {np.degrees(geom.twist_tip):.1f}", "deg", "moderate washout"),
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

    csv_path = os.path.join(OUT_DIR, "task5_rotor_design_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "Units", "Constraint_or_rationale"])
        for r in rows:
            writer.writerow(r)

    print(f"\nDesign table written to {csv_path}")