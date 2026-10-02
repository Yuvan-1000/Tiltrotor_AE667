"""Single source of truth for the Milestone-2 tiltrotor configuration.

All task scripts import the same aircraft/rotor data so report values and
simulation inputs cannot drift apart.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict
import json
import numpy as np
from bemt_solver import RotorGeometry, LinearAirfoil

G = 9.80665

@dataclass(frozen=True)
class RotorConfig:
    R: float = 2.60
    r_root: float = 0.35
    B: int = 3
    chord_root: float = 0.34
    taper_ratio: float = 0.80
    twist_root_deg: float = 0.0
    twist_tip_deg: float = -12.0
    n_stations: int = 80
    rpm_hover: float = 750.0
    rpm_cruise: float = 650.0
    collective_hover_deg_min: float = 2.0
    collective_hover_deg_max: float = 26.0
    collective_cruise_deg_min: float = 5.0
    collective_cruise_deg_max: float = 35.0
    cyclic_deg_limit: float = 8.0
    stall_alpha_deg: float = 12.0
    # M1 used 5% span as a conservative target. M2 retains that as a warning
    # threshold and adds a documented 25% hard transition rejection threshold
    # for the low-order linear-polar conversion demonstration.
    stall_span_fraction_warning: float = 0.05
    stall_span_fraction_limit: float = 0.25
    tip_mach_limit: float = 0.72

@dataclass(frozen=True)
class AircraftConfig:
    n_rotors: int = 2
    gross_weight_kg: float = 3000.0
    empty_weight_kg: float = 1950.0
    max_payload_kg: float = 550.0
    fuel_capacity_kg: float = 500.0
    reserve_fuel_fraction: float = 0.10
    takeoff_altitude_m: float = 0.0
    service_ceiling_m: float = 6000.0
    design_range_km: float = 550.0
    design_cruise_speed_ms: float = 110.0
    drivetrain_efficiency: float = 0.95
    n_engines: int = 2
    sfc_kg_per_kWh: float = 0.30
    sea_level_power_per_engine_kW: float = 400.0
    # Preliminary M2 transition airframe model.
    # The wing is enlarged relative to the M1 placeholder to provide explicit
    # lift sharing through conversion rather than relying on the old L/D proxy.
    wing_area_m2: float = 60.0
    wing_span_m: float = 12.0
    wing_mean_chord_m: float = 5.0
    wing_incidence_deg: float = 2.0
    wing_cl_alpha_per_rad: float = 5.2
    wing_cd0: float = 0.030
    wing_oswald: float = 0.78
    wing_airfoil: str = "linear-CL / parabolic-CD"
    wing_cl_max: float = 1.40
    htail_area_m2: float = 10.0
    htail_span_m: float = 3.8
    htail_arm_m: float = 5.0
    htail_incidence_deg: float = -1.0
    htail_cl_alpha_per_rad: float = 4.2
    htail_cd0: float = 0.012
    htail_efficiency: float = 0.85
    vtail_area_m2: float = 3.0
    vtail_arm_m: float = 4.5
    vtail_cy_rudder_per_rad: float = 0.45
    vtail_cn_rudder_per_rad: float = -0.22
    fuselage_flat_plate_m2: float = 0.75
    nacelle_flat_plate_each_m2: float = 0.12
    wing_cm0: float = 0.02
    wing_cma_per_rad: float = -0.25
    cg_x_m: float = 0.0
    cg_y_m: float = 0.0
    cg_z_m: float = 0.0
    rotor_hub_x_m: float = 0.35
    rotor_hub_z_m: float = 0.0
    rotor_half_span_m: float = 3.45
    elevator_deg_limit: float = 15.0
    rudder_deg_limit: float = 15.0
    aileron_deg_limit: float = 10.0
    nacelle_min_deg: float = 0.0
    nacelle_max_deg: float = 90.0
    nacelle_rate_deg_s: float = 5.0
    rpm_min: float = 620.0
    rpm_max: float = 780.0
    cruise_lift_to_drag: float = 8.5

ROTOR = RotorConfig()
AIRCRAFT = AircraftConfig()
TILTROTOR_GEOM = RotorGeometry(
    R=ROTOR.R, r_root=ROTOR.r_root, B=ROTOR.B,
    chord_root=ROTOR.chord_root, taper_ratio=ROTOR.taper_ratio,
    twist_root=np.radians(ROTOR.twist_root_deg),
    twist_tip=np.radians(ROTOR.twist_tip_deg),
    n_stations=ROTOR.n_stations,
)
TILTROTOR_AIRFOIL = LinearAirfoil(
    a0=5.73, cd_min=0.0090, eps=0.60,
    alpha_stall_pos=np.radians(ROTOR.stall_alpha_deg),
    alpha_stall_neg=np.radians(-ROTOR.stall_alpha_deg),
)

def config_dict() -> Dict:
    return {"rotor": asdict(ROTOR), "aircraft": asdict(AIRCRAFT)}

def save_json(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_dict(), f, indent=2)

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    save_json("outputs/aircraft_config.json")
