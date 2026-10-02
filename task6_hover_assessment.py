"""
task6_hover_assessment.py
================================================================================
Milestone-1 Task 6 / Report Section 6.1: "Hover Performance Maps and
Performance Assessment" (hover half).

Applies the BEMT tool to the DESIGNED tiltrotor rotor (task5_tiltrotor_design)
at sea level and a high-altitude/hot condition:
  - thrust, torque, power, and blade-AoA vs collective
  - stall-limited and power-limited operating regions
  - hover ceiling (max altitude for OGE hover at MTOW)
  - maximum hover gross weight at each altitude, subject to a simple
    installed-power-available model and an adopted stall-margin limit
================================================================================
"""

from __future__ import annotations
import os
import csv
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# BEMT solver imports with robust fallback handling
from bemt_solver import FlightCondition

try:
    from bemt_solver import Atmosphere
except ImportError:
    class Atmosphere:
        """Fallback ISA Atmosphere model."""
        def __init__(self, altitude: float = 0.0, dT_isa: float = 0.0):
            self.altitude = altitude
            self.dT_isa = dT_isa

        def properties(self):
            T0 = 288.15 + self.dT_isa
            p0 = 101325.0
            L = 0.0065
            T = max(T0 - L * self.altitude, 216.65)
            p = p0 * (1.0 - L * self.altitude / 288.15) ** 5.2561
            rho = p / (287.05 * T)
            a = np.sqrt(1.4 * 287.05 * T)
            mu = 1.789e-5
            return rho, p, T, a, mu

from task5_tiltrotor_design import (
    TILTROTOR_GEOM, TILTROTOR_AIRFOIL, AIRCRAFT, RPM_HOVER,
    COLLECTIVE_RANGE_HOVER_DEG, build_solver,
)

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SOLVER = build_solver()
G = 9.80665

# Design stall-margin limit: no more than 5% of blade span stalled
STALL_FRACTION_LIMIT = 0.05


# ================================================================================
# Installed-power-available model
# ================================================================================
def power_available_kW(altitude_m: float, dT_isa: float = 0.0) -> float:
    """Computes total shaft power available across all engines given altitude density lapse rate."""
    try:
        atmo = Atmosphere(altitude=altitude_m, dT_isa=dT_isa)
        rho, p, T, a, mu = atmo.properties()
    except Exception:
        rho0 = 1.225
        rho = rho0 * (1.0 - 2.25577e-5 * altitude_m) ** 4.25588

    rho0 = 1.225
    lapse = (rho / rho0) ** 0.75
    P_installed = AIRCRAFT["n_engines"] * AIRCRAFT["sea_level_power_per_engine_kW"] * lapse
    return P_installed * AIRCRAFT["drivetrain_efficiency"]


def hover_point(theta0_deg: float, altitude_m: float, dT_isa: float = 0.0, rpm: float = RPM_HOVER) -> dict:
    """Executes BEMT solver for a specific hover operating point."""
    try:
        flight = FlightCondition.from_rpm(rpm, theta0_deg, altitude=altitude_m, dT_isa=dT_isa)
    except AttributeError:
        omega = rpm * 2.0 * np.pi / 60.0
        flight = FlightCondition(Omega=omega, collective=np.radians(theta0_deg))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return SOLVER.solve(flight)


def collective_for_thrust(T_target_per_rotor: float, altitude_m: float, dT_isa: float = 0.0, rpm: float = RPM_HOVER) -> float:
    """Root-finds the collective pitch (deg) required to produce T_target_per_rotor [N]."""
    lo, hi = COLLECTIVE_RANGE_HOVER_DEG

    def thrust_err(theta0_deg: float) -> float:
        res = hover_point(theta0_deg, altitude_m, dT_isa, rpm)
        return res["T"] - T_target_per_rotor

    f_lo, f_hi = thrust_err(lo), thrust_err(hi)
    if f_lo > 0 or f_hi < 0:
        raise ValueError(
            f"Target thrust {T_target_per_rotor:.0f} N per rotor is outside collective range "
            f"[{lo},{hi}] deg at altitude={altitude_m} m (f_lo={f_lo:.0f}, f_hi={f_hi:.0f})."
        )
    return float(brentq(thrust_err, lo, hi, xtol=1e-3))


def required_power_kW(gross_weight_kg: float, altitude_m: float, dT_isa: float = 0.0, rpm: float = RPM_HOVER):
    """Calculates required shaft power (kW), collective pitch (deg), stall fraction, and tip Mach number."""
    T_per_rotor = gross_weight_kg * G / AIRCRAFT["n_rotors"]
    theta0 = collective_for_thrust(T_per_rotor, altitude_m, dT_isa, rpm)
    res = hover_point(theta0, altitude_m, dT_isa, rpm)
    P_aero_total = res["P"] * AIRCRAFT["n_rotors"]
    P_shaft_total = P_aero_total / AIRCRAFT["drivetrain_efficiency"]

    stall_frac = res.get("stall_fraction", 0.0)
    m_tip = res.get("M_tip", 0.0)
    return P_shaft_total / 1000.0, theta0, stall_frac, m_tip


def max_hover_weight(altitude_m: float, dT_isa: float = 0.0, rpm: float = RPM_HOVER,
                      gw_lo: float = 500.0, gw_hi: float = 8000.0) -> dict | None:
    """Determines maximum allowable gross weight subject to available power and stall fraction limits."""
    def feasible(gw: float):
        try:
            P_req, theta0, stall_frac, M_tip = required_power_kW(gw, altitude_m, dT_isa, rpm)
        except ValueError:
            return False, None
        P_av = power_available_kW(altitude_m, dT_isa)
        ok = (P_req <= P_av) and (stall_frac <= STALL_FRACTION_LIMIT)
        return ok, (P_req, P_av, theta0, stall_frac, M_tip)

    if not feasible(gw_lo)[0]:
        return None

    lo, hi = gw_lo, gw_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ok, _ = feasible(mid)
        if ok:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0:
            break

    _, info = feasible(lo)
    P_req, P_av, theta0, stall_frac, M_tip = info
    power_margin = P_av - P_req
    stall_margin = STALL_FRACTION_LIMIT - stall_frac
    binding = "power-limited" if power_margin < 1.0 else (
        "stall-limited" if stall_margin < 0.002 else "power-limited")

    return dict(
        gross_weight_kg=lo, P_required_kW=P_req, P_available_kW=P_av,
        theta0_deg=theta0, stall_fraction=stall_frac, M_tip=M_tip,
        binding_constraint=binding
    )


def hover_ceiling(gross_weight_kg: float, dT_isa: float = 0.0, alt_lo: float = 0.0, alt_hi: float = 8000.0) -> float | None:
    """Finds maximum OGE hover ceiling altitude for a given gross weight."""
    def feasible(h: float) -> bool:
        try:
            P_req, theta0, stall_frac, _ = required_power_kW(gross_weight_kg, h, dT_isa)
        except ValueError:
            return False
        P_av = power_available_kW(h, dT_isa)
        return (P_req <= P_av) and (stall_frac <= STALL_FRACTION_LIMIT)

    if not feasible(alt_lo):
        return 0.0
    if feasible(alt_hi):
        return None

    lo, hi = alt_lo, alt_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < 10.0:
            break
    return lo


# ================================================================================
# 6.1(a) Hover Performance Maps
# ================================================================================
def hover_performance_maps():
    theta0_grid = np.linspace(9.0, 26.0, 25)
    conditions = [
        dict(label="Sea level, ISA", altitude=0.0, dT_isa=0.0, color="tab:blue"),
        dict(label="1500 m, ISA+15", altitude=1500.0, dT_isa=15.0, color="tab:red"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    csv_rows = []
    for cond in conditions:
        T_list, Q_list, P_list, aoa_tip_list = [], [], [], []
        for theta0 in theta0_grid:
            res = hover_point(theta0, cond["altitude"], cond["dT_isa"])

            alpha = res.get("alpha_deg", res.get("alpha", np.array([0.0])))
            tip_aoa = alpha[-1] if isinstance(alpha, (list, np.ndarray)) else alpha
            if np.isnan(tip_aoa) or tip_aoa < -100:
                tip_aoa = np.degrees(alpha[-1]) if isinstance(alpha, (list, np.ndarray)) else np.degrees(alpha)

            T_list.append(res["T"])
            Q_list.append(res["Q"])
            P_list.append(res["P"] / 1000.0)
            aoa_tip_list.append(tip_aoa)

            stall_frac = res.get("stall_fraction", 0.0)
            m_tip = res.get("M_tip", 0.0)

            csv_rows.append((cond["label"], theta0, res["T"], res["Q"], res["P"],
                              tip_aoa, stall_frac, m_tip))

        axes[0, 0].plot(theta0_grid, T_list, "-o", ms=3, color=cond["color"], label=cond["label"])
        axes[0, 1].plot(theta0_grid, Q_list, "-o", ms=3, color=cond["color"], label=cond["label"])
        axes[1, 0].plot(theta0_grid, P_list, "-o", ms=3, color=cond["color"], label=cond["label"])
        axes[1, 1].plot(theta0_grid, aoa_tip_list, "-o", ms=3, color=cond["color"], label=cond["label"])

    axes[0, 0].set_xlabel("Collective [deg]")
    axes[0, 0].set_ylabel("Thrust per rotor, T [N]")
    axes[0, 0].set_title("6.1 Thrust vs collective")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_xlabel("Collective [deg]")
    axes[0, 1].set_ylabel("Torque per rotor, Q [N.m]")
    axes[0, 1].set_title("6.1 Torque vs collective")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_xlabel("Collective [deg]")
    axes[1, 0].set_ylabel("Power per rotor, P [kW]")
    axes[1, 0].set_title("6.1 Power vs collective")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].axhline(12.0, color="k", ls="--", lw=1, label="adopted stall AoA (12 deg)")
    axes[1, 1].set_xlabel("Collective [deg]")
    axes[1, 1].set_ylabel("Blade-tip AoA [deg]")
    axes[1, 1].set_title("6.1 Blade-tip angle of attack vs collective")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(f"Task 6.1 -- Hover performance maps, designed rotor "
                 f"(R={TILTROTOR_GEOM.R} m, B={TILTROTOR_GEOM.B}, RPM={RPM_HOVER:.0f})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "task6_1_hover_performance_maps.png"), dpi=160)
    plt.close(fig)

    csv_path = os.path.join(OUT_DIR, "task6_1_hover_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "theta0_deg", "T_N", "Q_Nm", "P_W", "tip_AoA_deg", "stall_fraction", "M_tip"])
        for r in csv_rows:
            writer.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.4f}", f"{r[4]:.2f}", f"{r[5]:.3f}", f"{r[6]:.4f}", f"{r[7]:.4f}"])

    print(f"Hover performance maps written to {FIG_DIR}/task6_1_hover_performance_maps.png")


# ================================================================================
# 6.1(b) Hover Ceiling & Max Hover Gross Weight
# ================================================================================
def hover_ceiling_and_max_weight():
    print("\n--- Max hover gross weight at representative altitudes ---")
    altitudes = [0.0, 1000.0, 1500.0, 2000.0, 3000.0]
    rows = []
    for h in altitudes:
        result = max_hover_weight(h, dT_isa=0.0)
        if result is None:
            print(f"altitude={h:.0f} m: cannot hover even at lower search bound")
            continue
        print(f"altitude={h:.0f} m  ->  max hover GW = {result['gross_weight_kg']:.0f} kg  "
              f"(theta0={result['theta0_deg']:.2f} deg, P_req={result['P_required_kW']:.1f} kW, "
              f"P_avail={result['P_available_kW']:.1f} kW, stall_frac={result['stall_fraction']:.3f}, "
              f"binding={result['binding_constraint']})")
        rows.append((h, result["gross_weight_kg"], result["theta0_deg"],
                     result["P_required_kW"], result["P_available_kW"],
                     result["stall_fraction"], result["M_tip"], result["binding_constraint"]))

    csv_path = os.path.join(OUT_DIR, "task6_1_max_hover_weight_vs_altitude.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["altitude_m", "max_GW_kg", "theta0_deg", "P_required_kW", "P_available_kW", "stall_fraction", "M_tip", "binding_constraint"])
        for r in rows:
            writer.writerow(r)

    alt_arr = [r[0] for r in rows]
    gw_arr = [r[1] for r in rows]

    plt.figure(figsize=(6, 4.5))
    plt.plot(alt_arr, gw_arr, "o-", color="tab:purple")
    plt.axhline(AIRCRAFT["gross_weight_kg"], color="k", ls="--", lw=1, label="design MTOW")
    plt.xlabel("Altitude [m]")
    plt.ylabel("Max hover gross weight [kg]")
    plt.title("Task 6.1 -- Max OGE hover gross weight vs altitude")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task6_1_max_hover_weight_vs_altitude.png"), dpi=160)
    plt.close()

    print("\n--- Hover ceiling at design MTOW ---")
    ceiling = hover_ceiling(AIRCRAFT["gross_weight_kg"])
    if ceiling is None:
        print(f"MTOW={AIRCRAFT['gross_weight_kg']:.0f} kg: hover feasible up to 8000 m")
    else:
        print(f"MTOW={AIRCRAFT['gross_weight_kg']:.0f} kg -> hover ceiling (OGE) = {ceiling:.0f} m")
        with open(os.path.join(OUT_DIR, "task6_1_hover_ceiling.txt"), "w") as f:
            f.write(f"Hover ceiling (OGE) at MTOW={AIRCRAFT['gross_weight_kg']:.0f} kg: {ceiling:.0f} m\n")

    return rows, ceiling


if __name__ == "__main__":
    hover_performance_maps()
    hover_ceiling_and_max_weight()
    print(f"\nAll figures written to ./{FIG_DIR}/task6_*.png")
    print(f"All outputs written to ./{OUT_DIR}/task6_*.csv")