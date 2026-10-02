"""
task4_design_variable_study.py
================================================================================
Milestone-1 Task 4 / Report Sections 4.1-4.3: Rotor Design-Variable Study.

Uses the validated baseline rotor (Knight & Hefner geometry/airfoil) and independently sweeps:
  4.1  Solidity / blade number   (>= 4 values)
  4.2  Taper ratio               (>= 4 values)
  4.3  Linear twist              (>= 4 values, trimmed to constant thrust)
  4.1c Rotational speed (RPM)    (bonus parameter)

holding every other parameter fixed at a single representative hover operating point.
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

# Primary solver imports with fallback for LinearAirfoil
from bemt_solver import RotorGeometry, FlightCondition, BEMTSolver, AirfoilModel

try:
    from bemt_solver import LinearAirfoil
except ImportError:
    class LinearAirfoil(AirfoilModel):
        """Fallback thin-airfoil implementation with linear lift slope."""
        def __init__(self, a0: float = 5.75, cd_min: float = 0.0113, eps: float = 1.25, alpha_stall: float = np.radians(14.0)):
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


FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================================
# Baseline Setup
# ================================================================================
R_BASE = 0.762
R_ROOT_BASE = 0.125
CHORD_BASE = 0.0508
B_BASE = 2
AIRFOIL = LinearAirfoil(a0=5.75, cd_min=0.0113, eps=1.25, alpha_stall=np.radians(14.0))

OP_RPM = 960.0
OP_COLLECTIVE_DEG = 10.0

try:
    BASE_FLIGHT = FlightCondition.from_rpm(OP_RPM, collective_deg=OP_COLLECTIVE_DEG, altitude=0.0, dT_isa=0.0)
except AttributeError:
    omega_base = OP_RPM * 2.0 * np.pi / 60.0
    BASE_FLIGHT = FlightCondition(Omega=omega_base, collective=np.radians(OP_COLLECTIVE_DEG))


def run_point(geom: RotorGeometry, flight: FlightCondition = BASE_FLIGHT, airfoil: AirfoilModel = AIRFOIL) -> dict:
    """Solves hover BEMT for a given rotor geometry and flight condition."""
    solver = BEMTSolver(geom, airfoil, use_tip_loss=True, use_root_loss=False)
    return solver.solve(flight)


def save_and_plot(x: list | np.ndarray, T: list | np.ndarray, P: list | np.ndarray, eff: list | np.ndarray,
                  xlabel: str, title_prefix: str, fname_prefix: str, extra_label: str = ""):
    """Generates a 3-panel plot showing Thrust, Power, and Figure of Merit."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(x, T, "o-", color="tab:blue")
    axes[0].ticklabel_format(useOffset=False, style="plain", axis="y")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Thrust, T [N]")
    axes[0].set_title(f"{title_prefix}: Thrust")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, np.array(P) / 1000.0, "o-", color="tab:red")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Power, P [kW]")
    axes[1].set_title(f"{title_prefix}: Power")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x, eff, "o-", color="tab:green")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel("Figure of Merit, FM")
    axes[2].set_title(f"{title_prefix}: Efficiency (FM)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Task 4 - {title_prefix} sweep{extra_label} "
                 f"(collective={OP_COLLECTIVE_DEG:.0f} deg, {OP_RPM:.0f} RPM, sea level)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{fname_prefix}.png"), dpi=160)
    plt.close(fig)


# ================================================================================
# 4.1(a) - Blade-Number Variation
# ================================================================================
def study_blade_number():
    B_values = [2, 3, 4, 5]
    rows = []
    for B in B_values:
        geom = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=B,
                              chord_root=CHORD_BASE, taper_ratio=1.0, n_stations=80)
        res = run_point(geom)

        sol = geom.solidity() if callable(getattr(geom, "solidity", None)) else getattr(geom, "solidity", 0.0)
        fm = res.get("FM", res.get("figure_of_merit", 0.0))
        stall_frac = res.get("stall_fraction", 0.0)
        m_tip = res.get("M_tip", 0.0)

        rows.append((B, sol, res["T"], res["P"], fm, stall_frac, m_tip))
        print(f"B={B}  sigma={sol:.4f}  T={res['T']:.1f} N  "
              f"P={res['P']/1000:.2f} kW  FM={fm:.3f}  stall_frac={stall_frac:.2f}")

    B_arr = [r[0] for r in rows]
    T_arr = [r[2] for r in rows]
    P_arr = [r[3] for r in rows]
    FM_arr = [r[4] for r in rows]

    save_and_plot(B_arr, T_arr, P_arr, FM_arr, "Number of blades, B",
                  "4.1(a) Blade number", "task4_1a_blade_number",
                  extra_label=" (chord fixed -> discrete solidity steps)")

    with open(os.path.join(OUT_DIR, "task4_1a_blade_number.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["B", "sigma", "T_N", "P_W", "FM", "stall_fraction", "M_tip"])
        for r in rows:
            writer.writerow(r)
    return rows


# ================================================================================
# 4.1(b) - Continuous Solidity Variation via Chord
# ================================================================================
def study_solidity_continuous():
    chord_values = np.linspace(0.03, 0.09, 7)
    rows = []
    for c in chord_values:
        geom = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=2,
                              chord_root=c, taper_ratio=1.0, n_stations=80)
        res = run_point(geom)

        sol = geom.solidity() if callable(getattr(geom, "solidity", None)) else getattr(geom, "solidity", 0.0)
        fm = res.get("FM", res.get("figure_of_merit", 0.0))
        stall_frac = res.get("stall_fraction", 0.0)

        rows.append((c, sol, res["T"], res["P"], fm, stall_frac))

    sigma_arr = [r[1] for r in rows]
    T_arr = [r[2] for r in rows]
    P_arr = [r[3] for r in rows]
    FM_arr = [r[4] for r in rows]

    save_and_plot(sigma_arr, T_arr, P_arr, FM_arr, r"Solidity, $\sigma$",
                  "4.1(b) Solidity (continuous, via chord)",
                  "task4_1b_solidity_continuous")

    with open(os.path.join(OUT_DIR, "task4_1b_solidity_continuous.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chord_m", "sigma", "T_N", "P_W", "FM", "stall_fraction"])
        for r in rows:
            writer.writerow(r)
    return rows


# ================================================================================
# 4.2 - Taper Ratio Variation
# ================================================================================
def study_taper():
    taper_values = [0.4, 0.6, 0.8, 0.9, 1.0]
    rows = []
    for tr in taper_values:
        geom = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=B_BASE,
                              chord_root=CHORD_BASE, taper_ratio=tr, n_stations=80)
        res = run_point(geom)

        sol = geom.solidity() if callable(getattr(geom, "solidity", None)) else getattr(geom, "solidity", 0.0)
        fm = res.get("FM", res.get("figure_of_merit", 0.0))
        stall_frac = res.get("stall_fraction", 0.0)

        rows.append((tr, sol, res["T"], res["P"], fm, stall_frac))
        print(f"taper={tr:.2f}  sigma={sol:.4f}  T={res['T']:.1f} N  "
              f"P={res['P']/1000:.2f} kW  FM={fm:.3f}")

    x_arr = [r[0] for r in rows]
    T_arr = [r[2] for r in rows]
    P_arr = [r[3] for r in rows]
    FM_arr = [r[4] for r in rows]

    save_and_plot(x_arr, T_arr, P_arr, FM_arr, "Taper ratio (tip chord / root chord)",
                  "4.2 Taper ratio", "task4_2_taper_ratio")

    with open(os.path.join(OUT_DIR, "task4_2_taper_ratio.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["taper_ratio", "sigma", "T_N", "P_W", "FM", "stall_fraction"])
        for r in rows:
            writer.writerow(r)
    return rows


# ================================================================================
# 4.3 - Linear Twist Variation (Trimmed to Constant Thrust)
# ================================================================================
def find_trim_collective(geom: RotorGeometry, T_target: float, rpm: float = OP_RPM,
                          altitude: float = 0.0, dT_isa: float = 0.0,
                          theta0_scan_deg: np.ndarray = np.arange(4.0, 30.0 + 1e-9, 1.0)) -> float:
    """Finds the collective pitch reproducing T_target using bracketed root finding."""
    pts = []
    for theta0 in theta0_scan_deg:
        try:
            flight = FlightCondition.from_rpm(rpm, theta0, altitude=altitude, dT_isa=dT_isa)
        except AttributeError:
            omega = rpm * 2.0 * np.pi / 60.0
            flight = FlightCondition(Omega=omega, collective=np.radians(theta0))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = run_point(geom, flight=flight)

        if res.get("converged", True):
            pts.append((theta0, res["T"] - T_target))

    for (t_a, f_a), (t_b, f_b) in zip(pts, pts[1:]):
        if abs(f_a) < 1e-6:
            return t_a
        if f_a * f_b < 0:
            def thrust_error(theta0_deg: float) -> float:
                try:
                    f_cond = FlightCondition.from_rpm(rpm, theta0_deg, altitude=altitude, dT_isa=dT_isa)
                except AttributeError:
                    om = rpm * 2.0 * np.pi / 60.0
                    f_cond = FlightCondition(Omega=om, collective=np.radians(theta0_deg))
                res = run_point(geom, flight=f_cond)
                return res["T"] - T_target

            try:
                return float(brentq(thrust_error, t_a, t_b, xtol=1e-3))
            except Exception:
                # Fallback to linear interpolation within bracket
                return float(t_a - f_a * (t_b - t_a) / (f_b - f_a))

    raise RuntimeError(f"No bracket found for T_target={T_target:.2f} N in scan range.")


def study_twist():
    twist_tip_deg_values = [0.0, -4.0, -8.0, -12.0, -16.0]

    geom0 = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=B_BASE,
                           chord_root=CHORD_BASE, taper_ratio=1.0,
                           twist_root=0.0, twist_tip=0.0, n_stations=80)
    T_target = run_point(geom0)["T"]
    print(f"Trimming every twist case to match baseline thrust T_target = {T_target:.2f} N")

    rows = []
    for tw in twist_tip_deg_values:
        geom = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=B_BASE,
                              chord_root=CHORD_BASE, taper_ratio=1.0,
                              twist_root=0.0, twist_tip=np.radians(tw), n_stations=80)

        theta0_trim = find_trim_collective(geom, T_target)
        try:
            flight_trim = FlightCondition.from_rpm(OP_RPM, theta0_trim, altitude=0.0, dT_isa=0.0)
        except AttributeError:
            omega = OP_RPM * 2.0 * np.pi / 60.0
            flight_trim = FlightCondition(Omega=omega, collective=np.radians(theta0_trim))

        res = run_point(geom, flight=flight_trim)
        fm = res.get("FM", res.get("figure_of_merit", 0.0))
        stall_frac = res.get("stall_fraction", 0.0)
        conv = res.get("converged", True)

        rows.append((tw, theta0_trim, res["T"], res["P"], fm, stall_frac))
        print(f"twist_tip={tw:+.1f} deg  (trim theta0={theta0_trim:.2f} deg)  "
              f"T={res['T']:.1f} N  P={res['P']/1000:.3f} kW  FM={fm:.3f}  "
              f"stall_frac={stall_frac:.2f}  converged={conv}")

    x_arr = [r[0] for r in rows]
    T_arr = [r[2] for r in rows]
    P_arr = [r[3] for r in rows]
    FM_arr = [r[4] for r in rows]

    save_and_plot(x_arr, T_arr, P_arr, FM_arr,
                  "Linear twist, tip relative to root [deg]",
                  "4.3 Linear twist (trimmed to constant thrust)", "task4_3_twist",
                  extra_label=f", T held = {T_target:.1f} N via trimmed collective")

    with open(os.path.join(OUT_DIR, "task4_3_twist.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["twist_tip_deg", "theta0_trim_deg", "T_N", "P_W", "FM", "stall_fraction"])
        for r in rows:
            writer.writerow(r)
    return rows


# ================================================================================
# 4.1(c) - Rotational Speed Variation (Bonus)
# ================================================================================
def study_rpm():
    rpm_values = [700, 850, 960, 1100, 1250]
    rows = []
    geom = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=B_BASE,
                          chord_root=CHORD_BASE, taper_ratio=1.0, n_stations=80)
    for rpm in rpm_values:
        try:
            flight = FlightCondition.from_rpm(rpm, collective_deg=OP_COLLECTIVE_DEG, altitude=0.0, dT_isa=0.0)
        except AttributeError:
            omega = rpm * 2.0 * np.pi / 60.0
            flight = FlightCondition(Omega=omega, collective=np.radians(OP_COLLECTIVE_DEG))

        res = run_point(geom, flight=flight)
        fm = res.get("FM", res.get("figure_of_merit", 0.0))
        m_tip = res.get("M_tip", 0.0)

        rows.append((rpm, res["T"], res["P"], fm, m_tip))
        print(f"RPM={rpm}  T={res['T']:.1f} N  P={res['P']/1000:.2f} kW  "
              f"FM={fm:.3f}  M_tip={m_tip:.3f}")

    x_arr = [r[0] for r in rows]
    T_arr = [r[1] for r in rows]
    P_arr = [r[2] for r in rows]
    FM_arr = [r[3] for r in rows]

    save_and_plot(x_arr, T_arr, P_arr, FM_arr, "Rotor speed [RPM]",
                  "4.1(c) Rotational speed (bonus)", "task4_1c_rpm_bonus")

    with open(os.path.join(OUT_DIR, "task4_1c_rpm_bonus.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rpm", "T_N", "P_W", "FM", "M_tip"])
        for r in rows:
            writer.writerow(r)
    return rows


if __name__ == "__main__":
    print("=== 4.1(a) Blade number ===")
    study_blade_number()
    print("\n=== 4.1(b) Solidity (continuous) ===")
    study_solidity_continuous()
    print("\n=== 4.2 Taper ratio ===")
    study_taper()
    print("\n=== 4.3 Twist ===")
    study_twist()
    print("\n=== 4.1(c) RPM (bonus) ===")
    study_rpm()
    print(f"\nAll figures written to ./{FIG_DIR}/task4_*.png")
    print(f"All tables written to ./{OUT_DIR}/task4_*.csv")