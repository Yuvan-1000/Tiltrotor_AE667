"""
task4_design_variable_study.py
================================================================================
Milestone-1 Task 4 / Report Sections 4.1-4.3: Rotor Design-Variable Study.

Uses the validated baseline rotor (Knight & Hefner geometry/airfoil, same as
task3_validation.py) and independently sweeps:
  4.1  solidity / blade number   (>= 4 values)
  4.2  taper ratio               (>= 4 values)
  4.3  linear twist              (>= 4 values)
holding every other parameter fixed at a single representative hover
operating point (OP_RPM, OP_COLLECTIVE_DEG below).

NOTE ON RPM: with the linear/quadratic airfoil model (no Reynolds or Mach
dependence), CT and CQ are RPM-independent in hover -- only the DIMENSIONAL
T, Q, P and the reported tip Mach number change with RPM. OP_RPM here just
sets a physically reasonable tip speed for reporting dimensional thrust/
power and M_tip; it does not bias the trends being studied.

4.1 solidity is varied two ways:
  (a) via blade number B in {2,3,4,5} (chord fixed -> discrete sigma steps;
      also matches Knight & Hefner's four test rotors if you later digitize
      their B=3,4,5 tables too)
  (b) via chord (B fixed at 2) over a continuous range, for the >=4
      continuous points the handout asks for

4.3 twist: each twist distribution is TRIMMED (root collective adjusted via
root-find) to reproduce the SAME thrust as the untwisted baseline before
comparing power/FM. Comparing twist cases at a FIXED collective instead
would mostly just change tip angle-of-attack and swamp the actual physical
effect of twist (a more uniform inflow -> lower induced power for the same
thrust) -- trimming to constant thrust is the standard way to isolate it.
================================================================================
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from bemt_solver import RotorGeometry, LinearAirfoil, FlightCondition, BEMTSolver

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================================
# Baseline (same rotor/airfoil as Task 3 validation)
# ================================================================================
R_BASE = 0.762
R_ROOT_BASE = 0.125
CHORD_BASE = 0.0508
B_BASE = 2
AIRFOIL = LinearAirfoil(a0=5.75, cd_min=0.0113, eps=1.25, alpha_stall=np.radians(14.0))

OP_RPM = 960.0
OP_COLLECTIVE_DEG = 10.0     # representative point: above the low-theta noise
                              # seen in Task 3, below stall
BASE_FLIGHT = FlightCondition.from_rpm(OP_RPM, collective_deg=OP_COLLECTIVE_DEG,
                                        altitude=0.0, dT_isa=0.0)


def run_point(geom, flight=BASE_FLIGHT, airfoil=AIRFOIL):
    solver = BEMTSolver(geom, airfoil, use_tip_loss=True, use_root_loss=False)
    return solver.solve(flight)


def save_and_plot(x, T, P, eff, xlabel, title_prefix, fname_prefix, extra_label=""):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(x, T, "o-", color="tab:blue")
    axes[0].ticklabel_format(useOffset=False, style="plain", axis="y")
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel("Thrust, T [N]")
    axes[0].set_title(f"{title_prefix}: Thrust"); axes[0].grid(alpha=0.3)

    axes[1].plot(x, np.array(P) / 1000.0, "o-", color="tab:red")
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel("Power, P [kW]")
    axes[1].set_title(f"{title_prefix}: Power"); axes[1].grid(alpha=0.3)

    axes[2].plot(x, eff, "o-", color="tab:green")
    axes[2].set_xlabel(xlabel); axes[2].set_ylabel("Figure of Merit, FM")
    axes[2].set_title(f"{title_prefix}: Efficiency (FM)"); axes[2].grid(alpha=0.3)

    fig.suptitle(f"Task 4 - {title_prefix} sweep{extra_label} "
                 f"(collective={OP_COLLECTIVE_DEG:.0f} deg, {OP_RPM:.0f} RPM, sea level)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{fname_prefix}.png"), dpi=160)
    plt.close(fig)


# ================================================================================
# 4.1(a) - blade-number variation (discrete, matches Knight & Hefner B=2..5)
# ================================================================================
def study_blade_number():
    B_values = [2, 3, 4, 5]
    rows = []
    for B in B_values:
        geom = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=B,
                              chord_root=CHORD_BASE, taper_ratio=1.0, n_stations=80)
        res = run_point(geom)
        rows.append((B, geom.solidity(), res["T"], res["P"], res["FM"],
                     res["stall_fraction"], res["M_tip"]))
        print(f"B={B}  sigma={geom.solidity():.4f}  T={res['T']:.1f} N  "
              f"P={res['P']/1000:.2f} kW  FM={res['FM']:.3f}  "
              f"stall_frac={res['stall_fraction']:.2f}")

    B_arr = [r[0] for r in rows]
    T_arr = [r[2] for r in rows]
    P_arr = [r[3] for r in rows]
    FM_arr = [r[4] for r in rows]
    save_and_plot(B_arr, T_arr, P_arr, FM_arr, "Number of blades, B",
                  "4.1(a) Blade number", "task4_1a_blade_number",
                  extra_label=" (chord fixed -> discrete solidity steps)")

    with open(os.path.join(OUT_DIR, "task4_1a_blade_number.csv"), "w") as f:
        f.write("B,sigma,T_N,P_W,FM,stall_fraction,M_tip\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    return rows


# ================================================================================
# 4.1(b) - continuous solidity variation via chord, B fixed = 2
# ================================================================================
def study_solidity_continuous():
    chord_values = np.linspace(0.03, 0.09, 7)   # -> sigma ~ 0.025 to 0.075
    rows = []
    for c in chord_values:
        geom = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=2,
                              chord_root=c, taper_ratio=1.0, n_stations=80)
        res = run_point(geom)
        rows.append((c, geom.solidity(), res["T"], res["P"], res["FM"],
                     res["stall_fraction"]))

    sigma_arr = [r[1] for r in rows]
    T_arr = [r[2] for r in rows]
    P_arr = [r[3] for r in rows]
    FM_arr = [r[4] for r in rows]
    save_and_plot(sigma_arr, T_arr, P_arr, FM_arr, r"Solidity, $\sigma$",
                  "4.1(b) Solidity (continuous, via chord)",
                  "task4_1b_solidity_continuous")

    with open(os.path.join(OUT_DIR, "task4_1b_solidity_continuous.csv"), "w") as f:
        f.write("chord_m,sigma,T_N,P_W,FM,stall_fraction\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    return rows


# ================================================================================
# 4.2 - taper ratio variation (tip chord / root chord), B=2, chord_root fixed
# ================================================================================
def study_taper():
    taper_values = [0.4, 0.6, 0.8, 0.9, 1.0]
    rows = []
    for tr in taper_values:
        geom = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=B_BASE,
                              chord_root=CHORD_BASE, taper_ratio=tr, n_stations=80)
        res = run_point(geom)
        rows.append((tr, geom.solidity(), res["T"], res["P"], res["FM"],
                     res["stall_fraction"]))
        print(f"taper={tr:.2f}  sigma={geom.solidity():.4f}  T={res['T']:.1f} N  "
              f"P={res['P']/1000:.2f} kW  FM={res['FM']:.3f}")

    x_arr = [r[0] for r in rows]
    T_arr = [r[2] for r in rows]
    P_arr = [r[3] for r in rows]
    FM_arr = [r[4] for r in rows]
    save_and_plot(x_arr, T_arr, P_arr, FM_arr, "Taper ratio (tip chord / root chord)",
                  "4.2 Taper ratio", "task4_2_taper_ratio")

    with open(os.path.join(OUT_DIR, "task4_2_taper_ratio.csv"), "w") as f:
        f.write("taper_ratio,sigma,T_N,P_W,FM,stall_fraction\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    return rows


# ================================================================================
# 4.3 - linear twist variation, trimmed to constant thrust (see module docstring)
# ================================================================================
def find_trim_collective(geom, T_target, rpm=OP_RPM, altitude=0.0, dT_isa=0.0,
                          theta0_scan_deg=np.arange(4.0, 30.0 + 1e-9, 1.0)):
    """
    Finds the collective pitch reproducing T_target, restricted to operating
    points where the BEMT solve actually converges.

    At low collective on a heavily twisted blade, the local blade angle can
    go strongly negative over part of the span. There, the combined blade-
    element/momentum-theory equation can have NO real solution (a known
    limitation of classical BEMT/momentum theory in windmill-brake-like
    conditions -- confirmed here by scanning the residual directly, not
    assumed). Those points are excluded from the trim search rather than
    silently trusted, instead of using a single wide bracket that would
    probe straight through that invalid region.
    """
    pts = []
    for theta0 in theta0_scan_deg:
        flight = FlightCondition.from_rpm(rpm, theta0, altitude=altitude, dT_isa=dT_isa)
        with warnings.catch_warnings():
            # This scan deliberately probes some collectives low enough to be
            # outside the model's valid range (see docstring) -- suppress the
            # solver's per-point warning here since non-convergence is
            # expected and explicitly checked via res["converged"] below,
            # not ignored.
            warnings.simplefilter("ignore", UserWarning)
            res = run_point(geom, flight=flight)
        if res["converged"]:
            pts.append((theta0, res["T"] - T_target))

    for (t_a, f_a), (t_b, f_b) in zip(pts, pts[1:]):
        if abs(f_a) < 1e-6:
            return t_a
        if f_a * f_b < 0:
            def thrust_error(theta0_deg):
                flight = FlightCondition.from_rpm(rpm, theta0_deg, altitude=altitude, dT_isa=dT_isa)
                res = run_point(geom, flight=flight)
                if not res["converged"]:
                    raise RuntimeError(
                        f"BEMT did not converge at theta0={theta0_deg:.2f} deg during "
                        f"trim search -- narrow theta0_scan_deg to stay inside the "
                        f"converged region found by the coarse scan.")
                return res["T"] - T_target
            return brentq(thrust_error, t_a, t_b, xtol=1e-3)

    raise RuntimeError(
        f"No converged bracket found for T_target={T_target:.2f} N in scan range "
        f"{theta0_scan_deg[0]:.1f}-{theta0_scan_deg[-1]:.1f} deg for this geometry; "
        f"widen theta0_scan_deg.")


def study_twist():
    twist_tip_deg_values = [0.0, -4.0, -8.0, -12.0, -16.0]

    geom0 = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=B_BASE,
                           chord_root=CHORD_BASE, taper_ratio=1.0,
                           twist_root=0.0, twist_tip=0.0, n_stations=80)
    T_target = run_point(geom0)["T"]
    print(f"Trimming every twist case to match the baseline thrust T_target = {T_target:.2f} N")

    rows = []
    for tw in twist_tip_deg_values:
        geom = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=B_BASE,
                              chord_root=CHORD_BASE, taper_ratio=1.0,
                              twist_root=0.0, twist_tip=np.radians(tw), n_stations=80)

        theta0_trim = find_trim_collective(geom, T_target)
        flight_trim = FlightCondition.from_rpm(OP_RPM, theta0_trim, altitude=0.0, dT_isa=0.0)
        res = run_point(geom, flight=flight_trim)
        rows.append((tw, theta0_trim, res["T"], res["P"], res["FM"], res["stall_fraction"]))
        print(f"twist_tip={tw:+.1f} deg  (trim theta0={theta0_trim:.2f} deg)  "
              f"T={res['T']:.1f} N  P={res['P']/1000:.3f} kW  FM={res['FM']:.3f}  "
              f"stall_frac={res['stall_fraction']:.2f}  converged={res['converged']}")

    x_arr = [r[0] for r in rows]
    T_arr = [r[2] for r in rows]
    P_arr = [r[3] for r in rows]
    FM_arr = [r[4] for r in rows]
    save_and_plot(x_arr, T_arr, P_arr, FM_arr,
                  "Linear twist, tip relative to root [deg]",
                  "4.3 Linear twist (trimmed to constant thrust)", "task4_3_twist",
                  extra_label=f", T held = {T_target:.1f} N via trimmed collective")

    with open(os.path.join(OUT_DIR, "task4_3_twist.csv"), "w") as f:
        f.write("twist_tip_deg,theta0_trim_deg,T_N,P_W,FM,stall_fraction\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    return rows


# ================================================================================
# 4.1(c) - rotational speed variation (bonus; RPM listed as a variable in the
# handout's Task 4 intro). See the module-level RPM note: CT/CQ don't move
# with RPM in this model, so this sweep mainly shows the DIMENSIONAL T/P/M_tip
# trend at fixed collective, and is a useful check for tip Mach limits.
# ================================================================================
def study_rpm():
    rpm_values = [700, 850, 960, 1100, 1250]
    rows = []
    geom = RotorGeometry(R=R_BASE, r_root=R_ROOT_BASE, B=B_BASE,
                          chord_root=CHORD_BASE, taper_ratio=1.0, n_stations=80)
    for rpm in rpm_values:
        flight = FlightCondition.from_rpm(rpm, collective_deg=OP_COLLECTIVE_DEG,
                                           altitude=0.0, dT_isa=0.0)
        res = run_point(geom, flight=flight)
        rows.append((rpm, res["T"], res["P"], res["FM"], res["M_tip"]))
        print(f"RPM={rpm}  T={res['T']:.1f} N  P={res['P']/1000:.2f} kW  "
              f"FM={res['FM']:.3f}  M_tip={res['M_tip']:.3f}")

    x_arr = [r[0] for r in rows]
    T_arr = [r[1] for r in rows]
    P_arr = [r[2] for r in rows]
    FM_arr = [r[3] for r in rows]
    save_and_plot(x_arr, T_arr, P_arr, FM_arr, "Rotor speed [RPM]",
                  "4.1(c) Rotational speed (bonus)", "task4_1c_rpm_bonus")

    with open(os.path.join(OUT_DIR, "task4_1c_rpm_bonus.csv"), "w") as f:
        f.write("rpm,T_N,P_W,FM,M_tip\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
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
    print(f"\nAll figures in ./{FIG_DIR}/task4_*.png ; all tables in ./{OUT_DIR}/task4_*.csv")