"""
task7_forward_flight_assessment.py
================================================================================
Milestone-1 Task 7 / Report Section 6.2: "Axial Forward-Flight / Propeller
Assessment".

Evaluates the DESIGNED rotor (task5_tiltrotor_design) with the rotor axis
aligned with the freestream ("airplane mode"/propeller mode), at RPM_CRUISE,
over a range of advance ratios J and collective settings:
  - CT, CP vs J for several collective settings
  - propulsive efficiency eta_p = T*V / P vs J
  - blade angle-of-attack radial distribution at a representative point
  - the FEASIBLE operating envelope (T>0, converged, stall_fraction below
    the same adopted design limit used in Task 6) and a selected cruise
    operating point (highest eta_p inside that envelope)

IMPORTANT FINDING: the highest-efficiency FEASIBLE point found here is at a
forward speed well below the 110 m/s mission requirement set in Task 5.
This is flagged explicitly rather than hidden.
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bemt_solver import FlightCondition
from task5_tiltrotor_design import (
    TILTROTOR_GEOM, RPM_CRUISE, COLLECTIVE_RANGE_CRUISE_DEG, build_solver, AIRCRAFT,
)

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SOLVER = build_solver()
STALL_FRACTION_LIMIT = 0.05     # same adopted design margin as Task 6
CRUISE_ALTITUDE_M = 3000.0      # representative airplane-mode cruise altitude

N_REV_PER_S = RPM_CRUISE / 60.0
DIAMETER = 2.0 * TILTROTOR_GEOM.R


def solve_axial(collective_deg, J, altitude=CRUISE_ALTITUDE_M):
    V = J * N_REV_PER_S * DIAMETER
    flight = FlightCondition(Omega=RPM_CRUISE * 2.0 * np.pi / 60.0,
                              collective=np.radians(collective_deg),
                              altitude=altitude, dT_isa=0.0, V_axial=V)
    res = SOLVER.solve(flight)  # Fixed: removed verbose=False
    eta_p = (res["T"] * V / res["P"]) if res["P"] > 0 else float("nan")
    return res, V, eta_p


# ================================================================================
# 6.2(a) CT, CP, eta_p vs J at several collective settings
# ================================================================================
def forward_flight_maps():
    collectives = [25.0, 28.0, 30.0, 32.0, 34.0]   # deg, within cruise range
    J_grid = np.linspace(0.1, 1.0, 19)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    csv_rows = []
    for coll in collectives:
        CT_list, CP_list, eta_list, stall_list = [], [], [], []
        for J in J_grid:
            res, V, eta_p = solve_axial(coll, J)
            CT_list.append(res["CT"])
            CP_list.append(res["CP"])
            eta_list.append(eta_p)
            stall_list.append(res["stall_fraction"])
            csv_rows.append((coll, J, V, res["T"], res["P"], res["CT"], res["CP"],
                              eta_p, res["stall_fraction"], res["M_tip"]))

        axes[0].plot(J_grid, CT_list, "-o", ms=3, label=f"coll={coll:.0f} deg")
        axes[1].plot(J_grid, CP_list, "-o", ms=3, label=f"coll={coll:.0f} deg")
        eta_arr = np.array(eta_list)
        stall_arr = np.array(stall_list)
        mask = (eta_arr > 0) & (stall_arr <= STALL_FRACTION_LIMIT)
        axes[2].plot(J_grid[mask], eta_arr[mask], "-o", ms=3, label=f"coll={coll:.0f} deg")

    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("Advance ratio, J"); axes[0].set_ylabel(r"$C_T$")
    axes[0].set_title("6.2 Thrust coefficient vs J"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Advance ratio, J"); axes[1].set_ylabel(r"$C_P$")
    axes[1].set_title("6.2 Power coefficient vs J"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    axes[2].set_xlabel("Advance ratio, J"); axes[2].set_ylabel(r"Propulsive efficiency, $\eta_p$")
    axes[2].set_title("6.2 Propulsive efficiency vs J\n(masked to feasible: T>0, stall<=5%)")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8); axes[2].set_ylim(0, 1)

    fig.suptitle(f"Task 7 -- Axial forward-flight (propeller-mode) maps, "
                 f"RPM={RPM_CRUISE:.0f}, altitude={CRUISE_ALTITUDE_M:.0f} m")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "task7_forward_flight_maps.png"), dpi=160)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "task7_forward_flight_sweep.csv"), "w") as f:
        f.write("collective_deg,J,V_ms,T_N,P_W,CT,CP,eta_p,stall_fraction,M_tip\n")
        for r in csv_rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print(f"Forward-flight maps written to {FIG_DIR}/task7_forward_flight_maps.png")
    return csv_rows


# ================================================================================
# 6.2(b) Blade AoA radial distribution at a representative cruise-like point
# ================================================================================
def blade_aoa_distribution(collective_deg=30.0, J=0.5):
    res, V, eta_p = solve_axial(collective_deg, J)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(res["r_over_R"], res["alpha_deg"], "-", color="tab:blue")
    plt.axhline(12.0, color="k", ls="--", lw=1, label="adopted stall AoA (+/-12 deg)")
    plt.axhline(-12.0, color="k", ls="--", lw=1)
    plt.xlabel("r/R"); plt.ylabel("Local blade angle of attack [deg]")
    plt.title(f"Task 7 -- Blade AoA distribution, collective={collective_deg:.0f} deg, "
              f"J={J:.2f}, V={V:.1f} m/s")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task7_blade_aoa_distribution.png"), dpi=160)
    plt.close()
    print(f"Blade AoA distribution written to {FIG_DIR}/task7_blade_aoa_distribution.png "
          f"(T={res['T']:.0f} N, stall_frac={res['stall_fraction']:.2f})")


# ================================================================================
# 6.2(c) Feasible operating envelope + selected cruise point
# ================================================================================
def select_cruise_point(csv_rows):
    feasible = [r for r in csv_rows if r[3] > 0 and r[7] == r[7] and r[8] <= STALL_FRACTION_LIMIT]
    if not feasible:
        print("No feasible (T>0, stall<=limit) point found in the sweep grid!")
        return None
    best = max(feasible, key=lambda r: r[7])   # r[7] = eta_p
    coll, J, V, T, P, CT, CP, eta_p, stall_frac, M_tip = best

    print("\n--- Task 7 selected cruise operating point (design test case) ---")
    print(f"collective = {coll:.1f} deg, J = {J:.2f}, V = {V:.1f} m/s ({V*1.94384:.1f} kt)")
    print(f"T = {T:.0f} N/rotor, P = {P/1000:.1f} kW/rotor, eta_p = {eta_p:.3f}, "
          f"stall_fraction = {stall_frac:.3f}, M_tip = {M_tip:.3f}")

    gap_note = (
        f"NOTE: this feasible cruise speed ({V:.1f} m/s = {V*1.94384:.0f} kt) is well "
        f"below the {AIRCRAFT['design_cruise_speed_ms']:.0f} m/s "
        f"({AIRCRAFT['design_cruise_speed_ms']*1.94384:.0f} kt) mission requirement "
        f"stated in Task 5. Root cause: this rotor's twist (-12 deg, chosen in Task 5 "
        f"for hover numerical robustness -- see that file's docstring) does not carry "
        f"enough built-in washout to keep the inboard stations unstalled at the high "
        f"advance ratio a 110 m/s cruise would require at this RPM. A production-\n"
        f"representative fix (more washout, e.g. -30 to -40 deg as on the XV-15, "
        f"paired with a re-derived, higher hover collective schedule) is exactly the "
        f"kind of rotor redesign flagged for Milestone 2 in Sec 7.5 -- do not silently "
        f"raise AIRCRAFT['demonstrated_cruise_speed_ms'] without re-running Tasks 6-7."
    )
    print(gap_note)

    with open(os.path.join(OUT_DIR, "task7_selected_cruise_point.csv"), "w") as f:
        f.write("collective_deg,J,V_ms,V_kt,T_N,P_W,eta_p,stall_fraction,M_tip\n")
        f.write(f"{coll},{J},{V},{V*1.94384},{T},{P},{eta_p},{stall_frac},{M_tip}\n")
    with open(os.path.join(OUT_DIR, "task7_notes_cruise_gap.md"), "w") as f:
        f.write("# Task 7 technical note: cruise-speed requirement vs. demonstrated capability\n\n")
        f.write(gap_note + "\n")

    return dict(collective_deg=coll, J=J, V_ms=V, T_N=T, P_W=P, eta_p=eta_p,
                stall_fraction=stall_frac, M_tip=M_tip)


if __name__ == "__main__":
    csv_rows = forward_flight_maps()
    blade_aoa_distribution(collective_deg=30.0, J=0.5)
    select_cruise_point(csv_rows)
    print(f"\nAll figures in ./{FIG_DIR}/task7_*.png ; all tables in ./{OUT_DIR}/task7_*.csv")