"""
task2_edgewise_model.py
================================================================================
Milestone-2 Task 2 / Report Sec 2.1: "Edgewise BEMT formulation" -- a worked
demonstration of edgewise_bemt.py on the actual designed rotor
(task5_tiltrotor_design.TILTROTOR_GEOM / TILTROTOR_AIRFOIL), at a
representative low-speed helicopter-mode forward-flight point.

WHAT THIS SCRIPT DOES
  1. Solves one edgewise operating point and prints the full summary table
     (T, Q, P, CT, CQ, CP, mu, reverse-flow fraction, advancing-tip Mach,
     rigid-hub 1/rev pitch/roll moments) -- everything Sec 2.1 needs to
     narrate the formulation with real numbers, not symbols only.
  2. Plots the blade-element thrust loading dT/dr at four representative
     azimuths (psi = 0/90/180/270 deg) on one axis, to make the once-per-rev
     asymmetry (the whole point of Milestone 2) visually obvious.
  3. Saves the full 2D (r, psi) loading array to CSV for reproducibility.

NOTE: formal verification (mu->0 recovery, discretization sensitivity, the
full azimuthal contour plot) is task3_verification.py, not here -- this
script is the "here is the model, here is what it predicts" demonstration.
================================================================================
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task5_tiltrotor_design import TILTROTOR_GEOM, TILTROTOR_AIRFOIL, RPM_HOVER
from edgewise_bemt import EdgewiseFlightCondition, EdgewiseBEMTSolver

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Representative operating point -------------------------------------
# Low-speed helicopter-mode forward flight: hover RPM, a collective that
# converges cleanly in the M1 axial solver (see dev_notes_bemt_quirks.md /
# task3_verification.py Sec 3.4 for why very low collectives don't), and a
# modest edgewise speed (mu ~ 0.15, well below the mu~0.3-0.5 range where
# retreating-blade stall/reverse-flow would dominate -- appropriate for an
# "early transition" demonstration point, not the edge of the envelope).
COLLECTIVE_DEG = 14.0
V_EDGE_DEMO = 30.0   # [m/s] in-plane speed
V_AXIAL_DEMO = 0.0   # [m/s] level flight, no climb

solver = EdgewiseBEMTSolver(TILTROTOR_GEOM, TILTROTOR_AIRFOIL)
flight = EdgewiseFlightCondition.from_rpm(RPM_HOVER, COLLECTIVE_DEG,
                                          V_axial=V_AXIAL_DEMO, V_edge=V_EDGE_DEMO)


def save_azimuthal_csv(res, path):
    r, psi = res["r"], res["psi"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_m"] + [f"psi_{np.degrees(p):.1f}deg" for p in psi])
        for i in range(len(r)):
            writer.writerow([r[i]] + list(res["dT_blade"][i, :]))


def main():
    res = solver.solve(flight)

    print("=== Task 2 -- Edgewise BEMT formulation demo ===")
    print(f"Rotor: R={TILTROTOR_GEOM.R:.2f} m, B={TILTROTOR_GEOM.B}, sigma={res['solidity']:.4f}")
    print(f"Condition: {flight.rpm:.0f} RPM, theta0={COLLECTIVE_DEG:.1f} deg, "
          f"V_edge={V_EDGE_DEMO:.1f} m/s, V_axial={V_AXIAL_DEMO:.1f} m/s")
    print(f"Advance ratio mu = {res['mu']:.4f}")
    print(f"Disk-average baseline inflow lambda_bar = {res['lam_bar']:.5f}, "
          f"Glauert azimuthal-correction coefficient k = {res['k']:.4f}")
    print()
    print(f"T = {res['T']:.1f} N   Q = {res['Q']:.1f} N.m   P = {res['P']/1000:.1f} kW")
    print(f"CT = {res['CT']:.5f}   CQ = {res['CQ']:.6f}   CP = {res['CP']:.6f}")
    print(f"Reverse-flow fraction of disk = {res['reverse_flow_fraction']*100:.2f}%")
    print(f"Stall fraction = {res['stall_fraction']*100:.2f}%")
    print(f"Advancing-tip Mach number = {res['M_adv_tip']:.3f} (hover tip Mach {res['M_tip']:.3f})")
    print(f"Rigid-hub 1/rev moments (no flapping): "
          f"M_pitch = {res['hub_pitch_moment']:.1f} N.m, M_roll = {res['hub_roll_moment']:.1f} N.m")

    # ---- Sec 2.1 figure: radial loading at 4 representative azimuths ------
    r_over_R = res["r_over_R"]
    psi_deg_targets = [0, 90, 180, 270]
    plt.figure(figsize=(7, 5))
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for psi_t, color in zip(psi_deg_targets, colors):
        j = int(np.argmin(np.abs(np.degrees(res["psi"]) - psi_t)))
        plt.plot(r_over_R, res["dT_blade"][:, j], color=color,
                 label=f"$\\psi$ = {psi_t}$^\\circ$")
    plt.xlabel(r"$r/R$")
    plt.ylabel(r"Blade-element thrust loading $dT/dr$ [N/m] (one blade)")
    plt.title(f"Azimuthal thrust-loading asymmetry, $\\mu$={res['mu']:.3f}, "
              f"$\\theta_0$={COLLECTIVE_DEG:.0f}$^\\circ$")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task2_1_azimuthal_loading_slices.png"), dpi=160)
    plt.close()

    save_azimuthal_csv(res, os.path.join(OUT_DIR, "task2_1_azimuthal_loading.csv"))

    print(f"\nFigure written to ./{FIG_DIR}/task2_1_azimuthal_loading_slices.png")
    print(f"Full (r,psi) loading table written to ./{OUT_DIR}/task2_1_azimuthal_loading.csv")


if __name__ == "__main__":
    main()
