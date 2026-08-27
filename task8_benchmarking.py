"""
task8_benchmarking.py
================================================================================
Milestone-1 Task 8 / Report Section 6.3: "Comparable-Rotor Benchmarking".

Compares the DESIGNED rotor (task5_tiltrotor_design) against two published,
real tiltrotor/proprotor designs using nondimensional parameters (solidity,
disk loading, hover tip Mach number, hover Figure of Merit) so that
differences in absolute size/operating condition are normalized out, per
the handout's requirement.
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task5_tiltrotor_design import TILTROTOR_GEOM, AIRCRAFT, RPM_HOVER, build_solver
from bemt_solver import FlightCondition
from scipy.optimize import brentq

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

G = 9.80665
A_SOUND_SL = 340.3

# ================================================================================
# Reference rotor data
# ================================================================================
REFERENCE_ROTORS = {
    "XV-15 (Bell, metal blade)": dict(
        R=3.81, B=3, sigma=0.089, twist_deg=-40.25,
        gross_weight_kg=6350.0,   # midpoint of 5900-6804 kg cited range
        n_rotors=2,
        Vtip_hover=589.0 * 2 * np.pi / 60.0 * 3.81,
        FM_hover=0.75,   # representative published value
        FM_is_approximate=True,
    ),
    "V-22 (Bell-Boeing)": dict(
        R=5.79, B=3, sigma=0.105,   # commonly cited in open literature
        twist_deg=None,
        gross_weight_kg=27443.0,
        n_rotors=2,
        Vtip_hover=202.0,
        FM_hover=0.808,   # isolated-rotor peak
        FM_is_approximate=False,
        sigma_is_approximate=True,
    ),
}


def our_rotor_metrics():
    geom = TILTROTOR_GEOM
    sigma = geom.solidity()
    A_disk = np.pi * geom.R ** 2
    disk_loading_kgm2 = (AIRCRAFT["gross_weight_kg"] / AIRCRAFT["n_rotors"]) / A_disk
    Vtip = RPM_HOVER * 2.0 * np.pi / 60.0 * geom.R
    M_tip = Vtip / A_SOUND_SL

    # hover FM at the collective that produces this design's per-rotor weight share
    solver = build_solver()
    T_target = AIRCRAFT["gross_weight_kg"] * G / AIRCRAFT["n_rotors"]

    def f(theta0_deg):
        flight = FlightCondition.from_rpm(RPM_HOVER, theta0_deg, altitude=0.0, dT_isa=0.0)
        return solver.solve(flight)["T"] - T_target  # Fixed: removed verbose=False

    theta0_trim = brentq(f, 2.0, 26.0, xtol=1e-3)
    flight = FlightCondition.from_rpm(RPM_HOVER, theta0_trim, altitude=0.0, dT_isa=0.0)
    res = solver.solve(flight)  # Fixed: removed verbose=False

    return dict(R=geom.R, B=geom.B, sigma=sigma,
                twist_deg=np.degrees(geom.twist_tip - geom.twist_root),
                gross_weight_kg=AIRCRAFT["gross_weight_kg"], n_rotors=AIRCRAFT["n_rotors"],
                disk_loading_kgm2=disk_loading_kgm2, Vtip_hover=Vtip, M_tip=M_tip,
                FM_hover=res["FM"])


def benchmark():
    ours = our_rotor_metrics()
    rows = [("This design", ours)]
    for name, ref in REFERENCE_ROTORS.items():
        A_disk = np.pi * ref["R"] ** 2
        disk_loading_kgm2 = (ref["gross_weight_kg"] / ref["n_rotors"]) / A_disk
        M_tip = ref["Vtip_hover"] / A_SOUND_SL
        rows.append((name, dict(R=ref["R"], B=ref["B"], sigma=ref["sigma"],
                                 twist_deg=ref["twist_deg"],
                                 gross_weight_kg=ref["gross_weight_kg"],
                                 n_rotors=ref["n_rotors"],
                                 disk_loading_kgm2=disk_loading_kgm2,
                                 Vtip_hover=ref["Vtip_hover"], M_tip=M_tip,
                                 FM_hover=ref["FM_hover"])))

    print("=== Task 8 -- Nondimensional rotor benchmarking ===")
    header = f"{'Rotor':<28}{'R [m]':>8}{'B':>4}{'sigma':>9}{'DL [kg/m2]':>13}{'M_tip':>8}{'FM':>7}"
    print(header)
    csv_lines = ["Rotor,R_m,B,sigma,disk_loading_kg_m2,M_tip,FM_hover"]
    for name, d in rows:
        print(f"{name:<28}{d['R']:>8.2f}{d['B']:>4}{d['sigma']:>9.4f}"
              f"{d['disk_loading_kgm2']:>13.1f}{d['M_tip']:>8.3f}{d['FM_hover']:>7.3f}")
        csv_lines.append(f"{name},{d['R']:.3f},{d['B']},{d['sigma']:.4f},"
                          f"{d['disk_loading_kgm2']:.2f},{d['M_tip']:.4f},{d['FM_hover']:.3f}")

    with open(os.path.join(OUT_DIR, "task8_benchmarking_table.csv"), "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Bar chart comparison
    names = [r[0] for r in rows]
    sigmas = [r[1]["sigma"] for r in rows]
    dls = [r[1]["disk_loading_kgm2"] for r in rows]
    mtips = [r[1]["M_tip"] for r in rows]
    fms = [r[1]["FM_hover"] for r in rows]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for ax, vals, title, ylabel in zip(
            axes, [sigmas, dls, mtips, fms],
            ["Solidity", "Disk loading", "Hover tip Mach", "Hover Figure of Merit"],
            [r"$\sigma$", "kg/m^2", r"$M_{tip}$", "FM"]):
        ax.bar(names, vals, color=colors)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Task 8 -- Nondimensional benchmarking vs. published tiltrotor/proprotor data")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "task8_benchmarking_bars.png"), dpi=160)
    plt.close(fig)

    print(f"\nBenchmarking table written to {OUT_DIR}/task8_benchmarking_table.csv")
    print(f"Benchmarking chart written to {FIG_DIR}/task8_benchmarking_bars.png")

    return rows


if __name__ == "__main__":
    benchmark()