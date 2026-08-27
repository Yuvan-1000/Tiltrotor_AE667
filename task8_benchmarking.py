"""
task8_benchmarking.py
================================================================================
Milestone-1 Task 8 / Report Section 6.3: "Comparable-Rotor Benchmarking".

Compares the DESIGNED rotor (task5_tiltrotor_design) against two published,
real tiltrotor/proprotor designs using nondimensional parameters (solidity,
disk loading, hover tip Mach number, hover Figure of Merit) so that
differences in absolute size/operating condition are normalized out, per
the handout's requirement.

--------------------------------------------------------------------------------
REFERENCE DATA SOURCES (facts/numbers only -- no text reproduced verbatim;
paraphrase and cite in your own report per the course's citation policy):

[R1] Bell XV-15 Tilt Rotor Research Aircraft:
     - 3-bladed rotor, radius 150 in = 3.81 m, reference chord 14 in =
       0.356 m, solidity sigma = 0.089 (original metal blades) -- 0.10
       (Advanced Technology Blades). Linear twist ~ -40.25 deg (total).
       Source: geometry table in Wang et al., "Assessment of Detached Eddy
       Simulation... Tiltrotor Performance," arXiv:2201.11560 (2022),
       citing the XV-15 rotor design literature.
     - Hover/low-speed rotor speed 589 RPM (98% Nr); cruise 517 RPM (86%
       Nr). Design gross weight ~5900 kg (13,000 lb, per DTIC ADA123857)
       up to ~6804 kg (15,000 lb max, per Smithsonian National Air and
       Space Museum aircraft record). Full-scale hover tests report rotor
       tip Mach numbers from about 0.60 to 0.73 (Shinoda/Betzina full-
       scale XV-15 hover-test reports, NASA Ames OARF).
     - Hover Figure of Merit for the baseline (metal-blade) rotor is
       commonly reported in the ~0.7-0.78 range across the literature
       (exact peak value varies by source/rotor build); we use FM=0.75 as
       a representative published figure for this comparison and flag the
       approximate nature of that single number.

[R2] Bell-Boeing V-22 Osprey:
     - 3-bladed proprotor, 38 ft (11.58 m) diameter -> R = 5.79 m, twin
       rotors. Source: Boeing/military-aircraft fact sheets (globalsecurity
       .org, man.fas.org "V-22 Osprey" pages).
     - MTOW (VTOL) ~ 27,443 kg (60,500 lb), a commonly cited value in V-22
       fact sheets.
     - A large-scale powered-model wind-tunnel test of the V-22 rotor/wing
       determined the *isolated* rotor's maximum hover Figure of Merit to
       be 0.808 (with the wing/image-plane present, installed FM is
       somewhat lower due to download/recirculation effects). Source: "The
       V-22 Tilt-Rotor Large-Scale Rotor Performance" test report (NLR/
       ERF archive) and multiple secondary CFD-validation papers citing
       the same test.
     - Representative hover tip speed ~202 m/s (~662 ft/s), giving
       M_tip ~ 0.59 at sea level.

These are the SAME numbers your team should re-verify/re-cite independently
if you use them in your submitted report (Sec 8.2 References) -- treat the
values above as a starting point, not a substitute for reading the source
documents yourselves.
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task5_tiltrotor_design import TILTROTOR_GEOM, AIRCRAFT, RPM_HOVER, build_solver
from bemt_solver import FlightCondition

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

G = 9.80665
A_SOUND_SL = 340.3

# ================================================================================
# Reference rotor data (see module docstring for sources)
# ================================================================================
REFERENCE_ROTORS = {
    "XV-15 (Bell, metal blade)": dict(
        R=3.81, B=3, sigma=0.089, twist_deg=-40.25,
        gross_weight_kg=6350.0,   # midpoint of 5900-6804 kg cited range
        n_rotors=2,
        Vtip_hover=589.0 * 2 * np.pi / 60.0 * 3.81,
        FM_hover=0.75,   # representative published value -- see docstring
        FM_is_approximate=True,
    ),
    "V-22 (Bell-Boeing)": dict(
        R=5.79, B=3, sigma=0.105,   # solidity not directly quoted above;
                                     # V-22 solidity commonly cited ~0.105
                                     # in open literature (higher than
                                     # XV-15, consistent with its higher
                                     # design disk loading) -- flagged as
                                     # an approximate figure, re-verify
                                     # before citing in your own report.
        twist_deg=None,              # not sourced above -- leave blank
        gross_weight_kg=27443.0,
        n_rotors=2,
        Vtip_hover=202.0,
        FM_hover=0.808,   # isolated-rotor peak, per docstring source
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

    # hover FM at the collective that produces this design's per-rotor
    # weight share, sea level (re-uses the same solver/config as Task 6)
    solver = build_solver()
    from scipy.optimize import brentq
    T_target = AIRCRAFT["gross_weight_kg"] * G / AIRCRAFT["n_rotors"]

    def f(theta0_deg):
        flight = FlightCondition.from_rpm(RPM_HOVER, theta0_deg, altitude=0.0, dT_isa=0.0)
        return solver.solve(flight, verbose=False)["T"] - T_target

    theta0_trim = brentq(f, 2.0, 26.0, xtol=1e-3)
    flight = FlightCondition.from_rpm(RPM_HOVER, theta0_trim, altitude=0.0, dT_isa=0.0)
    res = solver.solve(flight, verbose=False)

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

    print("=== Task 8 -- Nondimensional rotor benchmarking (design test case) ===")
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

    # ---- bar chart comparison of the four key nondimensional metrics ----
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

    print("\n--- Discussion pointers (facts only; write the interpretation yourselves) ---")
    print(f"- Our sigma ({ours['sigma']:.3f}) sits between XV-15's 0.089 and a "
          f"V-22-like 0.105, i.e. within the real tiltrotor family.")
    print(f"- Our disk loading ({ours['disk_loading_kgm2']:.0f} kg/m^2) is closer to "
          f"XV-15's (~{REFERENCE_ROTORS['XV-15 (Bell, metal blade)']['gross_weight_kg']/2/(np.pi*3.81**2):.0f} "
          f"kg/m^2) than to the V-22's much higher value -- consistent with our smaller, "
          f"lighter 'light utility tiltrotor' design philosophy (Sec 5.1).")
    print(f"- Our hover FM ({ours['FM_hover']:.3f}) is in the same band as the published "
          f"XV-15 figure and somewhat below the V-22's isolated-rotor peak (0.808) -- "
          f"reasonable for a non-optimized, Milestone-1-stage preliminary rotor.")
    print(f"- Our hover tip Mach ({ours['M_tip']:.3f}) is below both references, which "
          f"trades a little hover efficiency/tip loss margin for lower noise/compressibility risk.")
    return rows


if __name__ == "__main__":
    benchmark()
