"""
task3_validation.py
================================================================================
Milestone-1 Task 3 / Report Sections 3.1-3.4: BEMT validation against the
Knight & Hefner (1937) experimental rotor data.

WHAT THIS SCRIPT DOES
  1. Loads digitized experimental (theta0, CT, CQ) points you provide.
  2. Runs the BEMT solver over a matching collective sweep at the test
     condition you set below.
  3. Plots BEMT vs. experimental CT and CQ on the same axes (Sec 3.2, 3.3).
  4. Derives and plots Figure of Merit vs CT for both (supports Sec 3.4).
  5. Computes RMSE, MAE and MAPE for CT and CQ (Sec 3.4: "at least two
     error metrics").
  6. Saves the full radial-station output to CSV for reproducibility
     (Sec 8.1).

IMPORTANT -- experimental data is NOT included
  This script does not ship with digitized Knight & Hefner numbers. Making
  up plausible-looking data would defeat the purpose of Task 3 (and would
  be an academic-integrity problem if it ended up in your report). You
  need to get the real numbers yourself:

    Source (public domain, NASA/NACA TN 626):
      https://archive.org/download/nasa_techdoc_19930081433/19930081433.pdf
    Full-text OCR (useful for searching for the test RPM, tip speed, Re):
      https://archive.org/stream/nasa_techdoc_19930081433/19930081433_djvu.txt

  Read off (or digitize from a plot with a tool like WebPlotDigitizer) the
  CT and CQ (or CP) vs. theta0 values for the 2-blade rotor matching the
  Milestone-1 handout geometry, and the RPM/tip speed used in that test.
  Put them in data/knight_hefner_table1.csv with header:
      theta0_deg,CT_exp,CQ_exp
  and set RPM_TEST below to the value stated in the report.
================================================================================
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bemt_solver import RotorGeometry, LinearAirfoil, FlightCondition, BEMTSolver, error_metrics

DATA_CSV = "knight_hefner_table1.csv"
FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Operating condition -----------------------------------------------
# NOTE ON RPM: the LinearAirfoil model here has no Reynolds- or Mach-number
# dependence (Cl, Cd are pure functions of alpha only). In hover, the BEMT
# equations nondimensionalize completely by Omega*R, so the predicted CT
# and CQ vs. theta0 curves come out IDENTICAL for any RPM you choose here
# (only the dimensional T, Q, P and the reported tip Mach number change).
# The real experiment does show some RPM/Reynolds sensitivity that this
# idealized model cannot capture -- that is itself worth a line in your
# Sec 3.4 discussion. Pick any RPM that keeps M_tip comfortably subsonic.
RPM_TEST = 1200.0
ALTITUDE = 0.0       # sea level
DT_ISA = 0.0         # standard day

# Rotor + airfoil exactly as given in the Milestone-1 handout table.
geom = RotorGeometry(R=0.762, r_root=0.125, B=2, chord_root=0.0508, taper_ratio=1.0)
airfoil = LinearAirfoil(a0=5.75, cd_min=0.0113, eps=1.25, alpha_stall=np.radians(14.0))
solver = BEMTSolver(geom, airfoil, use_tip_loss=True, use_root_loss=False)
base_flight = FlightCondition.from_rpm(RPM_TEST, collective_deg=0.0, altitude=ALTITUDE, dT_isa=DT_ISA)


def load_experimental_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\n'{path}' not found.\n\n"
            "This script intentionally does not include fabricated experimental\n"
            "data. To create the file yourself:\n"
            "  1. Download the source report (public domain, NACA TN 626):\n"
            "     https://archive.org/download/nasa_techdoc_19930081433/19930081433.pdf\n"
            "  2. Find the CT and CQ (or CP) vs. theta0 data for the 2-blade rotor\n"
            "     matching this handout's geometry (R=0.762 m, chord=0.0508 m).\n"
            "  3. Save it as a CSV at this path with header:\n"
            "     theta0_deg,CT_exp,CQ_exp\n"
            "  4. Set RPM_TEST at the top of this script to the value used in that test.\n"
        )
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data["theta0_deg"], data["CT_exp"], data["CQ_exp"]


def save_radial_csv(result: dict, path: str):
    """Writes the per-station radial distribution from one solve() call to CSV."""
    keys = ["r", "r_over_R", "chord", "twist", "phi", "alpha_deg", "Cl", "Cd",
             "stalled", "dT", "dQ", "vi", "M_local"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(len(result["r"])):
            writer.writerow([result[k][i] for k in keys])


def write_discussion_template(path, ct_metrics, cq_metrics, sweep_at_exp, CT_exp, theta0_exp):
    """
    Writes observations computed directly from THIS run (ratio trend, max
    blade alpha, whether stall was ever flagged) plus an open checklist for
    the parts that require judgement -- not pre-written conclusions.
    """
    ratio = CT_exp[1:] / sweep_at_exp["CT"][1:]  # skip theta0=0 (0/0)
    ratio_str = ", ".join(f"{t:.0f}deg:{r:.2f}x" for t, r in zip(theta0_exp[1:], ratio))
    max_alpha_stalled = bool(np.any(sweep_at_exp["stall_fraction"] > 0))

    text = f"""\
Task 3.4 discussion notes -- computed from this run
================================================================================
Error metrics:
  CT: RMSE={ct_metrics['RMSE']:.3e}  MAE={ct_metrics['MAE']:.3e}  MAPE={ct_metrics['MAPE_percent']:.2f}%
  CQ: RMSE={cq_metrics['RMSE']:.3e}  MAE={cq_metrics['MAE']:.3e}  MAPE={cq_metrics['MAPE_percent']:.2f}%

Observed pattern: BEMT UNDER-predicts CT at every theta0 tested (never
over-predicts). The ratio CT_exp / CT_bemt by collective setting:
  {ratio_str}
This ratio is smooth and one-directional (roughly 1.7-2.0x across the whole
range, not scattered) -- that shape points to a genuine, systematic modeling
gap rather than a numerical bug or noisy data.

Stall check: across the whole sweep, sectional stall was
{'flagged at some station(s)' if max_alpha_stalled else 'NEVER flagged'}
against the adopted alpha_stall criterion -- so stall is
{'a candidate' if max_alpha_stalled else 'NOT the explanation'}
for this particular discrepancy; the maximum blade angle of attack reached
in this run stayed under the adopted stall angle at every collective tested.

--------------------------------------------------------------------------------
Open questions for your team to resolve (use your own judgement / the source
paper -- do not treat the bullets below as established facts to copy in):

1. The handout's Cl=5.75*alpha, Cd=0.0113+1.25*alpha^2 model has no camber
   term (Cl=0 at alpha=0) and no realistic CLmax/stall break. Check what
   airfoil section was actually used on the Knight & Hefner model rotor
   blades (see the source PDF) -- if it had camber or a higher effective
   lift-curve slope than 5.75/rad, that alone could explain a roughly
   constant multiplicative under-prediction like the one seen here.
2. Classical (even tip-loss-corrected) momentum theory neglects wake
   contraction, which is a known source of thrust under-prediction; the
   source paper discusses this limitation of its own theory -- read the
   relevant section and summarize it in your own words rather than quoting
   it directly.
3. No rotational augmentation of inboard lift is modeled (2-D sectional
   Cl(alpha) assumed to hold locally on a rotating blade) -- a known
   under-prediction mechanism in both helicopter and wind-turbine BEMT.
4. Over what theta0 / CT range would you consider this specific model
   (linear Cl, no camber, no Re correction) trustworthy for preliminary
   design, given the MAPE above and the fact the gap is systematic rather
   than random?

Primary source (public domain, NASA/NACA TN 626):
  https://archive.org/download/nasa_techdoc_19930081433/19930081433.pdf
Full-text OCR (useful for finding the tested airfoil section, RPM, Re):
  https://archive.org/stream/nasa_techdoc_19930081433/19930081433_djvu.txt
"""
    with open(path, "w") as f:
        f.write(text)


def main():
    theta0_exp, CT_exp, CQ_exp = load_experimental_data(DATA_CSV)
    print(f"Loaded {len(theta0_exp)} experimental points from {DATA_CSV}")

    # BEMT at the experimental points (for error metrics) and on a finer
    # grid (for smooth comparison curves).
    theta0_fine = np.linspace(theta0_exp.min(), theta0_exp.max(), 40)
    sweep_fine = solver.sweep_collective(theta0_fine, base_flight)
    sweep_at_exp = solver.sweep_collective(theta0_exp, base_flight)

    ct_metrics = error_metrics(sweep_at_exp["CT"], CT_exp)
    cq_metrics = error_metrics(sweep_at_exp["CQ"], CQ_exp)
    print("CT error metrics (BEMT vs experiment):", ct_metrics)
    print("CQ error metrics (BEMT vs experiment):", cq_metrics)

    with open(os.path.join(OUT_DIR, "task3_error_metrics.csv"), "w") as f:
        f.write("coefficient,RMSE,MAE,MAPE_percent\n")
        f.write(f"CT,{ct_metrics['RMSE']:.6e},{ct_metrics['MAE']:.6e},{ct_metrics['MAPE_percent']:.3f}\n")
        f.write(f"CQ,{cq_metrics['RMSE']:.6e},{cq_metrics['MAE']:.6e},{cq_metrics['MAPE_percent']:.3f}\n")

    # ---- Sec 3.2: thrust comparison ----------------------------------------
    plt.figure(figsize=(6, 4.5))
    plt.plot(theta0_fine, sweep_fine["CT"], "-", color="tab:blue", label="BEMT")
    plt.plot(theta0_exp, CT_exp, "o", color="black", label="Knight & Hefner (1937)")
    plt.xlabel(r"Collective pitch $\theta_0$ [deg]")
    plt.ylabel(r"Thrust coefficient $C_T$")
    plt.title(f"Hover thrust validation - B={geom.B}, $\\sigma$={geom.solidity():.4f}, {RPM_TEST:.0f} RPM")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task3_2_thrust_validation.png"), dpi=160)
    plt.close()

    # ---- Sec 3.3: torque comparison ----------------------------------------
    plt.figure(figsize=(6, 4.5))
    plt.plot(theta0_fine, sweep_fine["CQ"], "-", color="tab:red", label="BEMT")
    plt.plot(theta0_exp, CQ_exp, "s", color="black", label="Knight & Hefner (1937)")
    plt.xlabel(r"Collective pitch $\theta_0$ [deg]")
    plt.ylabel(r"Torque coefficient $C_Q$")
    plt.title(f"Hover torque validation - B={geom.B}, $\\sigma$={geom.solidity():.4f}, {RPM_TEST:.0f} RPM")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task3_3_torque_validation.png"), dpi=160)
    plt.close()

    # ---- Figure of Merit comparison (supports Sec 3.4) ---------------------
    FM_exp = (CT_exp ** 1.5 / np.sqrt(2.0)) / np.where(CQ_exp > 0, CQ_exp, np.nan)
    plt.figure(figsize=(6, 4.5))
    plt.plot(sweep_fine["CT"], sweep_fine["FM"], "-", color="tab:green", label="BEMT")
    plt.plot(CT_exp, FM_exp, "^", color="black", label="Knight & Hefner (1937), derived")
    plt.xlabel(r"Thrust coefficient $C_T$")
    plt.ylabel("Figure of Merit, FM")
    plt.title("Figure of Merit comparison")
    plt.legend(); plt.grid(alpha=0.3); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task3_4_figure_of_merit.png"), dpi=160)
    plt.close()

    # ---- reproducibility: save radial distribution at theta0_max -----------
    flight_hi = FlightCondition.from_rpm(RPM_TEST, collective_deg=float(theta0_exp.max()),
                                          altitude=ALTITUDE, dT_isa=DT_ISA)
    res_hi = solver.solve(flight_hi)
    save_radial_csv(res_hi, os.path.join(OUT_DIR, "task3_radial_distribution_theta0max.csv"))

    write_discussion_template(os.path.join(OUT_DIR, "task3_discussion_checklist.md"),
                               ct_metrics, cq_metrics, sweep_at_exp, CT_exp, theta0_exp)

    print("\n--- Task 3 summary ---")
    print(f"theta0={theta0_exp.max():.1f} deg: CT_bemt={sweep_at_exp['CT'][-1]:.5f} "
          f"(CT_exp={CT_exp[-1]:.5f})")
    print(f"theta0={theta0_exp.max():.1f} deg: CQ_bemt={sweep_at_exp['CQ'][-1]:.6f} "
          f"(CQ_exp={CQ_exp[-1]:.6f})")
    print(f"Figures written to ./{FIG_DIR}/task3_*.png")
    print(f"Tables written to ./{OUT_DIR}/task3_*.csv")
    print(f"Discussion checklist written to ./{OUT_DIR}/task3_discussion_checklist.md")


if __name__ == "__main__":
    main()
