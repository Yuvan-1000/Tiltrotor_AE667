"""
task3_verification.py
================================================================================
Milestone-2 Task 3 / Report Sec 3: Verification of the Edgewise BEMT Model.

WHAT THIS SCRIPT DOES
  3.1  mu -> 0 recovery: Runs edgewise_bemt.py at V_edge = 0 across a
       collective sweep and compares T, Q, CT, CQ against the ALREADY
       VALIDATED Milestone-1 axial BEMTSolver at identical operating points.
  3.2  Azimuthal loading contour: A full (r, psi) heatmap of blade-element
       thrust loading at a representative forward-flight point, checking for
       physically realistic azimuthal asymmetry and reverse flow.
  3.3  Discretization sensitivity: Sweeps n_stations (radial) and n_azimuth
       (azimuthal) independently to verify CT grid convergence.
================================================================================
"""

from __future__ import annotations
import os
import csv
import copy
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Primary imports
from bemt_solver import FlightCondition, BEMTSolver, RotorGeometry, AirfoilModel
from edgewise_bemt import EdgewiseFlightCondition, EdgewiseBEMTSolver

# Attempt optional imports from task5_tiltrotor_design and bemt_solver
try:
    from task5_tiltrotor_design import TILTROTOR_GEOM, TILTROTOR_AIRFOIL, RPM_HOVER
except ImportError:
    class ConcreteAirfoil(AirfoilModel):
        """Fallback thin-airfoil implementation with linear lift slope."""
        def get_cl_cd(self, alpha: np.ndarray):
            alpha_arr = np.clip(np.asarray(alpha), -np.pi/2, np.pi/2)
            alpha_stall = np.radians(14.0)
            stalled = np.abs(alpha_arr) > alpha_stall
            cl = np.where(
                ~stalled,
                2.0 * np.pi * alpha_arr,
                2.0 * np.pi * alpha_stall * np.sign(alpha_arr) * np.cos((np.abs(alpha_arr) - alpha_stall) / (np.pi/2 - alpha_stall) * (np.pi/2))
            )
            cd = 0.01 + 0.05 * (alpha_arr ** 2)
            return cl, cd, stalled

    TILTROTOR_GEOM = RotorGeometry()
    TILTROTOR_AIRFOIL = ConcreteAirfoil()
    RPM_HOVER = 450.0

try:
    from bemt_solver import error_metrics
except ImportError:
    def error_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        err = np.array(y_pred) - np.array(y_true)
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        denom = np.where(np.abs(y_true) == 0, 1e-12, y_true)
        mape = float(np.mean(np.abs(err / denom))) * 100.0
        return {"RMSE": rmse, "MAE": mae, "MAPE_percent": mape}


FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

axial_solver = BEMTSolver(TILTROTOR_GEOM, TILTROTOR_AIRFOIL)
edge_solver = EdgewiseBEMTSolver(TILTROTOR_GEOM, TILTROTOR_AIRFOIL)

COLLECTIVE_SWEEP_DEG = np.arange(12.0, 22.0 + 1e-6, 1.0)


# ============================================================================
# 3.1 -- mu -> 0 recovery
# ============================================================================

def verify_mu_zero_recovery() -> bool:
    print("=== Task 3.1 -- mu -> 0 recovery check ===")
    axial_T, axial_Q, axial_CT, axial_CQ = [], [], [], []
    edge_T, edge_Q, edge_CT, edge_CQ = [], [], [], []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for th0 in COLLECTIVE_SWEEP_DEG:
            # Milestone 1 Axial Solve
            try:
                axf = FlightCondition.from_rpm(RPM_HOVER, th0)
            except AttributeError:
                omega = RPM_HOVER * 2.0 * np.pi / 60.0
                axf = FlightCondition(Omega=omega, collective=np.radians(th0))

            ax_res = axial_solver.solve(axf)
            axial_T.append(ax_res["T"])
            axial_Q.append(ax_res["Q"])
            axial_CT.append(ax_res.get("CT", ax_res.get("C_T", 0.0)))
            axial_CQ.append(ax_res.get("CQ", ax_res.get("C_Q", 0.0)))

            # Milestone 2 Edgewise Solve at mu = 0
            edf = EdgewiseFlightCondition.from_rpm(RPM_HOVER, th0, V_edge=0.0)
            ed_res = edge_solver.solve(edf)
            edge_T.append(ed_res["T"])
            edge_Q.append(ed_res["Q"])
            edge_CT.append(ed_res.get("CT", ed_res.get("C_T", 0.0)))
            edge_CQ.append(ed_res.get("CQ", ed_res.get("C_Q", 0.0)))

    a_T, e_T = np.array(axial_T), np.array(edge_T)
    a_Q, e_Q = np.array(axial_Q), np.array(edge_Q)
    a_CT, e_CT = np.array(axial_CT), np.array(edge_CT)
    a_CQ, e_CQ = np.array(axial_CQ), np.array(edge_CQ)

    T_metrics = error_metrics(e_T, a_T)
    Q_metrics = error_metrics(e_Q, a_Q)
    max_abs_CT_diff = float(np.max(np.abs(e_CT - a_CT)))
    max_abs_CQ_diff = float(np.max(np.abs(e_CQ - a_CQ)))

    print(f"T:  RMSE={T_metrics['RMSE']:.3e} N   MAE={T_metrics['MAE']:.3e} N   "
          f"MAPE={T_metrics['MAPE_percent']:.4f}%")
    print(f"Q:  RMSE={Q_metrics['RMSE']:.3e} N.m MAE={Q_metrics['MAE']:.3e} N.m "
          f"MAPE={Q_metrics['MAPE_percent']:.4f}%")
    print(f"max|CT_edge - CT_axial| = {max_abs_CT_diff:.3e}")
    print(f"max|CQ_edge - CQ_axial| = {max_abs_CQ_diff:.3e}")

    passed = max_abs_CT_diff < 1e-6 and max_abs_CQ_diff < 1e-6
    print(f"-> recovery check {'PASSED' if passed else 'FAILED'} "
          f"(tolerance 1e-6 on CT/CQ)")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "task3_1_mu_zero_recovery.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta0_deg", "T_axial", "T_edgewise", "Q_axial", "Q_edgewise",
                          "CT_axial", "CT_edgewise", "CQ_axial", "CQ_edgewise"])
        for i, th0 in enumerate(COLLECTIVE_SWEEP_DEG):
            writer.writerow([th0, a_T[i], e_T[i], a_Q[i], e_Q[i], a_CT[i], e_CT[i], a_CQ[i], e_CQ[i]])

    # Generate Plot
    plt.figure(figsize=(6, 4.5))
    plt.plot(COLLECTIVE_SWEEP_DEG, a_CT, "o-", color="black", label="Milestone-1 axial BEMT")
    plt.plot(COLLECTIVE_SWEEP_DEG, e_CT, "x--", color="tab:orange", label=r"Edgewise BEMT, $\mu$=0")
    plt.xlabel(r"Collective $\theta_0$ [deg]")
    plt.ylabel(r"$C_T$")
    plt.title(r"$\mu \to 0$ recovery: edgewise engine vs. Milestone-1 axial solver")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "task3_1_mu_zero_recovery.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return passed


# ============================================================================
# 3.2 -- azimuthal loading contour
# ============================================================================

def azimuthal_loading_contour(theta0_deg: float = 14.0, V_edge: float = 35.0, V_axial: float = 0.0):
    print("\n=== Task 3.2 -- azimuthal loading contour ===")
    flight = EdgewiseFlightCondition.from_rpm(RPM_HOVER, theta0_deg, V_axial=V_axial, V_edge=V_edge)
    res = edge_solver.solve(flight)

    mu = res.get("mu", res.get("advance_ratio", 0.0))
    rev_frac = res.get("reverse_flow_fraction", 0.0)
    m_tip = res.get("M_adv_tip", 0.0)

    print(f"mu={mu:.4f}, reverse-flow fraction={rev_frac*100:.2f}%, "
          f"advancing-tip Mach={m_tip:.3f}")

    psi_deg = np.degrees(res["psi"])
    r_over_R = res["r_over_R"]
    PSI, RR = np.meshgrid(psi_deg, r_over_R)

    plt.figure(figsize=(7.5, 5.5))
    levels = 25

    # Check for thrust derivative variable name
    dT_dr = res.get("dT_blade", res.get("dL_dr", None))
    if dT_dr is None:
        raise KeyError("Neither 'dT_blade' nor 'dL_dr' found in solver results.")

    cf = plt.contourf(PSI, RR, dT_dr, levels=levels, cmap="viridis")
    plt.colorbar(cf, label="Blade-element thrust loading $dT/dr$ [N/m]")

    # Overlay reverse-flow region boundary
    if "reverse_flow" in res and np.any(res["reverse_flow"]):
        plt.contour(PSI, RR, res["reverse_flow"].astype(float), levels=[0.5],
                    colors="red", linewidths=1.5, linestyles="--")
        plt.plot([], [], color="red", linestyle="--", label="Reverse-flow boundary")
        plt.legend(loc="upper right")

    plt.xlabel(r"Azimuth $\psi$ [deg]  (0 = reference, 90 = advancing side)")
    plt.ylabel(r"$r/R$")
    plt.title(f"Azimuthal blade loading, $\\mu$={mu:.3f}, "
              f"$\\theta_0$={theta0_deg:.0f}$^\\circ$")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "task3_2_azimuthal_contour.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"Figure written to ./{fig_path}")


# ============================================================================
# 3.3 -- discretization sensitivity
# ============================================================================

def discretization_sensitivity(theta0_deg: float = 14.0, V_edge: float = 35.0):
    print("\n=== Task 3.3 -- discretization sensitivity ===")
    n_stations_list = [20, 40, 60, 80, 120, 160]
    n_azimuth_list = [12, 18, 24, 36, 48, 72, 96]

    # 1. Radial grid convergence
    CT_vs_nr = []
    for n_r in n_stations_list:
        geom_n = copy.deepcopy(TILTROTOR_GEOM)
        geom_n.n_stations = n_r

        solver_n = EdgewiseBEMTSolver(geom_n, TILTROTOR_AIRFOIL)
        flight = EdgewiseFlightCondition.from_rpm(RPM_HOVER, theta0_deg, V_edge=V_edge)
        res = solver_n.solve(flight)
        ct_val = res.get("CT", res.get("C_T", 0.0))
        CT_vs_nr.append(ct_val)

    CT_vs_nr = np.array(CT_vs_nr)
    print("CT vs n_stations:", dict(zip(n_stations_list, np.round(CT_vs_nr, 6))))

    # 2. Azimuthal grid convergence
    CT_vs_npsi = []
    for n_psi in n_azimuth_list:
        flight = EdgewiseFlightCondition.from_rpm(RPM_HOVER, theta0_deg, V_edge=V_edge, n_azimuth=n_psi)
        res = edge_solver.solve(flight)
        ct_val = res.get("CT", res.get("C_T", 0.0))
        CT_vs_npsi.append(ct_val)

    CT_vs_npsi = np.array(CT_vs_npsi)
    print("CT vs n_azimuth:", dict(zip(n_azimuth_list, np.round(CT_vs_npsi, 6))))

    # Save sensitivity CSV
    csv_path = os.path.join(OUT_DIR, "task3_3_discretization_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_stations", "CT_vs_n_stations", "n_azimuth", "CT_vs_n_azimuth"])
        n_rows = max(len(n_stations_list), len(n_azimuth_list))
        for i in range(n_rows):
            row = [
                n_stations_list[i] if i < len(n_stations_list) else "",
                CT_vs_nr[i] if i < len(n_stations_list) else "",
                n_azimuth_list[i] if i < len(n_azimuth_list) else "",
                CT_vs_npsi[i] if i < len(n_azimuth_list) else "",
            ]
            writer.writerow(row)

    # Plot Sensitivity Figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(n_stations_list, CT_vs_nr, "o-", color="tab:blue")
    axes[0].set_xlabel("Number of radial stations, $n_r$")
    axes[0].set_ylabel("$C_T$")
    axes[0].set_title("Radial discretization sensitivity")
    axes[0].grid(alpha=0.3)

    axes[1].plot(n_azimuth_list, CT_vs_npsi, "s-", color="tab:red")
    axes[1].set_xlabel(r"Number of azimuthal stations, $n_\psi$")
    axes[1].set_ylabel("$C_T$")
    axes[1].set_title("Azimuthal discretization sensitivity")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "task3_3_discretization_sensitivity.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"Figure written to ./{fig_path}")


def main():
    passed = verify_mu_zero_recovery()
    azimuthal_loading_contour()
    discretization_sensitivity()
    print("\n--- Task 3 Verification Summary ---")
    print(f"mu->0 recovery: {'PASSED' if passed else 'FAILED -- investigate before proceeding'}")
    print(f"Figures written to ./{FIG_DIR}/task3_*.png")
    print(f"Tables written to ./{OUT_DIR}/task3_*.csv")


if __name__ == "__main__":
    main()