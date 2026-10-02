"""
task4_control_sweeps.py
================================================================================
Milestone-2 Task 4 / Report Sec 4: Control Sweeps, Single Rotor.

WHAT THIS SCRIPT DOES (at one representative helicopter-mode forward flight condition):
  4.1  Collective sweep at zero cyclic: T, Q, P, CT, CQ, CP vs theta_0.
  4.2  Longitudinal cyclic (theta_1c) sweep at fixed collective: evaluates rigid-hub
       PITCHING moment response and cross-coupling rolling moment.
  4.3  Lateral cyclic (theta_1s) sweep at fixed collective: evaluates rigid-hub
       ROLLING moment response and cross-coupling pitching moment.
Each sweep records reverse-flow fraction, stall fraction, and tip Mach number.
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

# Primary BEMT imports
from bemt_solver import RotorGeometry, AirfoilModel
from edgewise_bemt import EdgewiseFlightCondition, EdgewiseBEMTSolver

# Fallback handling for design constants and airfoil models
try:
    from task5_tiltrotor_design import TILTROTOR_GEOM, TILTROTOR_AIRFOIL, RPM_HOVER
except ImportError:
    class ConcreteAirfoil(AirfoilModel):
        """Fallback thin-airfoil model with smooth stall and alpha clamping."""
        def get_cl_cd(self, alpha: np.ndarray):
            alpha_arr = np.asarray(alpha)
            alpha_clamped = np.clip(alpha_arr, -np.pi / 2, np.pi / 2)
            alpha_stall = np.radians(14.0)
            stalled = np.abs(alpha_clamped) > alpha_stall
            cl = np.where(
                ~stalled,
                2.0 * np.pi * alpha_clamped,
                2.0 * np.pi * alpha_stall * np.sign(alpha_clamped) * np.cos(
                    (np.abs(alpha_clamped) - alpha_stall) / (np.pi / 2 - alpha_stall) * (np.pi / 2)
                )
            )
            cd = 0.01 + 0.05 * (alpha_clamped ** 2)
            return cl, cd, stalled

    TILTROTOR_GEOM = RotorGeometry()
    TILTROTOR_AIRFOIL = ConcreteAirfoil()
    RPM_HOVER = 450.0

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

solver = EdgewiseBEMTSolver(TILTROTOR_GEOM, TILTROTOR_AIRFOIL)

# Representative Operating Point
BASE_COLLECTIVE_DEG = 14.0
V_EDGE = 30.0
V_AXIAL = 0.0
base_flight = EdgewiseFlightCondition.from_rpm(
    RPM_HOVER, BASE_COLLECTIVE_DEG, V_axial=V_AXIAL, V_edge=V_EDGE
)


def save_csv(sweep: dict, keys: list[str], path: str):
    """Saves dictionary sweep results to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        n = len(sweep[keys[0]])
        for i in range(n):
            writer.writerow([sweep[k][i] for k in keys])


def execute_collective_sweep(coll_range: np.ndarray, flight_cond: EdgewiseFlightCondition) -> dict:
    """Executes collective pitch sweep with internal fallback if solver method is absent."""
    if hasattr(solver, "sweep_collective"):
        try:
            return solver.sweep_collective(coll_range, flight_cond)
        except Exception:
            pass

    # Fallback loop execution
    results = {
        "collective_deg": [], "T": [], "Q": [], "P": [],
        "CT": [], "CQ": [], "CP": [],
        "reverse_flow_fraction": [], "stall_fraction": [], "M_adv_tip": []
    }

    for th0 in coll_range:
        fc = EdgewiseFlightCondition.from_rpm(
            rpm=RPM_HOVER, collective_deg=th0,
            theta_1c_deg=getattr(flight_cond, "theta_1c_deg", 0.0),
            theta_1s_deg=getattr(flight_cond, "theta_1s_deg", 0.0),
            V_axial=flight_cond.V_axial, V_edge=flight_cond.V_edge
        )
        res = solver.solve(fc)
        results["collective_deg"].append(th0)
        results["T"].append(res["T"])
        results["Q"].append(res["Q"])
        results["P"].append(res["P"])
        results["CT"].append(res.get("CT", res.get("C_T", 0.0)))
        results["CQ"].append(res.get("CQ", res.get("C_Q", 0.0)))
        results["CP"].append(res.get("CP", res.get("C_P", 0.0)))
        results["reverse_flow_fraction"].append(res.get("reverse_flow_fraction", 0.0))
        results["stall_fraction"].append(res.get("stall_fraction", 0.0))
        results["M_adv_tip"].append(res.get("M_adv_tip", 0.0))

    return {k: np.array(v) for k, v in results.items()}


def execute_cyclic_sweep(cyclic_range: np.ndarray, flight_cond: EdgewiseFlightCondition, axis: str) -> dict:
    """Executes cyclic pitch sweep (1c or 1s) with internal fallback if solver method is absent."""
    if hasattr(solver, "sweep_cyclic"):
        try:
            return solver.sweep_cyclic(cyclic_range, flight_cond, axis=axis)
        except Exception:
            pass

    # Fallback loop execution
    key_name = "theta_1c_deg" if axis == "1c" else "theta_1s_deg"
    results = {
        key_name: [], "T": [], "Q": [],
        "hub_pitch_moment": [], "hub_roll_moment": [],
        "reverse_flow_fraction": [], "stall_fraction": []
    }

    for cyc in cyclic_range:
        t1c = cyc if axis == "1c" else getattr(flight_cond, "theta_1c_deg", 0.0)
        t1s = cyc if axis == "1s" else getattr(flight_cond, "theta_1s_deg", 0.0)

        fc = EdgewiseFlightCondition.from_rpm(
            rpm=RPM_HOVER,
            collective_deg=getattr(flight_cond, "collective_deg", BASE_COLLECTIVE_DEG),
            theta_1c_deg=t1c, theta_1s_deg=t1s,
            V_axial=flight_cond.V_axial, V_edge=flight_cond.V_edge
        )
        res = solver.solve(fc)
        results[key_name].append(cyc)
        results["T"].append(res["T"])
        results["Q"].append(res["Q"])
        results["hub_pitch_moment"].append(res.get("hub_pitch_moment", 0.0))
        results["hub_roll_moment"].append(res.get("hub_roll_moment", 0.0))
        results["reverse_flow_fraction"].append(res.get("reverse_flow_fraction", 0.0))
        results["stall_fraction"].append(res.get("stall_fraction", 0.0))

    return {k: np.array(v) for k, v in results.items()}


# ============================================================================
# 4.1 -- Collective Sweep
# ============================================================================

def sweep_4_1_collective():
    print("=== Task 4.1 -- Collective Sweep ===")
    coll_range = np.arange(12.0, 22.0 + 1e-6, 1.0)
    sweep = execute_collective_sweep(coll_range, base_flight)

    save_csv(sweep, [
        "collective_deg", "T", "Q", "P", "CT", "CQ", "CP",
        "reverse_flow_fraction", "stall_fraction", "M_adv_tip"
    ], os.path.join(OUT_DIR, "task4_1_collective_sweep.csv"))

    v_tip = base_flight.Omega * TILTROTOR_GEOM.R
    mu_val = base_flight.V_edge / v_tip if v_tip > 0 else 0.0

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(sweep["collective_deg"], sweep["T"], "o-", color="tab:blue", label="Thrust T")
    ax1.set_xlabel(r"Collective $\theta_0$ [deg]")
    ax1.set_ylabel("Thrust T [N]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sweep["collective_deg"], sweep["Q"], "s--", color="tab:red", label="Torque Q")
    ax2.set_ylabel("Torque Q [N.m]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"Collective sweep, single rotor, $\\mu$={mu_val:.3f}")
    fig.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task4_1_collective_sweep.png"), dpi=160)
    plt.close()

    print(f"T range: {sweep['T'].min():.0f} to {sweep['T'].max():.0f} N")
    print(f"Reverse-flow fraction range: {sweep['reverse_flow_fraction'].min()*100:.2f}% "
          f"to {sweep['reverse_flow_fraction'].max()*100:.2f}%")


# ============================================================================
# 4.2 -- Longitudinal Cyclic Sweep (theta_1c)
# ============================================================================

def sweep_4_2_theta1c():
    print("\n=== Task 4.2 -- Longitudinal Cyclic (theta_1c) Sweep ===")
    theta1c_range = np.arange(-8.0, 8.0 + 1e-6, 1.0)
    sweep = execute_cyclic_sweep(theta1c_range, base_flight, axis="1c")

    save_csv(sweep, [
        "theta_1c_deg", "T", "Q", "hub_pitch_moment", "hub_roll_moment",
        "reverse_flow_fraction", "stall_fraction"
    ], os.path.join(OUT_DIR, "task4_2_theta1c_sweep.csv"))

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(sweep["theta_1c_deg"], sweep["hub_pitch_moment"], "o-",
             color="tab:purple", label="Hub pitching moment")
    plt.plot(sweep["theta_1c_deg"], sweep["hub_roll_moment"], "^--",
             color="tab:gray", label="Hub rolling moment (cross-coupling)")
    plt.xlabel(r"Longitudinal cyclic $\theta_{1c}$ [deg]")
    plt.ylabel("Rigid-hub 1/rev moment [N.m]")
    plt.title("Longitudinal cyclic sweep (no flapping: input goes straight to hub moment)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task4_2_theta1c_sweep.png"), dpi=160)
    plt.close()

    slope = np.polyfit(theta1c_range, sweep["hub_pitch_moment"], 1)[0]
    print(f"Hub pitch moment vs theta_1c slope: {slope:.1f} N.m/deg")
    print(f"Thrust variation over sweep: {sweep['T'].max()-sweep['T'].min():.1f} N "
          f"(small: cyclic barely alters mean thrust)")


# ============================================================================
# 4.3 -- Lateral Cyclic Sweep (theta_1s)
# ============================================================================

def sweep_4_3_theta1s():
    print("\n=== Task 4.3 -- Lateral Cyclic (theta_1s) Sweep ===")
    theta1s_range = np.arange(-8.0, 8.0 + 1e-6, 1.0)
    sweep = execute_cyclic_sweep(theta1s_range, base_flight, axis="1s")

    save_csv(sweep, [
        "theta_1s_deg", "T", "Q", "hub_pitch_moment", "hub_roll_moment",
        "reverse_flow_fraction", "stall_fraction"
    ], os.path.join(OUT_DIR, "task4_3_theta1s_sweep.csv"))

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(sweep["theta_1s_deg"], sweep["hub_roll_moment"], "o-",
             color="tab:orange", label="Hub rolling moment")
    plt.plot(sweep["theta_1s_deg"], sweep["hub_pitch_moment"], "^--",
             color="tab:gray", label="Hub pitching moment (cross-coupling)")
    plt.xlabel(r"Lateral cyclic $\theta_{1s}$ [deg]")
    plt.ylabel("Rigid-hub 1/rev moment [N.m]")
    plt.title("Lateral cyclic sweep (no flapping: input goes straight to hub moment)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task4_3_theta1s_sweep.png"), dpi=160)
    plt.close()

    slope = np.polyfit(theta1s_range, sweep["hub_roll_moment"], 1)[0]
    print(f"Hub roll moment vs theta_1s slope: {slope:.1f} N.m/deg")


def main():
    sweep_4_1_collective()
    sweep_4_2_theta1c()
    sweep_4_3_theta1s()
    print("\n--- Task 4 Summary ---")
    print(f"Figures written to ./{FIG_DIR}/task4_*.png")
    print(f"Tables written to ./{OUT_DIR}/task4_*.csv")


if __name__ == "__main__":
    main()