"""
task2_edgewise_model.py
================================================================================
Milestone 2 Task 2: Verification and Diagnostic Suite for Edgewise BEMT.

Covers Section 3 of the Milestone 2 Specification:
- Section 3.1: Recovery of Milestone 1 axial BEMT limiting cases (V_edge = 0).
- Section 3.2: Azimuthal loading distribution and periodicity extraction.
- Section 3.3: Reverse-flow, stall, and advancing-tip Mach diagnostics.
- Section 3.4: Numerical sensitivity analysis (Radial and Azimuthal grid convergence).

Imports directly from:
- bemt_solver.py (RotorGeometry, AirfoilModel, BEMTSolver, FlightCondition)
- edgewise_bemt.py (EdgewiseBEMTSolver, EdgewiseFlightCondition)
================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List

from bemt_solver import RotorGeometry, AirfoilModel, BEMTSolver, FlightCondition
from edgewise_bemt import EdgewiseBEMTSolver, EdgewiseFlightCondition

# Try importing concrete airfoil model from bemt_solver, otherwise define fallback
try:
    from bemt_solver import NACA0012 as ConcreteAirfoil
except ImportError:
    class ConcreteAirfoil(AirfoilModel):
        """Concrete thin-airfoil implementation preventing abstract NotImplementedError."""

        def get_cl_cd(self, alpha: np.ndarray):
            alpha_arr = np.asarray(alpha)
            cl = 2.0 * np.pi * alpha_arr
            cd = 0.01 + 0.05 * (alpha_arr ** 2)
            stalled = np.abs(alpha_arr) > np.radians(14.0)
            return cl, cd, stalled


def verify_m1_recovery(
    geometry: RotorGeometry,
    airfoil: AirfoilModel,
    rpm: float = 450.0,
    collective_deg: float = 10.0,
    v_axial: float = 5.0,
    tol: float = 1e-3
) -> Dict[str, Any]:
    """Section 3.1: Verify recovery of Milestone 1 axial BEMT results when V_edge=0."""
    omega = rpm * 2.0 * np.pi / 60.0
    coll_rad = np.radians(collective_deg)

    # 1. Baseline Milestone 1 Axial BEMT Solver
    m1_solver = BEMTSolver(geometry, airfoil, use_tip_loss=True, use_root_loss=False)

    # Instantiate flight condition handling both V_climb and V_total_axial naming
    try:
        m1_flight = FlightCondition(Omega=omega, collective=coll_rad, V_climb=v_axial)
    except TypeError:
        m1_flight = FlightCondition(Omega=omega, collective=coll_rad, V_total_axial=v_axial)

    m1_res = m1_solver.solve(m1_flight)

    # 2. Milestone 2 Edgewise Solver in pure axial limit (V_edge = 0, cyclic = 0)
    m2_solver = EdgewiseBEMTSolver(geometry, airfoil, use_tip_loss=True, use_root_loss=False)
    m2_flight = EdgewiseFlightCondition.from_rpm(
        rpm=rpm,
        collective_deg=collective_deg,
        theta_1c_deg=0.0,
        theta_1s_deg=0.0,
        V_axial=v_axial,
        V_edge=0.0,
        n_azimuth=72
    )
    m2_res = m2_solver.solve(m2_flight)

    # Relative error evaluation
    err_T = abs(m1_res["T"] - m2_res["T"]) / abs(m1_res["T"]) if m1_res["T"] != 0 else 0.0
    err_Q = abs(m1_res["Q"] - m2_res["Q"]) / abs(m1_res["Q"]) if m1_res["Q"] != 0 else 0.0
    err_P = abs(m1_res["P"] - m2_res["P"]) / abs(m1_res["P"]) if m1_res["P"] != 0 else 0.0

    passed = bool((err_T < tol) and (err_Q < tol) and (err_P < tol))

    return {
        "passed": passed,
        "m1_T": m1_res["T"], "m2_T": m2_res["T"], "rel_err_T": err_T,
        "m1_Q": m1_res["Q"], "m2_Q": m2_res["Q"], "rel_err_Q": err_Q,
        "m1_P": m1_res["P"], "m2_P": m2_res["P"], "rel_err_P": err_P,
    }


def extract_azimuthal_loading_and_diagnostics(
    solver: EdgewiseBEMTSolver,
    flight_cond: EdgewiseFlightCondition
) -> Dict[str, Any]:
    """Section 3.2 & 3.3: Extract azimuthal loading, reverse flow, and stall fields."""
    res = solver.solve(flight_cond)

    return {
        "r_over_R": res["r_over_R"],
        "psi_deg": np.degrees(res["psi"]),
        "dL_dr": res["dT_blade"],
        "dQ_dr": res["dQ_blade"],
        "reverse_flow_mask": res["reverse_flow"],
        "stalled_mask": res["stalled"],
        "reverse_flow_fraction": res["reverse_flow_fraction"],
        "stall_fraction": res["stall_fraction"],
        "M_adv_tip": res["M_adv_tip"],
        "T": res["T"],
        "Q": res["Q"],
        "P": res["P"],
        "hub_pitch_moment": res["hub_pitch_moment"],
        "hub_roll_moment": res["hub_roll_moment"],
    }


def run_grid_convergence_study(
    geometry: RotorGeometry,
    airfoil: AirfoilModel,
    base_flight: EdgewiseFlightCondition,
    nr_list: List[int] = [10, 20, 40, 80],
    npsi_list: List[int] = [24, 36, 72, 144]
) -> Dict[str, Any]:
    """Section 3.4: Perform radial (Nr) and azimuthal (Npsi) grid convergence sweeps."""

    # 1. Radial Grid Sensitivity (fixed Npsi = 72)
    radial_results = []
    for nr in nr_list:
        geom_r = RotorGeometry(
            R=geometry.R, B=geometry.B, R_root=geometry.R_root,
            c_root=geometry.c_root, c_tip=geometry.c_tip,
            theta_root=geometry.theta_root, theta_tip=geometry.theta_tip,
            n_stations=nr
        )
        solver_r = EdgewiseBEMTSolver(geom_r, airfoil)
        f_r = EdgewiseFlightCondition(
            Omega=base_flight.Omega, collective=base_flight.collective,
            theta_1c=base_flight.theta_1c, theta_1s=base_flight.theta_1s,
            altitude=base_flight.altitude, dT_isa=base_flight.dT_isa,
            V_axial=base_flight.V_axial, V_edge=base_flight.V_edge,
            n_azimuth=72
        )
        res = solver_r.solve(f_r)
        radial_results.append({
            "nr": nr, "T": res["T"], "Q": res["Q"], "P": res["P"],
            "M_adv_tip": res["M_adv_tip"], "stall_fraction": res["stall_fraction"]
        })

    # 2. Azimuthal Grid Sensitivity (fixed Nr)
    azimuth_results = []
    solver_psi = EdgewiseBEMTSolver(geometry, airfoil)
    for npsi in npsi_list:
        f_psi = EdgewiseFlightCondition(
            Omega=base_flight.Omega, collective=base_flight.collective,
            theta_1c=base_flight.theta_1c, theta_1s=base_flight.theta_1s,
            altitude=base_flight.altitude, dT_isa=base_flight.dT_isa,
            V_axial=base_flight.V_axial, V_edge=base_flight.V_edge,
            n_azimuth=npsi
        )
        res = solver_psi.solve(f_psi)
        azimuth_results.append({
            "npsi": npsi, "T": res["T"], "Q": res["Q"], "P": res["P"],
            "hub_pitch_moment": res["hub_pitch_moment"], "hub_roll_moment": res["hub_roll_moment"]
        })

    return {
        "radial_sensitivity": radial_results,
        "azimuthal_sensitivity": azimuth_results,
    }


if __name__ == "__main__":
    geom = RotorGeometry()

    # Use concrete airfoil instance to prevent NotImplementedError
    airfoil = ConcreteAirfoil()

    print("=== Milestone 2 - Task 2 Verification ===")
    rec = verify_m1_recovery(geom, airfoil)
    print(f"M1 Recovery Check Passed : {rec['passed']}")
    print(f"Thrust Rel. Error        : {rec['rel_err_T']:.2e}")
    print(f"Torque Rel. Error        : {rec['rel_err_Q']:.2e}")
    print(f"Power Rel. Error         : {rec['rel_err_P']:.2e}")