"""
task3_edgewise_trim.py
================================================================================
Milestone 2 Task 3: Forward Flight Trimming & Performance Analysis for Edgewise BEMT.

Covers Section 4 of the Milestone 2 Specification:
- Section 4.1: Bounded moment and thrust trim solver (theta_0, theta_1c, theta_1s).
- Section 4.2: Parametric sweeps across advance ratios (mu = V_edge / V_tip).
- Section 4.3: Induced, profile, and propulsive power decomposition.
- Section 4.4: Summary output and performance reporting.
================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List
from scipy.optimize import least_squares

from bemt_solver import RotorGeometry, AirfoilModel
from edgewise_bemt import EdgewiseBEMTSolver, EdgewiseFlightCondition

# Airfoil fallback handling with AoA clamping and stall protection
try:
    from bemt_solver import NACA0012 as ConcreteAirfoil
except ImportError:
    class ConcreteAirfoil(AirfoilModel):
        """Concrete thin-airfoil model with smooth stall and alpha clamping."""
        def get_cl_cd(self, alpha: np.ndarray):
            alpha_arr = np.asarray(alpha)
            # Limit alpha to [-pi/2, pi/2] to prevent unphysical numerical divergence
            alpha_clamped = np.clip(alpha_arr, -np.pi / 2, np.pi / 2)

            # Smooth stall model above 14 degrees
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


def trim_rotor(
    solver: EdgewiseBEMTSolver,
    rpm: float,
    v_edge: float,
    target_thrust: float,
    v_axial: float = 0.0,
    initial_guess: Tuple[float, float, float] = (10.0, 0.0, 0.0),
    tol: float = 1e-4,
    n_azimuth: int = 72
) -> Dict[str, Any]:
    """
    Trims collective (theta_0) and cyclic pitch angles (theta_1c, theta_1s) in degrees
    using bounded least-squares to enforce physical control limits.
    """
    def trim_residuals(x: np.ndarray) -> np.ndarray:
        t0, t1c, t1s = x
        flight = EdgewiseFlightCondition.from_rpm(
            rpm=rpm,
            collective_deg=t0,
            theta_1c_deg=t1c,
            theta_1s_deg=t1s,
            V_axial=v_axial,
            V_edge=v_edge,
            n_azimuth=n_azimuth
        )
        res = solver.solve(flight)

        # Non-dimensionalized residual errors
        r_thrust = (res["T"] - target_thrust) / max(abs(target_thrust), 1.0)
        r_pitch = res["hub_pitch_moment"] / (max(abs(target_thrust), 1.0) * solver.geom.R)
        r_roll = res["hub_roll_moment"] / (max(abs(target_thrust), 1.0) * solver.geom.R)

        return np.array([r_thrust, r_pitch, r_roll])

    # Physical bounds on controls (deg): theta_0 in [0, 30], theta_1c/1s in [-25, 25]
    bounds = ([0.0, -25.0, -25.0], [30.0, 25.0, 25.0])

    # Clip initial guess inside physical bounds
    x0 = np.clip(initial_guess, bounds[0], bounds[1])

    opt_res = least_squares(
        trim_residuals,
        x0=x0,
        bounds=bounds,
        ftol=tol,
        xtol=tol,
        gtol=tol,
        max_nfev=150
    )

    t0_trim, t1c_trim, t1s_trim = opt_res.x

    # Evaluate final trimmed state
    trimmed_flight = EdgewiseFlightCondition.from_rpm(
        rpm=rpm,
        collective_deg=t0_trim,
        theta_1c_deg=t1c_trim,
        theta_1s_deg=t1s_trim,
        V_axial=v_axial,
        V_edge=v_edge,
        n_azimuth=n_azimuth
    )
    results = solver.solve(trimmed_flight)

    results["trim_converged"] = bool(opt_res.success and opt_res.cost < 1e-3)
    results["theta_0_deg"] = float(t0_trim)
    results["theta_1c_deg"] = float(t1c_trim)
    results["theta_1s_deg"] = float(t1s_trim)
    results["target_thrust"] = float(target_thrust)
    results["V_edge"] = float(v_edge)
    results["V_axial"] = float(v_axial)
    results["rpm"] = float(rpm)

    return results


def analyze_power_components(
    solver: EdgewiseBEMTSolver,
    res: Dict[str, Any]
) -> Dict[str, float]:
    """
    Decomposes total rotor power into induced and profile power components.
    """
    P_total = res["P"]
    T = res["T"]

    rho = 1.225
    R = solver.geom.R
    rpm = res.get("rpm", 450.0)
    Omega = rpm * 2.0 * np.pi / 60.0
    V_tip = Omega * R
    A = np.pi * (R ** 2)

    C_T = T / (rho * A * (V_tip ** 2)) if V_tip > 0 else 0.0
    C_P = P_total / (rho * A * (V_tip ** 3)) if V_tip > 0 else 0.0

    v_edge = res.get("V_edge", 0.0)
    mu = v_edge / V_tip if V_tip > 0 else 0.0

    # Glauert induced velocity estimation in forward flight
    v_i0 = np.sqrt(max(T, 0.0) / (2.0 * rho * A))
    if mu > 0.05:
        v_i = v_i0**2 / np.sqrt(v_edge**2 + v_i0**2)
    else:
        v_i = v_i0

    P_induced = T * v_i
    P_profile = max(P_total - P_induced, 0.0)

    return {
        "P_total": P_total,
        "P_induced": P_induced,
        "P_profile": P_profile,
        "C_T": C_T,
        "C_P": C_P,
        "advance_ratio": mu
    }


def sweep_advance_ratio(
    geometry: RotorGeometry,
    airfoil: AirfoilModel,
    rpm: float = 450.0,
    target_thrust: float = 12000.0,
    mu_range: Tuple[float, float, int] = (0.0, 0.30, 7)
) -> List[Dict[str, Any]]:
    """
    Performs a parametric sweep over advance ratio mu = V_edge / V_tip.
    """
    solver = EdgewiseBEMTSolver(geometry, airfoil)
    Omega = rpm * 2.0 * np.pi / 60.0
    V_tip = Omega * geometry.R

    mus = np.linspace(mu_range[0], mu_range[1], mu_range[2])
    sweep_data = []

    guess = (10.0, 0.0, 0.0)

    for mu in mus:
        v_edge = mu * V_tip
        res = trim_rotor(
            solver=solver,
            rpm=rpm,
            v_edge=v_edge,
            target_thrust=target_thrust,
            initial_guess=guess
        )

        # Use previous trimmed solution as next initial guess
        if res["trim_converged"]:
            guess = (res["theta_0_deg"], res["theta_1c_deg"], res["theta_1s_deg"])

        p_decomp = analyze_power_components(solver, res)
        res.update(p_decomp)
        sweep_data.append(res)

    return sweep_data


if __name__ == "__main__":
    geom = RotorGeometry()
    airfoil = ConcreteAirfoil()
    solver = EdgewiseBEMTSolver(geom, airfoil)

    print("=== Milestone 2 - Task 3: Trim & Performance Sweeps ===")

    # 1. Single-Point Trim Verification
    test_rpm = 450.0
    test_v_edge = 30.0  # m/s
    target_T = 12000.0   # N

    print(f"\n1. Executing Trim Search (V_edge = {test_v_edge} m/s, Target Thrust = {target_T} N)...")
    trim_res = trim_rotor(solver, rpm=test_rpm, v_edge=test_v_edge, target_thrust=target_T)

    print(f"   Trim Converged     : {trim_res['trim_converged']}")
    print(f"   Collective (θ0)    : {trim_res['theta_0_deg']:.2f} deg")
    print(f"   Pitch Cyclic (θ1c) : {trim_res['theta_1c_deg']:.2f} deg")
    print(f"   Roll Cyclic (θ1s)  : {trim_res['theta_1s_deg']:.2f} deg")
    print(f"   Resulting Thrust   : {trim_res['T']:.2f} N (Target: {target_T:.2f} N)")
    print(f"   Hub Pitch Moment   : {trim_res['hub_pitch_moment']:.2e} N*m")
    print(f"   Hub Roll Moment    : {trim_res['hub_roll_moment']:.2e} N*m")
    print(f"   Total Power (P)    : {trim_res['P'] / 1e3:.2f} kW")

    # 2. Parametric Advance Ratio Sweep
    print("\n2. Executing Parametric Advance Ratio Sweep (mu = 0.0 to 0.30)...")
    sweep_results = sweep_advance_ratio(
        geom, airfoil, rpm=test_rpm, target_thrust=target_T, mu_range=(0.0, 0.30, 7)
    )

    print("\n   " + f"{'mu':>6} | {'V_edge (m/s)':>12} | {'θ0 (deg)':>9} | {'θ1c (deg)':>9} | {'θ1s (deg)':>9} | {'Power (kW)':>10}")
    print("   " + "-" * 68)
    for s in sweep_results:
        print(f"   {s['advance_ratio']:6.2f} | {s['V_edge']:12.1f} | {s['theta_0_deg']:9.2f} | {s['theta_1c_deg']:9.2f} | {s['theta_1s_deg']:9.2f} | {s['P']/1e3:10.2f}")