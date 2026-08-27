"""
bemt_solver.py
================================================================================
Blade Element Momentum Theory (BEMT) solver for a rotor in axial flow:
hover, vertical climb/descent, or axial ("propeller mode") forward flight.

Classes:
    Atmosphere    - ISA atmosphere with altitude + temperature-offset input
    LinearAirfoil - Cl = a0*alpha, Cd = Cd_min + eps*alpha^2 (+ stall flag)
    TableAirfoil  - Cl/Cd from an interpolated (alpha, Cl, Cd) polar
    RotorGeometry - blade radius, root cutout, chord/twist distribution
    FlightCondition - one operating point (RPM, collective, altitude, speed)
    BEMTSolver    - combines the above and solves for rotor performance

Reference: Leishman, "Principles of Helicopter Aerodynamics" (combined
blade-element / momentum-theory formulation with Prandtl tip-loss).
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
import warnings
import numpy as np


# ============================================================================
# Atmosphere
# ============================================================================

@dataclass
class Atmosphere:
    """Standard ISA atmosphere (troposphere) with a temperature offset dT_isa
    to represent a hot/cold day. Pressure uses the standard lapse rate;
    density, speed of sound and viscosity use the offset temperature."""

    altitude: float = 0.0   # [m]
    dT_isa: float = 0.0     # [K] offset from standard day

    T0: float = 288.15      # [K] sea-level temperature
    p0: float = 101325.0    # [Pa] sea-level pressure
    g0: float = 9.80665     # [m/s^2]
    R_air: float = 287.05287
    gamma: float = 1.4
    lapse: float = -0.0065  # [K/m]

    def properties(self):
        """Returns (rho, p, T, speed_of_sound, viscosity)."""
        T_std = self.T0 + self.lapse * self.altitude
        p = self.p0 * (T_std / self.T0) ** (-self.g0 / (self.lapse * self.R_air))
        T = T_std + self.dT_isa
        if T <= 0:
            raise ValueError(f"Non-physical temperature ({T:.1f} K); check dT_isa.")

        rho = p / (self.R_air * T)
        a_sound = np.sqrt(self.gamma * self.R_air * T)
        mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)  # Sutherland's law
        return rho, p, T, a_sound, mu


# ============================================================================
# Airfoil models
# ============================================================================

class AirfoilModel:
    """Interface: every airfoil model returns Cl, Cd, stall_flag for alpha [rad]."""

    def get_cl_cd(self, alpha: np.ndarray):
        raise NotImplementedError


@dataclass
class LinearAirfoil(AirfoilModel):
    """Cl = a0*alpha, Cd = Cd_min + eps*alpha^2. Defaults match the
    Knight & Hefner (1937) validation-rotor airfoil model. The stall angle
    is an adopted criterion used only to flag stations -- the linear model
    itself has no physical stall break."""

    a0: float = 5.75
    cd_min: float = 0.0113
    eps: float = 1.25
    alpha_stall: float = np.radians(12.0)  # +/- stall angle

    def get_cl_cd(self, alpha: np.ndarray):
        alpha = np.asarray(alpha, dtype=float)
        cl = self.a0 * alpha
        cd = self.cd_min + self.eps * alpha ** 2
        stalled = np.abs(alpha) > self.alpha_stall
        return cl, cd, stalled


@dataclass
class TableAirfoil(AirfoilModel):
    """Cl/Cd interpolated from a supplied (alpha_deg, Cl, Cd) polar."""

    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    alpha_stall: float = np.radians(12.0)

    def get_cl_cd(self, alpha: np.ndarray):
        alpha_deg = np.degrees(np.asarray(alpha, dtype=float))
        cl = np.interp(alpha_deg, self.alpha_deg, self.cl)
        cd = np.interp(alpha_deg, self.alpha_deg, self.cd)
        stalled = np.abs(alpha) > self.alpha_stall
        return cl, cd, stalled


# ============================================================================
# Rotor geometry
# ============================================================================

@dataclass
class RotorGeometry:
    """Blade planform: radius, root cutout, blade count, and linear
    chord/twist distributions (or custom functions via chord_dist/twist_dist)."""

    R: float = 0.762            # tip radius [m]
    r_root: float = 0.125       # root cutout radius [m]
    B: int = 2                  # number of blades
    chord_root: float = 0.0508  # chord at root cutout [m]
    taper_ratio: float = 1.0    # tip chord / root chord
    twist_root: float = 0.0     # [rad]
    twist_tip: float = 0.0      # [rad]
    n_stations: int = 60        # radial control volumes

    chord_dist: Optional[Callable[[np.ndarray], np.ndarray]] = None
    twist_dist: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def stations(self):
        """Mid-point radial stations and annulus widths for integration."""
        edges = np.linspace(self.r_root, self.R, self.n_stations + 1)
        r = 0.5 * (edges[:-1] + edges[1:])
        dr = np.diff(edges)
        return r, dr

    def chord(self, r):
        if self.chord_dist:
            return np.asarray(self.chord_dist(r), dtype=float)
        frac = (r - self.r_root) / (self.R - self.r_root)
        return self.chord_root * (1 + frac * (self.taper_ratio - 1))

    def twist(self, r):
        if self.twist_dist:
            return np.asarray(self.twist_dist(r), dtype=float)
        frac = (r - self.r_root) / (self.R - self.r_root)
        return self.twist_root + frac * (self.twist_tip - self.twist_root)

    def solidity(self):
        """sigma = B * blade_area / (pi * R^2)"""
        r, dr = self.stations()
        return self.B * np.sum(self.chord(r) * dr) / (np.pi * self.R ** 2)


# ============================================================================
# Flight condition
# ============================================================================

@dataclass
class FlightCondition:
    """One operating point. V_climb (hover/climb) and V_axial (propeller-mode
    forward flight) are both purely axial and simply add together."""

    Omega: float           # [rad/s]
    collective: float      # [rad]
    altitude: float = 0.0  # [m]
    dT_isa: float = 0.0    # [K]
    V_climb: float = 0.0   # [m/s]
    V_axial: float = 0.0   # [m/s]

    @property
    def V_total_axial(self):
        return self.V_climb + self.V_axial

    @property
    def rpm(self):
        return self.Omega * 60.0 / (2.0 * np.pi)

    @classmethod
    def from_rpm(cls, rpm, collective_deg, **kwargs):
        return cls(Omega=rpm * 2.0 * np.pi / 60.0, collective=np.radians(collective_deg), **kwargs)


# ============================================================================
# BEMT solver
# ============================================================================


class BEMTSolver:
    """Solves for the induced velocity at each radial station by equating
    blade-element thrust to Prandtl-tip-loss-corrected momentum-theory
    thrust, then integrates thrust/torque/power over the blade.

    The per-station equation is solved by bisection: the residual
    (blade-element thrust minus momentum thrust) changes sign at the
    physical induced velocity, and bisection converges reliably there even
    where a simple fixed-point iteration can diverge at high blade loading.
    """

    def __init__(self, geometry: RotorGeometry, airfoil: AirfoilModel,
                 use_tip_loss: bool = True, use_root_loss: bool = False,
                 use_compressibility: bool = False,
                 max_iter: int = 100, tol: float = 1e-6):
        self.geom = geometry
        self.airfoil = airfoil
        self.use_tip_loss = use_tip_loss
        self.use_root_loss = use_root_loss
        self.use_compressibility = use_compressibility
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _prandtl_factor(B, r, edge_dist, phi):
        """F = (2/pi) * acos(exp(-f)), f = (B/2) * edge_dist / (r * sin(phi))."""
        sin_phi = np.clip(np.abs(np.sin(phi)), 1e-4, 1.0)
        f = np.clip((B / 2.0) * np.abs(edge_dist) / (np.maximum(r, 1e-6) * sin_phi), 0, 500)
        return np.clip((2.0 / np.pi) * np.arccos(np.exp(-f)), 1e-3, 1.0)

    def _tip_loss(self, r, phi):
        return self._prandtl_factor(self.geom.B, r, self.geom.R - r, phi)

    def _root_loss(self, r, phi):
        return self._prandtl_factor(self.geom.B, r, r - self.geom.r_root, phi)

    def _residual(self, vi, r, c, theta, Ut, V_axial, rho):
        """Blade-element thrust minus momentum thrust, per unit span, as a
        function of induced velocity vi (vectorized over all stations)."""
        Ua = V_axial + vi
        Ua = np.where(np.abs(Ua) < 1e-6, np.sign(Ua + 1e-12) * 1e-6, Ua)
        U = np.sqrt(Ua ** 2 + Ut ** 2)
        phi = np.arctan2(Ua, Ut)
        alpha = theta - phi
        Cl, Cd, _ = self.airfoil.get_cl_cd(alpha)

        dT_be = self.geom.B * 0.5 * rho * U ** 2 * c * (Cl * np.cos(phi) - Cd * np.sin(phi))

        F = np.ones_like(r)
        if self.use_tip_loss:
            F *= self._tip_loss(r, phi)
        if self.use_root_loss:
            F *= self._root_loss(r, phi)
        dT_momentum = 4.0 * np.pi * r * rho * F * Ua * vi

        return dT_be - dT_momentum

    def _solve_induced_velocity(self, r, c, theta, Ut, V_axial, rho, Vtip):
        """Brackets and bisects the induced velocity at every station."""
        n = len(r)
        args = (r, c, theta, Ut, V_axial, rho)

        grid = np.linspace(-1.5 * abs(Vtip) - abs(V_axial) - 1.0,
                            2.5 * abs(Vtip) + abs(V_axial) + 1.0, 300)
        vi_lo, vi_hi, best_dist = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.inf)

        sign_prev = np.sign(self._residual(np.full(n, grid[0]), *args))
        for i in range(1, len(grid)):
            sign_cur = np.sign(self._residual(np.full(n, grid[i]), *args))
            changed = sign_cur != sign_prev
            if np.any(changed):
                mid, dist = 0.5 * (grid[i - 1] + grid[i]), abs(0.5 * (grid[i - 1] + grid[i]))
                better = changed & (dist < best_dist)
                vi_lo, vi_hi = np.where(better, grid[i - 1], vi_lo), np.where(better, grid[i], vi_hi)
                best_dist = np.where(better, dist, best_dist)
            sign_prev = sign_cur

        unbracketed = np.isnan(vi_lo)
        if np.any(unbracketed):
            warnings.warn(f"No induced-velocity root found at {np.sum(unbracketed)} "
                           f"station(s); check for a non-physical operating point.")
        vi_lo, vi_hi = np.where(unbracketed, -1e-6, vi_lo), np.where(unbracketed, 1e-6, vi_hi)

        f_lo = self._residual(vi_lo, *args)
        vi = 0.5 * (vi_lo + vi_hi)
        for it in range(self.max_iter):
            f_mid = self._residual(vi, *args)
            same_side = np.sign(f_mid) == np.sign(f_lo)
            vi_lo, f_lo = np.where(same_side, vi, vi_lo), np.where(same_side, f_mid, f_lo)
            vi_hi = np.where(~same_side, vi, vi_hi)
            vi_new = 0.5 * (vi_lo + vi_hi)
            converged = np.max(np.abs(vi_new - vi)) < self.tol
            vi = vi_new
            if converged:
                break
        else:
            warnings.warn(f"Induced-velocity bisection did not converge in {self.max_iter} iterations.")

        return np.where(unbracketed, 0.0, vi), bool(converged) and not np.any(unbracketed)

    def solve(self, flight: FlightCondition) -> Dict[str, Any]:
        """Solves one operating point and returns radial distributions plus
        integrated rotor performance (T, Q, P, CT, CQ, CP, FM, stall info)."""
        rho, p, T_air, a_sound, mu = Atmosphere(flight.altitude, flight.dT_isa).properties()

        R, B = self.geom.R, self.geom.B
        r, dr = self.geom.stations()
        c = self.geom.chord(r)
        theta = flight.collective + self.geom.twist(r)
        Ut = flight.Omega * r
        Vtip = flight.Omega * R

        vi, converged = self._solve_induced_velocity(r, c, theta, Ut, flight.V_total_axial, rho, Vtip)

        Ua = flight.V_total_axial + vi
        Ua = np.where(np.abs(Ua) < 1e-4, np.sign(Ua + 1e-12) * 1e-4, Ua)
        U = np.sqrt(Ua ** 2 + Ut ** 2)
        phi = np.arctan2(Ua, Ut)
        alpha = theta - phi
        Cl, Cd, stalled = self.airfoil.get_cl_cd(alpha)

        dT = B * 0.5 * rho * U ** 2 * c * (Cl * np.cos(phi) - Cd * np.sin(phi))  # [N/m]
        dQ = B * r * 0.5 * rho * U ** 2 * c * (Cl * np.sin(phi) + Cd * np.cos(phi))  # [N.m/m]

        T = float(np.sum(dT * dr))
        Q = float(np.sum(dQ * dr))
        P = float(flight.Omega * Q)

        A_disk = np.pi * R ** 2
        CT = T / (rho * A_disk * Vtip ** 2)
        CQ = Q / (rho * A_disk * Vtip ** 2 * R)
        CP = P / (rho * A_disk * Vtip ** 3)
        FM = (CT ** 1.5 / np.sqrt(2.0)) / CP if (CT > 0 and CP > 0) else float("nan")

        M_tip = Vtip / a_sound
        if M_tip > 0.85:
            warnings.warn(f"Tip Mach number {M_tip:.3f} exceeds 0.85; compressibility "
                           f"effects are not modeled by the incompressible airfoil data.")

        return dict(
            r=r, r_over_R=r / R, chord=c, twist=self.geom.twist(r), phi=phi,
            alpha_deg=np.degrees(alpha), Cl=Cl, Cd=Cd, stalled=stalled,
            dT=dT, dQ=dQ, vi=vi, M_local=U / a_sound,
            T=T, Q=Q, P=P, CT=CT, CQ=CQ, CP=CP, FM=FM,
            M_tip=M_tip, stall_fraction=float(np.mean(stalled)),
            stalled_radii=r[stalled], converged=converged,
            rpm=flight.rpm, collective_deg=np.degrees(flight.collective),
            rho=rho, a_sound=a_sound, solidity=self.geom.solidity(),
        )

    def sweep_collective(self, collective_deg_array, base_flight: FlightCondition):
        """Runs solve() over a range of collective settings (Task 6: hover maps)."""
        keys = ["T", "Q", "P", "CT", "CQ", "CP", "FM", "M_tip", "stall_fraction", "converged"]
        out = {k: [] for k in ["collective_deg"] + keys}
        for theta0_deg in collective_deg_array:
            flight = FlightCondition(base_flight.Omega, np.radians(theta0_deg),
                                      base_flight.altitude, base_flight.dT_isa,
                                      base_flight.V_climb, base_flight.V_axial)
            res = self.solve(flight)
            out["collective_deg"].append(theta0_deg)
            for k in keys:
                out[k].append(res[k])
        return {k: np.array(v) for k, v in out.items()}

    def sweep_advance_ratio(self, J_array, base_flight: FlightCondition, collective_deg: float):
        """Axial forward-flight sweep at fixed collective over advance ratio
        J = V_axial / (n*D) (Task 7: propeller-mode performance)."""
        n_rev = base_flight.Omega / (2.0 * np.pi)
        D = 2.0 * self.geom.R
        keys = ["T", "Q", "P", "CT", "CQ", "CP", "M_tip", "stall_fraction", "converged"]
        out = {k: [] for k in ["J", "V_axial", "eta_p"] + keys}
        for J in J_array:
            V_axial = J * n_rev * D
            flight = FlightCondition(base_flight.Omega, np.radians(collective_deg),
                                      base_flight.altitude, base_flight.dT_isa,
                                      V_climb=0.0, V_axial=V_axial)
            res = self.solve(flight)
            out["J"].append(J)
            out["V_axial"].append(V_axial)
            out["eta_p"].append(res["T"] * V_axial / res["P"] if res["P"] > 0 else float("nan"))
            for k in keys:
                out[k].append(res[k])
        return {k: np.array(v) for k, v in out.items()}


# ============================================================================
# Validation helper
# ============================================================================

def error_metrics(y_pred, y_exp) -> Dict[str, float]:
    """RMSE, MAE and MAPE between BEMT predictions and experimental data,
    evaluated at matching abscissa points (Task 3: at least two metrics)."""
    y_pred, y_exp = np.asarray(y_pred, dtype=float), np.asarray(y_exp, dtype=float)
    rmse = float(np.sqrt(np.mean((y_pred - y_exp) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_exp)))
    nonzero = np.abs(y_exp) > 1e-12
    mape = float(np.mean(np.abs((y_pred[nonzero] - y_exp[nonzero]) / y_exp[nonzero])) * 100.0)
    return dict(RMSE=rmse, MAE=mae, MAPE_percent=mape)


if __name__ == "__main__":
    geom = RotorGeometry(R=0.762, r_root=0.125, B=2, chord_root=0.0508, taper_ratio=1.0)
    airfoil = LinearAirfoil(a0=5.75, cd_min=0.0113, eps=1.25)
    solver = BEMTSolver(geom, airfoil)

    base_flight = FlightCondition.from_rpm(rpm=1200.0, collective_deg=8.0)
    sweep = solver.sweep_collective(np.arange(2.0, 16.0 + 1e-6, 2.0), base_flight)

    print(f"{'Collective[deg]':>16} {'CT':>10} {'CQ':>11} {'CP':>11} {'FM':>7} {'StallFrac':>10} {'Conv':>6}")
    for i in range(len(sweep["collective_deg"])):
        print(f"{sweep['collective_deg'][i]:>16.1f} {sweep['CT'][i]:>10.5f} "
              f"{sweep['CQ'][i]:>11.6f} {sweep['CP'][i]:>11.6f} {sweep['FM'][i]:>7.3f} "
              f"{sweep['stall_fraction'][i]:>10.2f} {str(bool(sweep['converged'][i])):>6}")