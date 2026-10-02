"""Milestone 2 azimuth-resolved edgewise rotor model (FIXED).

Changes vs. submitted version
  1. Inflow now uses Glauert's method: the M1 axial BEMT inflow is scaled by
     lam_i,Glauert(mu)/lam_i,Glauert(mu=0) at the same C_T (iterated), then the
     handout non-uniform factor is applied. mu=0 gives ratio==1 exactly, so M1
     recovery (Sec 3.1) is exact.
  2. Shaft torque no longer double counted (it is already inside r x F).
  3. Advancing-tip Mach is the true resultant tip Mach (includes axial flow).
     The lecture value (Vtip+Vedge)/a is returned as M_adv_lecture.
  4. Stall count excludes reverse-flow and near-reverse-flow (Ut<0.1 Vtip).
  5. sweep_* keep rotation_sign.

Course pitch convention:
    theta(psi) = theta0 + theta1c*cos(psi) + theta1s*sin(psi)
theta1c = lateral cyclic, theta1s = longitudinal cyclic.
psi = 0 is the x_R direction (aft in helicopter mode). For rotation_sign=+1 the
advancing blade is at psi=90 deg; for -1 it is at psi=270 deg.
Tip loss is applied in the M1 baseline inflow only, not in the (r,psi) loads.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import warnings
import numpy as np
from bemt_solver import Atmosphere, RotorGeometry, AirfoilModel, FlightCondition, BEMTSolver
from aircraft_config import ROTOR

STALL_NEAR_REVERSE_FRAC = 0.10   # elements with Ut < 0.1*Vtip are not counted as "stalled"


def glauert_induced(CT, mu, lam_c, tol=1e-10):
    """Glauert induced inflow ratio: lam_i = CT / (2 sqrt(mu^2 + (lam_c+lam_i)^2))."""
    CT = max(float(CT), 1e-8)
    lam_i = np.sqrt(CT / 2.0)
    for _ in range(200):
        new = 0.5 * lam_i + 0.5 * CT / (2.0 * np.sqrt(mu**2 + (lam_c + lam_i)**2))
        if abs(new - lam_i) < tol:
            lam_i = new
            break
        lam_i = new
    return lam_i


def rotor_body_basis(nacelle_angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotor-frame x_R,y_R,z_R unit vectors in body axes (+X fwd, +Y stbd, +Z down).

    gamma=90 deg: helicopter mode (+z_R up). gamma=0 deg: airplane mode (+z_R forward).
    """
    g = np.radians(float(nacelle_angle_deg))
    z_R = np.array([np.cos(g), 0.0, -np.sin(g)])
    x_R = np.array([-np.sin(g), 0.0, -np.cos(g)])
    y_R = np.cross(z_R, x_R)
    y_R /= np.linalg.norm(y_R)
    return x_R, y_R, z_R


def rotor_to_body_matrix(nacelle_angle_deg: float) -> np.ndarray:
    return np.column_stack(rotor_body_basis(nacelle_angle_deg))


@dataclass
class EdgewiseFlightCondition:
    Omega: float
    collective: float
    theta_1c: float = 0.0       # lateral cyclic [rad]
    theta_1s: float = 0.0       # longitudinal cyclic [rad]
    altitude: float = 0.0
    dT_isa: float = 0.0
    V_axial: float = 0.0         # through-disk component [m/s]
    V_edge: float = 0.0          # in-plane component [m/s] (>=0)
    n_azimuth: int = 72
    nacelle_angle_deg: float = 90.0
    hub_position_body: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_sign: float = 1.0   # +1 / -1 for opposite rotor spin sense

    def __post_init__(self):
        if self.Omega <= 0:
            raise ValueError("Omega must be > 0")
        if self.n_azimuth < 8:
            raise ValueError("n_azimuth must be >= 8")
        if not 0.0 <= self.nacelle_angle_deg <= 90.0:
            raise ValueError("nacelle_angle_deg must lie in [0,90] deg")
        if self.rotation_sign not in (-1.0, 1.0):
            raise ValueError("rotation_sign must be +1 or -1")
        if self.V_edge < 0:
            raise ValueError("V_edge must be >= 0 (direction is set by psi convention)")

    @property
    def rpm(self) -> float:
        return self.Omega * 60.0 / (2.0 * np.pi)

    @classmethod
    def from_rpm(cls, rpm: float, collective_deg: float,
                 theta_1c_deg: float = 0.0, theta_1s_deg: float = 0.0,
                 **kwargs) -> "EdgewiseFlightCondition":
        return cls(rpm * 2.0 * np.pi / 60.0, np.radians(collective_deg),
                   np.radians(theta_1c_deg), np.radians(theta_1s_deg), **kwargs)

    @classmethod
    def from_body_forward_speed(cls, rpm: float, collective_deg: float,
                                airspeed_ms: float, nacelle_angle_deg: float = 90.0,
                                **kwargs) -> "EdgewiseFlightCondition":
        """Resolve body-forward airspeed into rotor-axis and in-plane parts."""
        g = np.radians(float(nacelle_angle_deg))
        return cls.from_rpm(rpm, collective_deg, nacelle_angle_deg=nacelle_angle_deg,
                            V_axial=airspeed_ms * np.cos(g),
                            V_edge=airspeed_ms * np.sin(g), **kwargs)


class EdgewiseBEMTSolver:
    def __init__(self, geometry: RotorGeometry, airfoil: AirfoilModel,
                 use_tip_loss: bool = True, use_root_loss: bool = False,
                 reverse_flow_policy: str = "exclude",
                 tip_mach_limit: float = ROTOR.tip_mach_limit):
        if reverse_flow_policy not in ("exclude", "include"):
            raise ValueError("reverse_flow_policy must be 'exclude' or 'include'")
        self.geom = geometry
        self.airfoil = airfoil
        self.use_tip_loss = use_tip_loss
        self.use_root_loss = use_root_loss
        self.reverse_flow_policy = reverse_flow_policy
        self.tip_mach_limit = float(tip_mach_limit)
        self._axial_solver = BEMTSolver(geometry, airfoil, use_tip_loss=use_tip_loss,
                                        use_root_loss=use_root_loss)
        self._baseline_cache: Dict[tuple, tuple] = {}

    def _baseline_inflow(self, flight: EdgewiseFlightCondition):
        """Cached M1 axial BEMT at the through-disk speed (independent of azimuth/cyclic)."""
        key = (round(flight.Omega, 12), round(flight.collective, 12), round(flight.altitude, 8),
               round(flight.dT_isa, 8), round(flight.V_axial, 8), self.geom.n_stations)
        if key in self._baseline_cache:
            return self._baseline_cache[key]
        axial = FlightCondition(flight.Omega, flight.collective, flight.altitude, flight.dT_isa,
                                V_climb=flight.V_axial, V_axial=0.0)
        result = self._axial_solver.solve(axial)
        lam0 = result["vi"] / (flight.Omega * self.geom.R)
        if len(self._baseline_cache) > 4000:
            self._baseline_cache.clear()
        self._baseline_cache[key] = (result, lam0)
        return result, lam0

    @staticmethod
    def _safe_signed(value: np.ndarray, eps: float) -> np.ndarray:
        return np.where(np.abs(value) < eps, np.where(value < 0.0, -eps, eps), value)

    def solve(self, flight: EdgewiseFlightCondition) -> Dict[str, Any]:
        rho, _p, _T_air, a_sound, _mu_visc = Atmosphere(flight.altitude, flight.dT_isa).properties()
        R, B = self.geom.R, self.geom.B
        r, dr = self.geom.stations()
        chord = self.geom.chord(r)
        twist = self.geom.twist(r)
        Vtip = flight.Omega * R
        A = np.pi * R**2

        baseline, lam0 = self._baseline_inflow(flight)
        w = r * dr
        lam_bar = float(np.sum(lam0 * w) / np.sum(w))
        mu = flight.V_edge / Vtip
        lam_c = flight.V_axial / Vtip

        psi = np.linspace(0.0, 2.0 * np.pi, int(flight.n_azimuth), endpoint=False)
        Rg, PSIg = np.meshgrid(r, psi, indexing="ij")
        Ut = flight.Omega * Rg + flight.rotation_sign * flight.V_edge * np.sin(PSIg)
        reverse_flow = Ut < 0.0
        Ut_safe = self._safe_signed(Ut, 1e-6)
        theta = (flight.collective + twist[:, None]
                 + flight.theta_1c * np.cos(PSIg) + flight.theta_1s * np.sin(PSIg))

        # Glauert scaling of M1 inflow, iterated on C_T
        CT_est = max(float(baseline["CT"]), 1e-6)
        glauert_ok = False
        for _it in range(30):
            ratio = glauert_induced(CT_est, mu, lam_c) / glauert_induced(CT_est, 0.0, lam_c)
            lam_G = lam_c + ratio * lam_bar                 # handout lambda_G (total inflow)
            lam_G_safe = lam_G if abs(lam_G) > 1e-9 else 1e-9
            x = mu / lam_G_safe
            if 1.2 + x <= 0:
                raise ValueError("Glauert/Drees correction denominator is non-positive")
            k = (4.0 / 3.0) * x / (1.2 + x)
            kappa = 1.0 + k * (Rg / R) * np.cos(PSIg)
            lam_i = ratio * lam0[:, None] * kappa
            vi = lam_i * Vtip
            Up = flight.V_axial + vi
            Up_safe = self._safe_signed(Up, 1e-4)
            U = np.sqrt(Ut**2 + Up_safe**2)
            phi = np.arctan2(Up_safe, Ut_safe)
            alpha = theta - phi
            Cl, Cd, stalled = self.airfoil.get_cl_cd(alpha)
            q_c = 0.5 * rho * U**2 * chord[:, None]
            dT = q_c * (Cl * np.cos(phi) - Cd * np.sin(phi))
            dF_tangential = q_c * (Cl * np.sin(phi) + Cd * np.cos(phi))
            if self.reverse_flow_policy == "exclude":
                dT = np.where(reverse_flow, 0.0, dT)
                dF_tangential = np.where(reverse_flow, 0.0, dF_tangential)
            CT_new = float(np.sum(B * np.mean(dT, axis=1) * dr)) / (rho * A * Vtip**2)
            if abs(CT_new - CT_est) < 1e-6 * max(abs(CT_new), 1e-6):
                glauert_ok = True
                break
            CT_est = 0.5 * CT_est + 0.5 * max(CT_new, 1e-6)
        if not glauert_ok:
            warnings.warn("Edgewise Glauert/thrust iteration did not converge")

        dQ = Rg * dF_tangential
        dT_avg = B * np.mean(dT, axis=1)
        dQ_avg = B * np.mean(dQ, axis=1)
        T = float(np.sum(dT_avg * dr))
        Q = float(np.sum(dQ_avg * dr))
        P = float(flight.Omega * Q)
        CT = T / (rho * A * Vtip**2)
        CQ = Q / (rho * A * Vtip**2 * R)
        CP = P / (rho * A * Vtip**3)

        # Rotor-frame loads. +z_R is thrust. Shaft torque is already contained in
        # r x F (tangential force), so it is NOT added again.
        e_x = np.array([1.0, 0.0, 0.0]); e_y = np.array([0.0, 1.0, 0.0]); e_z = np.array([0.0, 0.0, 1.0])
        e_r = np.cos(PSIg)[..., None] * e_x + np.sin(PSIg)[..., None] * e_y
        e_t = flight.rotation_sign * (-np.sin(PSIg)[..., None] * e_x + np.cos(PSIg)[..., None] * e_y)
        dF = dT[..., None] * e_z - dF_tangential[..., None] * e_t
        dM = np.cross(Rg[..., None] * e_r, dF)
        F_rotor = np.sum(B * np.mean(dF, axis=1) * dr[:, None], axis=0)
        M_hub_rotor = np.sum(B * np.mean(dM, axis=1) * dr[:, None], axis=0)

        C_BR = rotor_to_body_matrix(flight.nacelle_angle_deg)
        F_body = C_BR @ F_rotor
        M_hub_body = C_BR @ M_hub_rotor
        r_hub = np.asarray(flight.hub_position_body, dtype=float)
        M_body_reference = M_hub_body + np.cross(r_hub, F_body)

        valid = ~reverse_flow if self.reverse_flow_policy == "exclude" else np.ones_like(reverse_flow, dtype=bool)
        M_local = U / a_sound
        M_adv_tip = float(np.max(np.where(valid, U, 0.0)) / a_sound)    # true resultant tip Mach
        M_adv_lecture = float((Vtip + flight.V_edge) / a_sound)
        if M_adv_tip > self.tip_mach_limit:
            warnings.warn(f"Advancing-tip Mach {M_adv_tip:.3f} exceeds adopted limit "
                          f"{self.tip_mach_limit:.2f}.")

        stall_valid = valid & (Ut > STALL_NEAR_REVERSE_FRAC * Vtip)
        nvalid = max(int(stall_valid.sum()), 1)
        stall_fraction = float(np.sum(stalled & stall_valid) / nvalid)
        stalled_span_fraction = float(np.count_nonzero(np.any(stalled & stall_valid, axis=1)) / len(r))

        return {
            "r": r, "r_over_R": r / R, "dr": dr, "psi": psi,
            "chord": chord, "twist": twist, "theta": theta, "phi": phi,
            "alpha": alpha, "alpha_deg": np.degrees(alpha), "Cl": Cl, "Cd": Cd,
            "stalled": stalled, "reverse_flow": reverse_flow,
            "dT_blade": dT, "dFt_blade": dF_tangential, "dQ_blade": dQ,
            "dT_avg": dT_avg, "dQ_avg": dQ_avg,
            "vi": vi, "lam_i": lam_i, "kappa": kappa,
            "Ut": Ut, "Up": Up, "U": U, "M_local": M_local,
            "mu": mu, "lam0": lam0, "lam_bar": lam_bar,
            "lam_G": lam_G, "k": k, "glauert_ratio": float(ratio),
            "T": T, "Q": Q, "P": P, "CT": CT, "CQ": CQ, "CP": CP,
            "F_rotor": F_rotor, "M_hub_rotor": M_hub_rotor,
            "F_body": F_body, "M_hub_body": M_hub_body,
            "M_body_reference": M_body_reference,
            "FX": float(F_body[0]), "FY": float(F_body[1]), "FZ": float(F_body[2]),
            "MX": float(M_hub_body[0]), "MY": float(M_hub_body[1]), "MZ": float(M_hub_body[2]),
            "reverse_flow_fraction": float(np.mean(reverse_flow)),
            "reverse_flow_span_fraction": float(np.mean(np.any(reverse_flow, axis=1))),
            "stall_fraction": stall_fraction,
            "stalled_span_fraction": stalled_span_fraction,
            "M_tip": Vtip / a_sound, "M_adv_tip": M_adv_tip, "M_adv_lecture": M_adv_lecture,
            "tip_mach_limit": self.tip_mach_limit,
            "tip_mach_margin": self.tip_mach_limit - M_adv_tip,
            "rho": rho, "a_sound": a_sound, "Vtip": Vtip,
            "solidity": self.geom.solidity(), "rpm": flight.rpm,
            "collective_deg": np.degrees(flight.collective),
            "theta1c_deg": np.degrees(flight.theta_1c),
            "theta1s_deg": np.degrees(flight.theta_1s),
            "nacelle_angle_deg": flight.nacelle_angle_deg,
            "baseline_axial": baseline,
            "glauert_converged": glauert_ok,
            "converged": bool(baseline.get("converged", True)) and glauert_ok,
            "baseline_n_iterations": int(baseline.get("n_iterations", 0)),
        }

    def _copy(self, base, **over):
        d = dict(Omega=base.Omega, collective=base.collective, theta_1c=base.theta_1c,
                 theta_1s=base.theta_1s, altitude=base.altitude, dT_isa=base.dT_isa,
                 V_axial=base.V_axial, V_edge=base.V_edge, n_azimuth=base.n_azimuth,
                 nacelle_angle_deg=base.nacelle_angle_deg,
                 hub_position_body=base.hub_position_body, rotation_sign=base.rotation_sign)
        d.update(over)
        return EdgewiseFlightCondition(**d)

    def sweep_collective(self, values, base: EdgewiseFlightCondition):
        return [self.solve(self._copy(base, collective=np.radians(v))) for v in values]

    def sweep_cyclic(self, values, base: EdgewiseFlightCondition, axis: str):
        if axis not in ("lateral", "longitudinal"):
            raise ValueError("axis must be 'lateral' or 'longitudinal'")
        out = []
        for v in values:
            over = {"theta_1c": np.radians(v)} if axis == "lateral" else {"theta_1s": np.radians(v)}
            out.append(self.solve(self._copy(base, **over)))
        return out