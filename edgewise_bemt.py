"""
edgewise_bemt.py
================================================================================
Milestone 2 extension of bemt_solver.py: rotor performance in EDGEWISE
(non-axisymmetric, forward-flight) flow, on top of the axial BEMT engine
already validated in Milestone 1.

What's physically new relative to bemt_solver.py's axial BEMT
--------------------------------------------------------------
Once there is an in-plane freestream component V_edge (edgewise flight, or
the in-plane part of a tilted-nacelle conversion condition), each blade sees
a tangential velocity that depends on azimuth psi:

    U_T(r, psi) = Omega*r + V_edge*sin(psi)

so aerodynamic loads vary once per revolution instead of being axisymmetric,
and the induced velocity across the disk becomes non-uniform. We build the
non-uniform inflow on top of a per-station BASELINE inflow lambda_i0(r) taken
from the *already-validated* axial BEMTSolver (bemt_solver.py), run at the
same collective / RPM / axial speed:

    lambda_i(r, psi) = lambda_i0(r) * kappa(r, psi)
    kappa(r, psi)    = 1 + (4/3)*(mu/lambda_bar)/(1.2 + mu/lambda_bar) * (r/R) * cos(psi)

(the linear Glauert/Drees-type azimuthal correction given in the M2 handout,
mu = V_edge/(Omega R) the advance ratio, lambda_bar the disk-average of
lambda_i0). Because kappa -> 1 identically as mu -> 0, this construction
GUARANTEES the edgewise engine collapses onto the Milestone-1 axial numbers
in the mu = 0 limit -- see task3_verification.py for the actual check, this
is the "cheapest correctness gate" flagged in the M2 walkthrough.

Cyclic pitch (theta_1c, theta_1s) is added to the blade pitch schedule --
the actuator a later trim solver (Track B) will use to redirect the thrust
vector / react hub moments.

Documented assumptions (state these explicitly in the M2 report, Sec 2.1/2.2)
-------------------------------------------------------------------------------
- Rigid rotor disk, NO blade flapping (beta = 0). U_P therefore has no
  r*beta_dot flapping-velocity term, and cyclic pitch does not get "eaten"
  by a flapping response -- it shows up directly as a once-per-rev loading
  variation and, since nothing relieves it, as a real HUB MOMENT (see
  hub_pitch_moment / hub_roll_moment below). This is explicitly allowed by
  the M2 handout; it is the same simplification used for control-sweep
  demonstrations before a trim solver exists.
- Reverse flow (U_T < 0: retreating-side root region at high mu) is FLAGGED,
  not aerodynamically modeled -- those blade elements are excluded from the
  thrust/torque integration (reverse_flow_policy="exclude", the default)
  rather than pushed through a forward-flow airfoil polar that is not valid
  there. Full reverse-flow airfoil modeling is out of scope for this
  milestone.
- Radial (spanwise) flow U_R = V_edge*cos(psi) is neglected in the 2D
  blade-element aerodynamics (standard BEMT simplification); it is not
  needed for the quantities this module reports.
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import warnings
import numpy as np

from bemt_solver import (
    Atmosphere, RotorGeometry, AirfoilModel, FlightCondition, BEMTSolver,
)


# ============================================================================
# Flight condition
# ============================================================================

@dataclass
class EdgewiseFlightCondition:
    """One edgewise operating point. V_axial is the through-the-disk
    (perpendicular / climb-like) speed component; V_edge is the in-plane
    (edgewise) speed component that drives the azimuthal asymmetry.
    theta_1c/theta_1s are cyclic pitch amplitudes [rad] (cos/sin harmonics,
    psi measured from the reference blade position at psi = 0)."""

    Omega: float             # [rad/s]
    collective: float        # [rad]  (theta_0)
    theta_1c: float = 0.0    # [rad]  longitudinal cyclic
    theta_1s: float = 0.0    # [rad]  lateral cyclic
    altitude: float = 0.0    # [m]
    dT_isa: float = 0.0      # [K]
    V_axial: float = 0.0     # [m/s]  through the disk
    V_edge: float = 0.0      # [m/s]  in-plane (edgewise)
    n_azimuth: int = 72      # azimuthal stations per revolution

    @property
    def rpm(self):
        return self.Omega * 60.0 / (2.0 * np.pi)

    @classmethod
    def from_rpm(cls, rpm, collective_deg, theta_1c_deg=0.0, theta_1s_deg=0.0, **kwargs):
        return cls(Omega=rpm * 2.0 * np.pi / 60.0,
                   collective=np.radians(collective_deg),
                   theta_1c=np.radians(theta_1c_deg),
                   theta_1s=np.radians(theta_1s_deg), **kwargs)


# ============================================================================
# Edgewise BEMT solver
# ============================================================================

class EdgewiseBEMTSolver:
    """Azimuthal generalization of BEMTSolver. Reuses the axial solver as the
    source of the per-station baseline inflow, and the same tip/root-loss
    factor, so Milestone 1's validated physics carries over unchanged."""

    def __init__(self, geometry: RotorGeometry, airfoil: AirfoilModel,
                 use_tip_loss: bool = True, use_root_loss: bool = False,
                 reverse_flow_policy: str = "exclude"):
        if reverse_flow_policy not in ("exclude", "include"):
            raise ValueError("reverse_flow_policy must be 'exclude' or 'include'")
        self.geom = geometry
        self.airfoil = airfoil
        self.use_tip_loss = use_tip_loss
        self.use_root_loss = use_root_loss
        self.reverse_flow_policy = reverse_flow_policy
        self._axial_solver = BEMTSolver(geometry, airfoil,
                                         use_tip_loss=use_tip_loss,
                                         use_root_loss=use_root_loss)

    def _baseline_inflow(self, flight: EdgewiseFlightCondition):
        """lambda_i0(r): per-station axial-equivalent inflow ratio from the
        Milestone-1 axial BEMTSolver, at the same collective/RPM/V_axial
        (V_edge is irrelevant to the axial baseline by construction)."""
        axial_flight = FlightCondition(flight.Omega, flight.collective,
                                        flight.altitude, flight.dT_isa,
                                        V_climb=flight.V_axial, V_axial=0.0)
        res = self._axial_solver.solve(axial_flight)
        lam0 = res["vi"] / (flight.Omega * self.geom.R)
        return res, lam0

    def solve(self, flight: EdgewiseFlightCondition) -> Dict[str, Any]:
        rho, p, T_air, a_sound, mu_visc = Atmosphere(flight.altitude, flight.dT_isa).properties()
        R, B = self.geom.R, self.geom.B
        r, dr = self.geom.stations()
        c = self.geom.chord(r)
        twist = self.geom.twist(r)
        Vtip = flight.Omega * R
        A_disk = np.pi * R ** 2

        baseline, lam0 = self._baseline_inflow(flight)

        mu = flight.V_edge / Vtip if Vtip != 0 else 0.0
        lam_bar = float(np.sum(lam0 * r * dr) / np.sum(r * dr)) if np.sum(r * dr) > 0 else 0.0
        lam_bar_safe = lam_bar if abs(lam_bar) > 1e-9 else 1e-9
        x = mu / lam_bar_safe
        k = (4.0 / 3.0) * x / (1.2 + x)

        psi = np.linspace(0.0, 2.0 * np.pi, flight.n_azimuth, endpoint=False)
        Rg, PSIg = np.meshgrid(r, psi, indexing="ij")          # (n_r, n_psi)
        kappa = 1.0 + k * (Rg / R) * np.cos(PSIg)
        lam_i = lam0[:, None] * kappa
        vi = lam_i * Vtip

        Ut = flight.Omega * Rg + flight.V_edge * np.sin(PSIg)
        reverse_flow = Ut < 0.0
        Ut_atan = np.where(np.abs(Ut) < 1e-6, np.sign(Ut + 1e-12) * 1e-6, Ut)

        Up = flight.V_axial + vi
        Up = np.where(np.abs(Up) < 1e-4, np.sign(Up + 1e-12) * 1e-4, Up)

        U = np.sqrt(Ut ** 2 + Up ** 2)
        phi = np.arctan2(Up, Ut_atan)
        theta = (flight.collective + twist[:, None]
                 + flight.theta_1c * np.cos(PSIg) + flight.theta_1s * np.sin(PSIg))
        alpha = theta - phi
        Cl, Cd, stalled = self.airfoil.get_cl_cd(alpha)

        # NOTE: tip/root loss is NOT reapplied to the blade-element force here.
        # It already enters through lam0 (the axial baseline was solved with
        # the tip/root-loss-corrected momentum equation), exactly mirroring
        # how bemt_solver.solve() reports dT/dQ from the blade-element form
        # alone once vi has been found. Reapplying F here would double-count it.
        dT_blade = 0.5 * rho * U ** 2 * c[:, None] * (Cl * np.cos(phi) - Cd * np.sin(phi))
        dQ_blade = Rg * 0.5 * rho * U ** 2 * c[:, None] * (Cl * np.sin(phi) + Cd * np.cos(phi))

        if self.reverse_flow_policy == "exclude":
            dT_blade = np.where(reverse_flow, 0.0, dT_blade)
            dQ_blade = np.where(reverse_flow, 0.0, dQ_blade)

        # time- (azimuth-) averaged loading for the whole B-bladed rotor
        dT_avg = B * np.mean(dT_blade, axis=1)     # [N/m]
        dQ_avg = B * np.mean(dQ_blade, axis=1)      # [N.m/m]

        T = float(np.sum(dT_avg * dr))
        Q = float(np.sum(dQ_avg * dr))
        P = float(flight.Omega * Q)
        CT = T / (rho * A_disk * Vtip ** 2)
        CQ = Q / (rho * A_disk * Vtip ** 2 * R)
        CP = P / (rho * A_disk * Vtip ** 3)

        # rigid-hub (no flapping) 1/rev hub moments from the unrelieved
        # once-per-rev thrust harmonic -- physically meaningful consequence
        # of the "no flapping" assumption, and the natural output of a
        # cyclic-pitch control sweep before a trim solver exists.
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        r_dT = Rg * dT_blade                                   # [N] per unit span, per blade
        hub_pitch_moment = float(B / (2.0 * np.pi) * _trapz(
            _trapz(r_dT * np.cos(PSIg), psi, axis=1), r))       # [N.m]
        hub_roll_moment = float(B / (2.0 * np.pi) * _trapz(
            _trapz(r_dT * np.sin(PSIg), psi, axis=1), r))       # [N.m]

        M_local = U / a_sound
        M_tip = Vtip / a_sound
        M_adv_tip = float((Vtip + flight.V_edge) / a_sound)     # worst case: r=R, psi=90 deg
        if M_adv_tip > 0.85:
            warnings.warn(f"Advancing-tip Mach number {M_adv_tip:.3f} exceeds 0.85; "
                           f"compressibility effects are not modeled by the "
                           f"incompressible airfoil data.")

        return dict(
            r=r, r_over_R=r / R, psi=psi, dr=dr,
            mu=mu, lam0=lam0, lam_bar=lam_bar, k=k,
            dT_blade=dT_blade, dQ_blade=dQ_blade, dT_avg=dT_avg, dQ_avg=dQ_avg,
            reverse_flow=reverse_flow, reverse_flow_fraction=float(np.mean(reverse_flow)),
            stalled=stalled, stall_fraction=float(np.mean(stalled)),
            M_local=M_local, M_tip=M_tip, M_adv_tip=M_adv_tip,
            T=T, Q=Q, P=P, CT=CT, CQ=CQ, CP=CP,
            hub_pitch_moment=hub_pitch_moment, hub_roll_moment=hub_roll_moment,
            rho=rho, a_sound=a_sound, Vtip=Vtip, solidity=self.geom.solidity(),
            baseline_axial=baseline,
            rpm=flight.rpm, collective_deg=np.degrees(flight.collective),
            theta1c_deg=np.degrees(flight.theta_1c), theta1s_deg=np.degrees(flight.theta_1s),
        )

    # ------------------------------------------------------------------
    # Control sweeps (Track A, Sec 4: single rotor, one flight condition)
    # ------------------------------------------------------------------

    def sweep_collective(self, collective_deg_array, base_flight: EdgewiseFlightCondition):
        keys = ["T", "Q", "P", "CT", "CQ", "CP", "hub_pitch_moment", "hub_roll_moment",
                "reverse_flow_fraction", "stall_fraction", "M_adv_tip"]
        out = {k: [] for k in ["collective_deg"] + keys}
        for th0_deg in collective_deg_array:
            flight = EdgewiseFlightCondition(base_flight.Omega, np.radians(th0_deg),
                                              base_flight.theta_1c, base_flight.theta_1s,
                                              base_flight.altitude, base_flight.dT_isa,
                                              base_flight.V_axial, base_flight.V_edge,
                                              base_flight.n_azimuth)
            res = self.solve(flight)
            out["collective_deg"].append(th0_deg)
            for k in keys:
                out[k].append(res[k])
        return {k: np.array(v) for k, v in out.items()}

    def sweep_cyclic(self, cyclic_deg_array, base_flight: EdgewiseFlightCondition, axis: str = "1c"):
        """axis='1c' sweeps theta_1c (longitudinal), '1s' sweeps theta_1s (lateral)."""
        if axis not in ("1c", "1s"):
            raise ValueError("axis must be '1c' or '1s'")
        keys = ["T", "Q", "P", "CT", "CQ", "hub_pitch_moment", "hub_roll_moment",
                "reverse_flow_fraction", "stall_fraction"]
        out = {k: [] for k in [f"theta_{axis}_deg"] + keys}
        for th_deg in cyclic_deg_array:
            th1c = base_flight.theta_1c if axis == "1s" else np.radians(th_deg)
            th1s = base_flight.theta_1s if axis == "1c" else np.radians(th_deg)
            flight = EdgewiseFlightCondition(base_flight.Omega, base_flight.collective,
                                              th1c, th1s, base_flight.altitude, base_flight.dT_isa,
                                              base_flight.V_axial, base_flight.V_edge,
                                              base_flight.n_azimuth)
            res = self.solve(flight)
            out[f"theta_{axis}_deg"].append(th_deg)
            for k in keys:
                out[k].append(res[k])
        return {k: np.array(v) for k, v in out.items()}
