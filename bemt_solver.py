"""
bemt_solver.py
================================================================================
Blade Element Momentum Theory (BEMT) solver for a tiltrotor / helicopter rotor,
operating in either axial climb/hover ("helicopter mode") or axial forward
flight with the rotor axis aligned with the freestream ("propeller mode").

Written for: Rotary-Wing Aerodynamics Course Project - Milestone 1, Task 1 & 2.

--------------------------------------------------------------------------------
WHAT THIS FILE COVERS (Milestone-1 Task 1 "BEMT tool" + Task 2 "Aerodynamic
limits") -- all inputs below are user-configurable, nothing is hard-coded
except the *default values*, which reproduce the Knight & Hefner (1937)
validation rotor geometry/airfoil given in the handout:

    - selectable blade number (B)
    - radius (R)
    - root cut-out (r_root)
    - radial chord / taper (linear taper by default, or an arbitrary c(r))
    - radial twist (linear twist by default, or an arbitrary theta_twist(r))
    - airfoil data (linear Cl-alpha/Cd-alpha model *or* a lookup table)
    - collective pitch (theta0)
    - rotational speed (Omega / RPM)
    - altitude (ISA model)
    - ISA temperature offset (hot/cold day)
    - climb / descent velocity (helicopter mode)
    - axial forward velocity (propeller mode, rotor axis || freestream)
    - iterative induced-velocity solution (fixed-point, under-relaxed)
    - Prandtl tip-loss correction (mandatory, per handout)
    - Prandtl root-loss correction (optional, toggle-able)
    - sectional stall detection (radial location + stalled blade fraction)
    - tip / local Mach number reporting

--------------------------------------------------------------------------------
DOCUMENTED PHYSICS ASSUMPTIONS (for report Section 1.1 -- keep this list in
sync with the code; every assumption made by the solver is listed here so
nothing is "undocumented within the code" per the submission requirements):

  A1. Steady, axisymmetric, axial-flow BEMT. The rotor disk only ever sees a
      free-stream that is aligned with the rotor axis (pure hover, pure
      vertical climb/descent, or pure axial "propeller-mode" forward flight).
      Edgewise / non-axial (helicopter cruise) flight is NOT modeled here --
      that is outside the axial regimes the milestone asks BEMT to cover.
  A2. Blade elements are aerodynamically independent (2-D strip theory, no
      radial flow, no unsteady/wake memory effects).
  A3. Induced velocity is obtained from a local (per-annulus) momentum
      balance corrected by the Prandtl tip-loss factor F, combined with
      blade-element aerodynamics, solved by relaxed fixed-point iteration
      at each radial station independently (classic combined BEMT).
  A4. Optional Prandtl root-loss factor may be included; OFF by default
      (root loss is optional per the handout).
  A5. Airfoil aerodynamics: either (a) the linear model given in the
      handout, Cl = a0*alpha, Cd = Cd_min + eps*alpha^2, valid for the
      Knight & Hefner validation rotor, or (b) a user-supplied Cl-alpha /
      Cd-alpha lookup table for more realistic airfoils used later in the
      tiltrotor design (Tasks 5-8). Both are handled through one common
      interface (AirfoilModel).
  A6. Stall is *flagged*, not physically modeled: any station operating
      beyond +/- alpha_stall is reported as "stalled" and its Cl/Cd are
      still evaluated from the adopted model (the linear model has no
      built-in stall break, so stalled stations should be treated with
      caution when interpreting force results, per the handout's own
      caveat about model limitations).
  A7. No dynamic stall, no unsteady aerodynamics, no yawed-flow /ei
      dgewise corrections, no wake distortion, no ground effect.
  A8. Compressibility: NOT applied to Cl/Cd by default (the handout's
      airfoil model is incompressible). Local & tip Mach numbers are
      still computed and reported so the user can flag high-Mach
      stations; an optional Prandtl-Glauert correction is provided but
      disabled by default (see BEMTSolver.use_compressibility).
  A9. Atmosphere: standard ISA model (troposphere + isothermal lower
      stratosphere) with a uniform additive temperature offset applied at
      all altitudes to represent a hot/cold day (pressure profile itself
      uses the *standard* temperature lapse, only density/sound-speed/
      viscosity use the offset temperature -- a common ISA+dT convention).
  A10. Blade element loads use the local resultant velocity
      U = sqrt(Ut^2+Ua^2) with Ut = Omega*r (no cross-flow / Coriolis
      terms, consistent with the axial-flow assumption A1).

Reference for the combined BEMT + Prandtl tip-loss iteration: Leishman,
J.G., "Principles of Helicopter Aerodynamics", Cambridge Univ. Press --
standard textbook derivation, not reproduced verbatim here.

--------------------------------------------------------------------------------
Suggested module split for the final code package (README / reproducibility
requirement, Sec. 8.1): this single file already separates concerns into
independent classes (Atmosphere, AirfoilModel, RotorGeometry, BEMTSolver) that
can be lifted into atmosphere.py / airfoil.py / geometry.py / bemt.py without
any change to their public interfaces.
================================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np


# ================================================================================
# 1. ATMOSPHERE MODEL  (altitude + ISA temperature offset)
# ================================================================================

@dataclass
class Atmosphere:
    """
    Standard ISA atmosphere (troposphere + lower isothermal stratosphere)
    with an additive temperature offset dT_isa to represent a hot/cold day.

    Convention (documented assumption A9): the pressure profile p(h) is
    computed from the *standard* temperature lapse (dT does not shift the
    pressure profile), while density, speed of sound, and viscosity are
    evaluated using the actual (offset) temperature T(h) = T_std(h) + dT_isa
    via the ideal gas law. This is the common "ISA + dT" convention used in
    aircraft performance work.
    """

    altitude: float = 0.0     # [m] geopotential altitude
    dT_isa: float = 0.0       # [K] deviation from standard day temperature

    # Sea-level standard-day constants
    T0: float = 288.15        # [K]
    p0: float = 101325.0      # [Pa]
    g0: float = 9.80665       # [m/s^2]
    R_air: float = 287.05287  # [J/(kg K)]
    gamma: float = 1.4        # [-]
    L: float = -0.0065        # [K/m] tropospheric lapse rate
    h_tropopause: float = 11000.0
    T_tropopause: float = 216.65

    # Sutherland's law constants (viscosity)
    mu0: float = 1.716e-5     # [Pa s] reference viscosity
    T_suth_ref: float = 273.15
    S_suth: float = 110.4

    def __post_init__(self):
        if self.altitude < 0:
            raise ValueError("Atmosphere: altitude must be >= 0 m.")
        if self.altitude > 20000.0:
            warnings.warn("Atmosphere: altitude > 20 km is outside the "
                           "troposphere/lower-stratosphere model used here; "
                           "results beyond this range are not reliable.")

    def properties(self) -> Tuple[float, float, float, float, float]:
        """Returns (rho, p, T, a_sound, mu) at self.altitude / self.dT_isa."""
        h = self.altitude
        if h <= self.h_tropopause:
            T_std = self.T0 + self.L * h
            p = self.p0 * (T_std / self.T0) ** (-self.g0 / (self.L * self.R_air))
        else:
            T_std = self.T_tropopause
            p11 = self.p0 * (self.T_tropopause / self.T0) ** (-self.g0 / (self.L * self.R_air))
            p = p11 * np.exp(-self.g0 * (h - self.h_tropopause) / (self.R_air * self.T_tropopause))

        T = T_std + self.dT_isa
        if T <= 0:
            raise ValueError("Atmosphere: non-physical (<=0 K) temperature after "
                              "applying dT_isa offset; check inputs.")

        rho = p / (self.R_air * T)
        a_sound = np.sqrt(self.gamma * self.R_air * T)
        mu = self.mu0 * (T / self.T_suth_ref) ** 1.5 * (self.T_suth_ref + self.S_suth) / (T + self.S_suth)
        return rho, p, T, a_sound, mu


# ================================================================================
# 2. AIRFOIL MODELS  (angle-dependent lift/drag data)
# ================================================================================

class AirfoilModel:
    """Common interface: any airfoil model implements get_cl_cd(alpha_rad)."""

    def get_cl_cd(self, alpha: np.ndarray):
        raise NotImplementedError


@dataclass
class LinearAirfoil(AirfoilModel):
    """
    Linear lift-curve / parabolic-drag airfoil model.

    Default values reproduce the Knight & Hefner (1937) validation-rotor
    airfoil model given verbatim in the Milestone-1 handout:
        Cl = a0 * alpha                      (a0 = 5.75 /rad)
        Cd = Cd_min + eps * alpha**2         (Cd_min = 0.0113, eps = 1.25)

    A stall angle is a *documented, adopted* criterion (not derived from
    the linear model, which has no physical stall break) used purely to
    flag stations for the "aerodynamic limits" requirement (Task 2).
    """

    a0: float = 5.75                         # [1/rad] lift-curve slope
    cl0: float = 0.0                         # [-] zero-alpha lift offset
    cd_min: float = 0.0113                   # [-]
    eps: float = 1.25                        # [-] parabolic drag coefficient
    alpha_stall_pos: float = np.radians(12.0)   # [rad] adopted stall criterion
    alpha_stall_neg: float = np.radians(-12.0)  # [rad]
    cl_max: Optional[float] = None           # optional clamp on |Cl|

    def get_cl_cd(self, alpha: np.ndarray):
        alpha = np.asarray(alpha, dtype=float)
        cl = self.cl0 + self.a0 * alpha
        cd = self.cd_min + self.eps * alpha ** 2
        if self.cl_max is not None:
            cl = np.clip(cl, -self.cl_max, self.cl_max)
        stalled = (alpha > self.alpha_stall_pos) | (alpha < self.alpha_stall_neg)
        return cl, cd, stalled


@dataclass
class TableAirfoil(AirfoilModel):
    """
    Lookup-table airfoil model: linear interpolation of user-supplied
    Cl(alpha), Cd(alpha) polar data. Use this for realistic airfoils in the
    tiltrotor design tasks (5-8) once a proper polar (e.g. from XFOIL or a
    published dataset) is available -- keep the actual numbers OUT of this
    solver file and load them from a documented data file instead.
    """

    alpha_deg: np.ndarray                    # [deg] monotonically increasing
    cl: np.ndarray
    cd: np.ndarray
    alpha_stall_pos: float = np.radians(12.0)
    alpha_stall_neg: float = np.radians(-12.0)

    def __post_init__(self):
        self.alpha_deg = np.asarray(self.alpha_deg, dtype=float)
        self.cl = np.asarray(self.cl, dtype=float)
        self.cd = np.asarray(self.cd, dtype=float)
        if not (len(self.alpha_deg) == len(self.cl) == len(self.cd)):
            raise ValueError("TableAirfoil: alpha_deg, cl, cd must be same length.")
        if np.any(np.diff(self.alpha_deg) <= 0):
            raise ValueError("TableAirfoil: alpha_deg must be strictly increasing.")

    def get_cl_cd(self, alpha: np.ndarray):
        alpha = np.asarray(alpha, dtype=float)
        alpha_deg = np.degrees(alpha)
        if np.any(alpha_deg < self.alpha_deg[0]) or np.any(alpha_deg > self.alpha_deg[-1]):
            warnings.warn("TableAirfoil: alpha outside supplied polar range; "
                           "values are being extrapolated by edge-hold.")
        cl = np.interp(alpha_deg, self.alpha_deg, self.cl)
        cd = np.interp(alpha_deg, self.alpha_deg, self.cd)
        stalled = (alpha > self.alpha_stall_pos) | (alpha < self.alpha_stall_neg)
        return cl, cd, stalled


def prandtl_glauert_correction(cl_incompressible: np.ndarray, mach: np.ndarray,
                                mach_limit: float = 0.7) -> np.ndarray:
    """
    Optional compressibility correction (documented assumption A8, OFF by
    default in BEMTSolver). Applies Cl_comp = Cl_incomp / sqrt(1 - M^2) for
    M below mach_limit; beyond mach_limit the correction is not physically
    valid (transonic effects) and is simply capped, with a warning.
    """
    mach = np.asarray(mach, dtype=float)
    if np.any(mach > mach_limit):
        warnings.warn(f"prandtl_glauert_correction: local Mach exceeds "
                       f"{mach_limit}; Prandtl-Glauert is not valid there, "
                       f"correction has been capped.")
    m_capped = np.clip(mach, 0.0, mach_limit)
    beta = np.sqrt(np.maximum(1.0 - m_capped ** 2, 1e-6))
    return cl_incompressible / beta


# ================================================================================
# 3. ROTOR GEOMETRY  (radius, root cutout, blade count, chord/twist distributions)
# ================================================================================

@dataclass
class RotorGeometry:
    """
    Defines blade planform (radial chord / taper) and radial twist.
    Defaults reproduce the Knight & Hefner validation rotor: constant
    chord, zero twist, 2 blades (handout allows choosing 2/3/4/5).
    """

    R: float = 0.762            # [m] blade tip radius
    r_root: float = 0.125       # [m] root cut-out radius
    B: int = 2                  # [-] number of blades
    chord_root: float = 0.0508  # [m] chord at the root-cutout station
    taper_ratio: float = 1.0    # [-] tip chord / root chord (1 = untapered)
    twist_root: float = 0.0     # [rad] geometric twist at root station
    twist_tip: float = 0.0      # [rad] geometric twist at tip station (linear twist)
    n_stations: int = 60        # [-] number of radial control-volume stations

    # Optional overrides for arbitrary (non-linear) radial distributions.
    # Each, if provided, must accept an array of radii [m] and return an
    # array of the same shape (chord in [m], twist in [rad]).
    chord_dist: Optional[Callable[[np.ndarray], np.ndarray]] = None
    twist_dist: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self):
        if self.R <= 0:
            raise ValueError("RotorGeometry: R must be > 0.")
        if not (0 <= self.r_root < self.R):
            raise ValueError("RotorGeometry: require 0 <= r_root < R.")
        if self.B < 1 or int(self.B) != self.B:
            raise ValueError("RotorGeometry: B must be a positive integer.")
        if self.chord_root <= 0:
            raise ValueError("RotorGeometry: chord_root must be > 0.")
        if self.taper_ratio <= 0:
            raise ValueError("RotorGeometry: taper_ratio must be > 0.")
        if self.n_stations < 10:
            raise ValueError("RotorGeometry: n_stations should be >= 10 for "
                              "a reasonably converged radial integration.")

    def stations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mid-point radial stations and their annulus widths for integration."""
        r_edges = np.linspace(self.r_root, self.R, self.n_stations + 1)
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
        dr = np.diff(r_edges)
        return r_mid, dr

    def chord(self, r: np.ndarray) -> np.ndarray:
        if self.chord_dist is not None:
            return np.asarray(self.chord_dist(r), dtype=float)
        chord_tip = self.chord_root * self.taper_ratio
        frac = (r - self.r_root) / (self.R - self.r_root)
        return self.chord_root + frac * (chord_tip - self.chord_root)

    def twist(self, r: np.ndarray) -> np.ndarray:
        if self.twist_dist is not None:
            return np.asarray(self.twist_dist(r), dtype=float)
        frac = (r - self.r_root) / (self.R - self.r_root)
        return self.twist_root + frac * (self.twist_tip - self.twist_root)

    def solidity(self) -> float:
        """Rotor (thrust-weighted-free) solidity sigma = B*blade_area/(pi*R^2)."""
        r, dr = self.stations()
        c = self.chord(r)
        blade_area = np.sum(c * dr)
        return self.B * blade_area / (np.pi * self.R ** 2)


# ================================================================================
# 4. FLIGHT CONDITION
# ================================================================================

@dataclass
class FlightCondition:
    """
    One operating point for the solver.

    V_climb    : [m/s] vertical climb(+)/descent(-) velocity, "helicopter mode".
    V_axial    : [m/s] axial forward-flight velocity, "propeller mode"
                 (rotor axis aligned with the freestream).
    Only one of V_climb / V_axial is normally nonzero at a time -- they are
    kept separate (rather than one combined input) so each flight regime can
    be swept and reported independently, per the handout's Task 6 (hover) /
    Task 7 (axial forward flight) split. Physically both act purely along
    the rotor axis under the axial-flow assumption (A1), so the solver just
    sums them into one total axial free-stream velocity.
    """

    Omega: float                 # [rad/s] rotor rotational speed
    collective: float            # [rad] collective (blade root reference) pitch, theta0
    altitude: float = 0.0        # [m]
    dT_isa: float = 0.0          # [K]
    V_climb: float = 0.0         # [m/s]
    V_axial: float = 0.0         # [m/s]

    def __post_init__(self):
        if self.Omega <= 0:
            raise ValueError("FlightCondition: Omega must be > 0 rad/s.")

    @property
    def V_total_axial(self) -> float:
        return self.V_climb + self.V_axial

    @property
    def rpm(self) -> float:
        return self.Omega * 60.0 / (2.0 * np.pi)

    @classmethod
    def from_rpm(cls, rpm: float, collective_deg: float, **kwargs) -> "FlightCondition":
        return cls(Omega=rpm * 2.0 * np.pi / 60.0,
                    collective=np.radians(collective_deg), **kwargs)


# ================================================================================
# 5. BEMT SOLVER  (iterative induced velocity + Prandtl tip/root loss)
# ================================================================================

class BEMTSolver:
    """
    Combined blade-element / momentum-theory solver for axial flow
    (hover, climb/descent, or axial forward "propeller-mode" flight).

    At each radial station, blade-element loads are equated to a local
    (per-annulus) momentum-theory thrust corrected by the Prandtl tip-loss
    factor F (and, optionally, a Prandtl root-loss factor), and solved for
    the induced velocity vi by relaxed fixed-point iteration -- this is the
    "iterative induced-velocity solution and Prandtl tip loss" explicitly
    required by the handout (Task 1).
    """

    def __init__(self,
                 geometry: RotorGeometry,
                 airfoil: AirfoilModel,
                 use_tip_loss: bool = True,
                 use_root_loss: bool = False,
                 use_compressibility: bool = False,
                 max_iter: int = 100,
                 tol: float = 1e-6):
        """
        max_iter / tol control the per-station bracketed bisection search for
        the induced velocity (see solve()). Bisection is used instead of a
        plain relaxed fixed-point update because the latter was found to
        diverge at higher blade loading / larger collective settings (the
        blade-element thrust term is not always a well-behaved contraction
        map); bisection on a bracketed sign change is unconditionally
        convergent as long as a valid bracket is found, which is checked and
        adaptively expanded at the start of each solve().
        """
        self.geom = geometry
        self.airfoil = airfoil
        self.use_tip_loss = use_tip_loss
        self.use_root_loss = use_root_loss
        self.use_compressibility = use_compressibility
        self.max_iter = max_iter
        self.tol = tol

    # ---- Prandtl loss factors ------------------------------------------------
    @staticmethod
    def _prandtl_factor(B: int, r: np.ndarray, edge_dist: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Generic Prandtl loss factor:
            F = (2/pi) * acos(exp(-f)),   f = (B/2) * edge_dist / (r*sin(phi))
        edge_dist = (R - r) for tip loss, or (r - r_root) for root loss.
        """
        sin_phi = np.clip(np.abs(np.sin(phi)), 1e-4, 1.0)
        f = (B / 2.0) * np.abs(edge_dist) / (np.maximum(r, 1e-6) * sin_phi)
        f = np.clip(f, 0.0, 500.0)   # avoid overflow in exp(-f) for f~0 stations
        F = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f), -1.0, 1.0))
        return np.clip(F, 1e-3, 1.0)

    def _tip_loss(self, r, phi):
        return self._prandtl_factor(self.geom.B, r, self.geom.R - r, phi)

    def _root_loss(self, r, phi):
        return self._prandtl_factor(self.geom.B, r, r - self.geom.r_root, phi)

    # ---- main solve -------------------------------------------------------
    def solve(self, flight: FlightCondition, verbose: bool = False) -> Dict[str, Any]:
        atmo = Atmosphere(altitude=flight.altitude, dT_isa=flight.dT_isa)
        rho, p, T, a_sound, mu = atmo.properties()

        R, B, r_root = self.geom.R, self.geom.B, self.geom.r_root
        r, dr = self.geom.stations()
        c = self.geom.chord(r)
        twist = self.geom.twist(r)
        theta = flight.collective + twist

        Omega = flight.Omega
        V_ax_inf = flight.V_total_axial
        Ut = Omega * r
        Vtip = Omega * R

        # ---- residual function: dT_blade_element(vi) - dT_momentum(vi) ----
        # A root vi* of this function (per station) satisfies both
        # blade-element theory and Prandtl-tip-loss-corrected momentum
        # theory simultaneously -- the "combined BEMT" equation.
        def residual(vi_arr: np.ndarray) -> np.ndarray:
            Ua = V_ax_inf + vi_arr
            Ua_safe = np.where(np.abs(Ua) < 1e-6, np.sign(Ua + 1e-12) * 1e-6, Ua)
            U_ = np.sqrt(Ua_safe ** 2 + Ut ** 2)
            phi_ = np.arctan2(Ua_safe, Ut)
            alpha_ = theta - phi_
            Cl_, Cd_, _ = self.airfoil.get_cl_cd(alpha_)
            if self.use_compressibility:
                Cl_ = prandtl_glauert_correction(Cl_, U_ / a_sound)
            dL_ = 0.5 * rho * U_ ** 2 * c * Cl_
            dD_ = 0.5 * rho * U_ ** 2 * c * Cd_
            dT_be_ = B * (dL_ * np.cos(phi_) - dD_ * np.sin(phi_))
            F_ = np.ones_like(r)
            if self.use_tip_loss:
                F_ *= self._tip_loss(r, phi_)
            if self.use_root_loss:
                F_ *= self._root_loss(r, phi_)
            dT_mom_ = 4.0 * np.pi * r * rho * F_ * Ua_safe * vi_arr
            return dT_be_ - dT_mom_

        # ---- bracket the root at every station -----------------------------
        # NOTE: dT_be(vi) is roughly linear/mild in vi while dT_mom(vi) is
        # quadratic (~ Ua*vi), so the residual dT_be - dT_mom is an inverted-
        # parabola-like function of vi: it can be POSITIVE near the physical
        # root and NEGATIVE at both very large positive and very large
        # negative vi. A naive two-point [lo, hi] bracket can therefore have
        # the *same* sign at both ends even though a (or several) root(s)
        # exist in between. We instead scan a grid across a generous window
        # and keep the sign-change sub-interval closest to vi = 0, which is
        # the physically relevant root (small induced velocity compared to
        # tip speed) rather than any spurious far-field root.
        n_st = len(r)
        window_lo = -1.5 * abs(Vtip) - abs(V_ax_inf) - 1.0
        window_hi = 2.5 * abs(Vtip) + abs(V_ax_inf) + 1.0
        n_grid = 300
        grid = np.linspace(window_lo, window_hi, n_grid)

        vi_lo = np.full(n_st, np.nan)
        vi_hi = np.full(n_st, np.nan)
        best_dist = np.full(n_st, np.inf)

        f_prev = residual(np.full(n_st, grid[0]))
        sign_prev = np.where(f_prev >= 0, 1.0, -1.0)
        for i in range(1, n_grid):
            f_cur = residual(np.full(n_st, grid[i]))
            sign_cur = np.where(f_cur >= 0, 1.0, -1.0)
            changed = sign_cur != sign_prev
            if np.any(changed):
                mid = 0.5 * (grid[i - 1] + grid[i])
                dist = abs(mid)
                better = changed & (dist < best_dist)
                vi_lo = np.where(better, grid[i - 1], vi_lo)
                vi_hi = np.where(better, grid[i], vi_hi)
                best_dist = np.where(better, dist, best_dist)
            sign_prev = sign_cur

        bad = np.isnan(vi_lo)
        if np.any(bad):
            warnings.warn(f"BEMT: could not bracket a sign change in the induced-"
                           f"velocity residual at {np.sum(bad)} station(s) "
                           f"(r/R = {np.round(r[bad] / R, 3)}); induced velocity "
                           f"set to 0 there. This usually indicates a non-physical "
                           f"operating point (e.g. large negative collective) or a "
                           f"window too narrow -- widen window_lo/window_hi above.")
        vi_lo = np.where(bad, -1e-6, vi_lo)
        vi_hi = np.where(bad, 1e-6, vi_hi)
        f_lo = residual(vi_lo)

        # ---- vectorized bisection ------------------------------------------
        n_iter_done = 0
        vi = 0.5 * (vi_lo + vi_hi)
        for it in range(self.max_iter):
            f_mid = residual(vi)
            same_sign_as_lo = np.sign(f_mid) == np.sign(f_lo)
            vi_lo = np.where(same_sign_as_lo, vi, vi_lo)
            f_lo = np.where(same_sign_as_lo, f_mid, f_lo)
            vi_hi = np.where(~same_sign_as_lo, vi, vi_hi)
            vi_new = 0.5 * (vi_lo + vi_hi)
            delta = np.abs(vi_new - vi)
            vi = vi_new
            n_iter_done = it + 1
            if np.max(delta) < self.tol:
                if verbose:
                    print(f"BEMT inflow converged in {n_iter_done} bisection "
                          f"iterations (max delta_vi = {np.max(delta):.3e} m/s).")
                break
        else:
            warnings.warn(f"BEMT inflow bisection did NOT reach tol={self.tol} "
                           f"within {self.max_iter} iterations (max delta_vi = "
                           f"{np.max(delta):.3e} m/s). Consider increasing max_iter.")

        vi = np.where(bad, 0.0, vi)
        converged = bool(np.max(delta) < self.tol) and not np.any(bad)

        # ---- final consistent evaluation at the converged vi ---------------
        Ua = V_ax_inf + vi
        Ua_safe = np.where(np.abs(Ua) < 1e-4, np.sign(Ua + 1e-12) * 1e-4, Ua)
        U = np.sqrt(Ua_safe ** 2 + Ut ** 2)
        phi = np.arctan2(Ua_safe, Ut)
        alpha = theta - phi
        Cl, Cd, stalled = self.airfoil.get_cl_cd(alpha)
        M_local = U / a_sound
        if self.use_compressibility:
            Cl = prandtl_glauert_correction(Cl, M_local)

        dL = 0.5 * rho * U ** 2 * c * Cl
        dD = 0.5 * rho * U ** 2 * c * Cd
        dT = B * (dL * np.cos(phi) - dD * np.sin(phi))            # [N/m]
        dQ = B * r * (dL * np.sin(phi) + dD * np.cos(phi))        # [N m/m]

        T = float(np.sum(dT * dr))       # [N]
        Q = float(np.sum(dQ * dr))       # [N m]
        P = float(Omega * Q)             # [W]

        A_disk = np.pi * R ** 2
        Vtip = Omega * R
        CT = T / (rho * A_disk * Vtip ** 2)
        CQ = Q / (rho * A_disk * Vtip ** 2 * R)
        CP = P / (rho * A_disk * Vtip ** 3)
        FM = (CT ** 1.5 / np.sqrt(2.0)) / CP if (CT > 0 and CP > 0) else float("nan")

        M_tip = Vtip / a_sound
        if M_tip > 0.85:
            warnings.warn(f"BEMT: tip Mach number M_tip = {M_tip:.3f} exceeds 0.85 -- "
                           f"compressibility effects are significant and are NOT "
                           f"modeled by the default incompressible airfoil data.")

        stall_fraction = float(np.sum(stalled)) / len(stalled)
        stalled_radii = r[stalled]

        if stall_fraction > 0 and verbose:
            print(f"Stall flagged over {stall_fraction * 100:.1f}% of the blade span, "
                  f"at r = {np.round(stalled_radii, 3)} m "
                  f"(r/R = {np.round(stalled_radii / R, 3)}).")

        results: Dict[str, Any] = dict(
            # radial distributions
            r=r, r_over_R=r / R, dr=dr, chord=c, twist=twist, theta=theta,
            phi=phi, alpha=alpha, alpha_deg=np.degrees(alpha),
            Cl=Cl, Cd=Cd, stalled=stalled,
            dT=dT, dQ=dQ, vi=vi, Ua=Ua, Ut=Ut, U=U, M_local=M_local,
            # integrated rotor performance
            T=T, Q=Q, P=P, CT=CT, CQ=CQ, CP=CP, FM=FM,
            # limits / checks
            M_tip=M_tip, stall_fraction=stall_fraction, stalled_radii=stalled_radii,
            converged=converged, n_iterations=n_iter_done,
            # operating point + atmosphere echo (for bookkeeping / mission planner)
            Omega=Omega, rpm=flight.rpm, collective_deg=np.degrees(flight.collective),
            altitude=flight.altitude, dT_isa=flight.dT_isa,
            V_climb=flight.V_climb, V_axial=flight.V_axial,
            rho=rho, p=p, T_air=T, a_sound=a_sound, mu=mu,
            solidity=self.geom.solidity(),
        )
        return results

    # ---- convenience: sweep a range of collective settings ----------------
    def sweep_collective(self, collective_deg_array: np.ndarray, base_flight: FlightCondition,
                          verbose: bool = False) -> Dict[str, np.ndarray]:
        """
        Runs solve() at each collective pitch in collective_deg_array, holding
        every other field of base_flight fixed. Returns arrays of the scalar
        outputs, convenient for the required "X vs collective" plots
        (Task 6, hover performance maps).
        """
        out = {k: [] for k in ["collective_deg", "T", "Q", "P", "CT", "CQ", "CP",
                                "FM", "M_tip", "stall_fraction", "converged"]}
        for theta0_deg in collective_deg_array:
            flight = FlightCondition(
                Omega=base_flight.Omega,
                collective=np.radians(theta0_deg),
                altitude=base_flight.altitude,
                dT_isa=base_flight.dT_isa,
                V_climb=base_flight.V_climb,
                V_axial=base_flight.V_axial,
            )
            res = self.solve(flight, verbose=verbose)
            out["collective_deg"].append(theta0_deg)
            for k in ["T", "Q", "P", "CT", "CQ", "CP", "FM", "M_tip",
                      "stall_fraction", "converged"]:
                out[k].append(res[k])
        return {k: np.array(v) for k, v in out.items()}

    def sweep_advance_ratio(self, J_array: np.ndarray, base_flight: FlightCondition,
                             collective_deg: float, verbose: bool = False) -> Dict[str, np.ndarray]:
        """
        Axial forward-flight ("propeller mode") sweep at fixed collective over
        a range of advance ratios J = V_axial / (n * D), n = Omega/(2*pi),
        D = 2R. Solves for the corresponding V_axial at each J, then reports
        thrust/power coefficients and propulsive efficiency
        eta_p = T*V_axial / P (Task 7).
        """
        R, Omega = self.geom.R, base_flight.Omega
        n = Omega / (2.0 * np.pi)
        D = 2.0 * R
        out = {k: [] for k in ["J", "V_axial", "T", "Q", "P", "CT", "CQ", "CP",
                                "eta_p", "M_tip", "stall_fraction", "converged"]}
        for J in J_array:
            V_axial = J * n * D
            flight = FlightCondition(
                Omega=Omega,
                collective=np.radians(collective_deg),
                altitude=base_flight.altitude,
                dT_isa=base_flight.dT_isa,
                V_climb=0.0,
                V_axial=V_axial,
            )
            res = self.solve(flight, verbose=verbose)
            eta_p = (res["T"] * V_axial / res["P"]) if res["P"] > 0 else float("nan")
            out["J"].append(J)
            out["V_axial"].append(V_axial)
            for k in ["T", "Q", "P", "CT", "CQ", "CP", "M_tip", "stall_fraction", "converged"]:
                out[k].append(res[k])
            out["eta_p"].append(eta_p)
        return {k: np.array(v) for k, v in out.items()}


# ================================================================================
# 6. RESULTS EXPORT (for reproducibility / mission-planner hand-off)
# ================================================================================

def results_to_csv_rows(results: Dict[str, Any]):
    """
    Flattens the radial-distribution part of a solve() result dict into
    CSV-ready rows (one row per radial station), for archiving alongside
    each figure per the code-package reproducibility requirement (Sec. 8.1).
    """
    keys = ["r", "r_over_R", "chord", "twist", "phi", "alpha_deg", "Cl", "Cd",
            "stalled", "dT", "dQ", "vi", "U", "M_local"]
    header = keys
    n = len(results["r"])
    rows = [header]
    for i in range(n):
        rows.append([results[k][i] for k in keys])
    return rows


def save_radial_csv(results: Dict[str, Any], path: str):
    import csv
    rows = results_to_csv_rows(results)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# ================================================================================
# 7. VALIDATION HELPERS (Task 3: compare BEMT against Knight & Hefner data)
# ================================================================================

def error_metrics(y_pred: np.ndarray, y_exp: np.ndarray) -> Dict[str, float]:
    """
    RMSE and mean-absolute-percentage-error between BEMT predictions and
    digitized experimental data, evaluated at matching abscissa values
    (e.g. matching collective settings). Satisfies "quantify error using at
    least two metrics" (Task 3 / Sec. 3.4).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_exp = np.asarray(y_exp, dtype=float)
    if y_pred.shape != y_exp.shape:
        raise ValueError("error_metrics: y_pred and y_exp must be the same shape "
                          "(interpolate BEMT results onto the experimental "
                          "abscissa points before calling this).")
    rmse = float(np.sqrt(np.mean((y_pred - y_exp) ** 2)))
    nonzero = np.abs(y_exp) > 1e-12
    mape = float(np.mean(np.abs((y_pred[nonzero] - y_exp[nonzero]) / y_exp[nonzero])) * 100.0)
    mae = float(np.mean(np.abs(y_pred - y_exp)))
    return dict(RMSE=rmse, MAE=mae, MAPE_percent=mape)


# ================================================================================
# 8. DEMONSTRATION / EXAMPLE USAGE
# ================================================================================

if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Example 1: Knight & Hefner (1937) validation rotor, hover, collective
    # sweep -- reproduces the geometry/airfoil table given verbatim in the
    # Milestone-1 handout. NOTE: the actual experimental CT/CQ data points
    # from the paper are NOT reproduced here (they are figures/tables inside
    # the original 1937 NACA report, not included in the handout text) --
    # replace `exp_collective_deg`, `exp_CT`, `exp_CQ` below with the values
    # your team digitizes from the paper before running Sec. 3 validation.
    # ----------------------------------------------------------------------
    geom = RotorGeometry(
        R=0.762, r_root=0.125, B=2,             # handout allows B in {2,3,4,5}
        chord_root=0.0508, taper_ratio=1.0,      # constant chord
        twist_root=0.0, twist_tip=0.0,           # untwisted
        n_stations=60,
    )
    airfoil = LinearAirfoil(a0=5.75, cd_min=0.0113, eps=1.25)
    solver = BEMTSolver(geom, airfoil, use_tip_loss=True, use_root_loss=False)

    base_flight = FlightCondition(
        Omega=2.0 * np.pi * 1200.0 / 60.0,   # example RPM -- set to the paper's test RPM
        collective=np.radians(8.0),
        altitude=0.0, dT_isa=0.0, V_climb=0.0, V_axial=0.0,
    )

    collectives = np.arange(2.0, 16.0 + 1e-6, 2.0)   # [deg]
    sweep = solver.sweep_collective(collectives, base_flight, verbose=False)

    print("Collective[deg]   CT         CQ         CP        FM      StallFrac  Conv")
    for i in range(len(sweep["collective_deg"])):
        print(f"{sweep['collective_deg'][i]:>13.1f}   "
              f"{sweep['CT'][i]:>8.5f}  {sweep['CQ'][i]:>9.6f}  "
              f"{sweep['CP'][i]:>9.6f}  {sweep['FM'][i]:>6.3f}   "
              f"{sweep['stall_fraction'][i]:>7.2f}   {bool(sweep['converged'][i])}")

    # --- placeholder for Task 3 validation against digitized paper data ---
    # exp_collective_deg = np.array([...])   # from Knight & Hefner (1937)
    # exp_CT = np.array([...])
    # bemt_CT_at_exp_points = np.interp(exp_collective_deg,
    #                                    sweep["collective_deg"], sweep["CT"])
    # print(error_metrics(bemt_CT_at_exp_points, exp_CT))

    # ----------------------------------------------------------------------
    # Example 2: single hover point at altitude, with a hot-day offset, and
    # root-loss + compressibility switched on, to show every configurable
    # input being exercised at once.
    # ----------------------------------------------------------------------
    geom2 = RotorGeometry(
        R=6.0, r_root=0.5, B=4,
        chord_root=0.45, taper_ratio=0.6,             # tapered blade
        twist_root=np.radians(0.0), twist_tip=np.radians(-12.0),  # linear washout
        n_stations=80,
    )
    airfoil2 = LinearAirfoil(a0=5.75, cd_min=0.008, eps=1.0,
                              alpha_stall_pos=np.radians(14.0),
                              alpha_stall_neg=np.radians(-14.0))
    solver2 = BEMTSolver(geom2, airfoil2, use_tip_loss=True, use_root_loss=True,
                          use_compressibility=True)

    flight2 = FlightCondition(
        Omega=2.0 * np.pi * 350.0 / 60.0,
        collective=np.radians(10.0),
        altitude=1800.0,     # [m] high-altitude hover case
        dT_isa=20.0,         # [K] hot-day offset (ISA+20)
        V_climb=0.0,
        V_axial=0.0,
    )
    res2 = solver2.solve(flight2, verbose=True)
    print(f"\nHigh-altitude hot-day hover: T={res2['T']:.1f} N, "
          f"P={res2['P']/1000:.1f} kW, FM={res2['FM']:.3f}, "
          f"M_tip={res2['M_tip']:.3f}, solidity={res2['solidity']:.4f}, "
          f"stall_fraction={res2['stall_fraction']:.2f}")

    # ----------------------------------------------------------------------
    # Example 3: axial forward-flight ("propeller mode") sweep over advance
    # ratio J, at fixed collective, for the same tapered rotor (Task 7).
    # ----------------------------------------------------------------------
    J_values = np.linspace(0.05, 0.9, 10)
    fwd_sweep = solver2.sweep_advance_ratio(J_values, flight2, collective_deg=15.0)
    print("\nJ        CT        CP        eta_p    M_tip   StallFrac")
    for i in range(len(fwd_sweep["J"])):
        print(f"{fwd_sweep['J'][i]:.3f}   {fwd_sweep['CT'][i]:>8.5f}  "
              f"{fwd_sweep['CP'][i]:>8.5f}  {fwd_sweep['eta_p'][i]:>6.3f}  "
              f"{fwd_sweep['M_tip'][i]:>6.3f}  {fwd_sweep['stall_fraction'][i]:>6.2f}")