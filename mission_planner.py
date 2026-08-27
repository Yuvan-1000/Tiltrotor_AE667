"""
mission_planner.py
================================================================================
Milestone-1 Task 9 / Report Section 2.2 & 7: "Mission Planner v1".

A segment-based mission planner that uses bemt_solver.py (via the designed
rotor in task5_tiltrotor_design.py) as its aerodynamic backend. Supports:
  - hover
  - vertical climb / descent (helicopter mode)
  - axial airplane-mode cruise (with headwind/tailwind)
  - loiter (= sustained hover)
  - payload pickup / drop (instantaneous mass change)

At every time step the planner:
  1. computes the thrust each rotor must produce for the current segment
  2. root-finds the collective that gives that thrust (at the appropriate
     RPM for the current mode) using the SAME validated BEMT solver as
     every other task
  3. computes aerodynamic power -> shaft power (via drivetrain efficiency)
  4. computes power AVAILABLE at the current altitude/temperature (same
     lapse model as Task 6)
  5. burns fuel for that time step using the adopted SFC model, updates
     gross weight
  6. runs ALL Task-10 feasibility checks (fuel, power margin, stall, tip
     Mach, collective/RPM bounds) and stops the mission immediately, with
     a clear segment/time/reason message, on the first violation
================================================================================
"""

import numpy as np
from scipy.optimize import brentq

from bemt_solver import FlightCondition, Atmosphere
from task5_tiltrotor_design import (
    TILTROTOR_GEOM, AIRCRAFT, RPM_HOVER, RPM_CRUISE,
    COLLECTIVE_RANGE_HOVER_DEG, COLLECTIVE_RANGE_CRUISE_DEG, build_solver,
)

G = 9.80665
CRUISE_L_OVER_D = 8.5          # documented assumption A3
STALL_FRACTION_LIMIT = 0.05     # design margin adopted in Tasks 6-7
M_TIP_LIMIT = 0.72               # mission-level compressibility limit

SOLVER = build_solver()


# ================================================================================
# Shared helpers
# ================================================================================
def power_available_kW(altitude_m, dT_isa=0.0):
    atmo = Atmosphere(altitude=altitude_m, dT_isa=dT_isa)
    rho, p, T, a, mu = atmo.properties()
    rho0 = 1.225
    lapse = (rho / rho0) ** 0.75
    P_installed = AIRCRAFT["n_engines"] * AIRCRAFT["sea_level_power_per_engine_kW"] * lapse
    return P_installed * AIRCRAFT["drivetrain_efficiency"]


def _collective_for_thrust(T_target_per_rotor, altitude_m, dT_isa, rpm, coll_range,
                            V_climb=0.0, V_axial=0.0):
    lo, hi = coll_range

    def f(theta0_deg):
        flight = FlightCondition(Omega=rpm * 2.0 * np.pi / 60.0,
                                  collective=np.radians(theta0_deg),
                                  altitude=altitude_m, dT_isa=dT_isa,
                                  V_climb=V_climb, V_axial=V_axial)
        return SOLVER.solve(flight)["T"] - T_target_per_rotor  # Fixed: removed verbose=False

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:
        return None   # cannot reach thrust within allowed collective range
    return brentq(f, lo, hi, xtol=1e-3)


def _solve_for_thrust(T_target_per_rotor, altitude_m, dT_isa, rpm, coll_range,
                       V_climb=0.0, V_axial=0.0):
    theta0 = _collective_for_thrust(T_target_per_rotor, altitude_m, dT_isa, rpm,
                                     coll_range, V_climb, V_axial)
    if theta0 is None:
        return None, None
    flight = FlightCondition(Omega=rpm * 2.0 * np.pi / 60.0, collective=np.radians(theta0),
                              altitude=altitude_m, dT_isa=dT_isa,
                              V_climb=V_climb, V_axial=V_axial)
    res = SOLVER.solve(flight)  # Fixed: removed verbose=False
    return theta0, res


# ================================================================================
# Aircraft state & Exception handling
# ================================================================================
class AircraftState:
    def __init__(self, gross_weight_kg, fuel_kg):
        self.gross_weight_kg = gross_weight_kg
        self.fuel_kg = fuel_kg
        self.time_s = 0.0
        self.distance_km = 0.0

    @property
    def empty_plus_payload_kg(self):
        return self.gross_weight_kg - self.fuel_kg


class MissionFailure(Exception):
    def __init__(self, segment_label, time_s, reason):
        self.segment_label = segment_label
        self.time_s = time_s
        self.reason = reason
        super().__init__(f"[t={time_s:.1f}s, segment='{segment_label}'] {reason}")


class MissionPlanner:
    def __init__(self, gross_weight_kg, fuel_kg, dt_s=30.0, dT_isa=0.0, wind_ms=0.0):
        self.state = AircraftState(gross_weight_kg, fuel_kg)
        self.dt_s = dt_s
        self.dT_isa = dT_isa
        self.wind_ms = wind_ms          # +tailwind / -headwind, cruise only
        self.reserve_fuel_kg = AIRCRAFT["reserve_fuel_fraction"] * AIRCRAFT["fuel_capacity_kg"]
        self.log = []          # list of dict rows per time step
        self.failed = False
        self.failure = None

    # ---- core per-step feasibility + fuel/mass update -----------------
    def _step(self, label, altitude_m, T_target_per_rotor, rpm, coll_range,
              V_climb=0.0, V_axial=0.0, dt_s=None):
        dt_s = self.dt_s if dt_s is None else dt_s
        theta0, res = _solve_for_thrust(T_target_per_rotor, altitude_m, self.dT_isa,
                                         rpm, coll_range, V_climb, V_axial)

        row = dict(t_s=self.state.time_s, segment=label, altitude_m=altitude_m,
                   gross_weight_kg=self.state.gross_weight_kg, fuel_kg=self.state.fuel_kg)

        if theta0 is None:
            reason = (f"required thrust {T_target_per_rotor:.0f} N/rotor is outside the "
                      f"allowed collective range {coll_range} deg at altitude={altitude_m:.0f} m")
            self._fail(label, reason, row)
            return row

        P_aero_total_kW = res["P"] * AIRCRAFT["n_rotors"] / 1000.0
        P_shaft_kW = P_aero_total_kW / AIRCRAFT["drivetrain_efficiency"]
        P_avail_kW = power_available_kW(altitude_m, self.dT_isa)

        row.update(theta0_deg=theta0, P_required_kW=P_shaft_kW, P_available_kW=P_avail_kW,
                   stall_fraction=res["stall_fraction"], M_tip=res["M_tip"])

        # ---- Task 10 feasibility checks (in defined priority order) ----
        if self.state.fuel_kg <= self.reserve_fuel_kg:
            self._fail(label, f"fuel at/below reserve ({self.state.fuel_kg:.1f} kg <= "
                               f"reserve {self.reserve_fuel_kg:.1f} kg)", row)
            return row
        if P_shaft_kW > P_avail_kW:
            self._fail(label, f"power required {P_shaft_kW:.1f} kW exceeds power available "
                               f"{P_avail_kW:.1f} kW", row)
            return row
        if res["stall_fraction"] > STALL_FRACTION_LIMIT:
            self._fail(label, f"stall fraction {res['stall_fraction']:.3f} exceeds limit "
                               f"{STALL_FRACTION_LIMIT:.3f}", row)
            return row
        if res["M_tip"] > M_TIP_LIMIT:
            self._fail(label, f"tip Mach {res['M_tip']:.3f} exceeds limit {M_TIP_LIMIT:.3f}", row)
            return row
        if not (coll_range[0] <= theta0 <= coll_range[1]):
            self._fail(label, f"collective {theta0:.2f} deg outside allowed range {coll_range}", row)
            return row

        # ---- fuel burn & mass update ----
        fuel_used_kg = AIRCRAFT["sfc_kg_per_kWh"] * P_shaft_kW * (dt_s / 3600.0)
        self.state.fuel_kg -= fuel_used_kg
        self.state.gross_weight_kg -= fuel_used_kg
        self.state.time_s += dt_s
        row["fuel_used_kg"] = fuel_used_kg
        self.log.append(row)
        return row

    def _fail(self, label, reason, row=None):
        self.failed = True
        self.failure = MissionFailure(label, self.state.time_s, reason)
        if row is not None:
            row["FAILED"] = True
            row["failure_reason"] = reason
            self.log.append(row)

    # ---- segment implementations ---------------------------------------
    def hover(self, duration_s, altitude_m, label="hover"):
        t_end = self.state.time_s + duration_s
        while self.state.time_s < t_end and not self.failed:
            dt = min(self.dt_s, t_end - self.state.time_s)
            T_target = self.state.gross_weight_kg * G / AIRCRAFT["n_rotors"]
            self._step(label, altitude_m, T_target, RPM_HOVER,
                       COLLECTIVE_RANGE_HOVER_DEG, V_climb=0.0, dt_s=dt)
            if self.failed:
                return

    def loiter(self, duration_s, altitude_m):
        self.hover(duration_s, altitude_m, label="loiter")

    def climb(self, target_altitude_m, climb_rate_ms, label="climb"):
        current_alt = self._last_altitude(default=0.0)
        distance_m = target_altitude_m - current_alt
        if distance_m <= 0:
            return
        duration_s = distance_m / climb_rate_ms
        t_end = self.state.time_s + duration_s
        alt_now = current_alt
        while self.state.time_s < t_end and not self.failed:
            dt = min(self.dt_s, t_end - self.state.time_s)
            alt_now = min(target_altitude_m, alt_now + climb_rate_ms * dt)
            T_target = self.state.gross_weight_kg * G / AIRCRAFT["n_rotors"]
            self._step(label, alt_now, T_target, RPM_HOVER,
                       COLLECTIVE_RANGE_HOVER_DEG, V_climb=climb_rate_ms, dt_s=dt)
            if self.failed:
                return

    def descent(self, target_altitude_m, descent_rate_ms, label="descent"):
        current_alt = self._last_altitude(default=0.0)
        distance_m = current_alt - target_altitude_m
        if distance_m <= 0:
            return
        duration_s = distance_m / descent_rate_ms
        t_end = self.state.time_s + duration_s
        alt_now = current_alt
        while self.state.time_s < t_end and not self.failed:
            dt = min(self.dt_s, t_end - self.state.time_s)
            alt_now = max(target_altitude_m, alt_now - descent_rate_ms * dt)
            T_target = self.state.gross_weight_kg * G / AIRCRAFT["n_rotors"]
            self._step(label, alt_now, T_target, RPM_HOVER,
                       COLLECTIVE_RANGE_HOVER_DEG, V_climb=-descent_rate_ms, dt_s=dt)
            if self.failed:
                return

    def cruise(self, distance_km, altitude_m, airspeed_ms, label="cruise"):
        ground_speed_ms = airspeed_ms + self.wind_ms
        if ground_speed_ms <= 0:
            self._fail(label, f"ground speed {ground_speed_ms:.1f} m/s <= 0 "
                               f"(headwind {abs(self.wind_ms):.1f} m/s exceeds airspeed)")
            return
        distance_m = distance_km * 1000.0
        duration_s = distance_m / ground_speed_ms
        t_end = self.state.time_s + duration_s
        while self.state.time_s < t_end and not self.failed:
            dt = min(self.dt_s, t_end - self.state.time_s)
            T_target = self.state.gross_weight_kg * G / (AIRCRAFT["n_rotors"] * CRUISE_L_OVER_D)
            self._step(label, altitude_m, T_target, RPM_CRUISE,
                       COLLECTIVE_RANGE_CRUISE_DEG, V_axial=airspeed_ms, dt_s=dt)
            if self.failed:
                return
            self.state.distance_km += ground_speed_ms * dt / 1000.0

    def payload_event(self, delta_kg, label="payload"):
        """Instantaneous payload pickup (delta_kg>0) or drop (delta_kg<0)."""
        self.state.gross_weight_kg += delta_kg
        self.log.append(dict(t_s=self.state.time_s, segment=label, altitude_m=self._last_altitude(0.0),
                              gross_weight_kg=self.state.gross_weight_kg, fuel_kg=self.state.fuel_kg,
                              payload_delta_kg=delta_kg))

    def _last_altitude(self, default=0.0):
        for row in reversed(self.log):
            if "altitude_m" in row:
                return row["altitude_m"]
        return default

    def run(self, segments):
        for seg in segments:
            if self.failed:
                break
            kind = seg["type"]
            if kind == "hover":
                self.hover(seg["duration_s"], seg["altitude_m"])
            elif kind == "loiter":
                self.loiter(seg["duration_s"], seg["altitude_m"])
            elif kind == "climb":
                self.climb(seg["target_altitude_m"], seg["climb_rate_ms"])
            elif kind == "descent":
                self.descent(seg["target_altitude_m"], seg["descent_rate_ms"])
            elif kind == "cruise":
                self.cruise(seg["distance_km"], seg["altitude_m"], seg["airspeed_ms"])
            elif kind == "payload":
                self.payload_event(seg["delta_kg"])
            else:
                raise ValueError(f"Unknown segment type: {kind}")
        return self.summary()

    def summary(self):
        return dict(
            completed=not self.failed,
            failure=None if not self.failed else dict(
                segment=self.failure.segment_label, time_s=self.failure.time_s,
                reason=self.failure.reason),
            final_time_s=self.state.time_s,
            final_distance_km=self.state.distance_km,
            final_fuel_kg=self.state.fuel_kg,
            final_gross_weight_kg=self.state.gross_weight_kg,
            n_log_rows=len(self.log),
        )