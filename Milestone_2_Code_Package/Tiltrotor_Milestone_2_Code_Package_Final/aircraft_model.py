"""Preliminary aircraft force/moment model used by Milestone 2.

The model combines the two edgewise rotor loads with explicit wing,
fuselage/nacelle, and empennage aerodynamics. It is intentionally low-order:
wing/rotor interference, rotor wake download, sideslip, and unsteady dynamics
are omitted and are documented for the report/M3 plan.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from bemt_solver import Atmosphere, RotorGeometry
from aircraft_config import AIRCRAFT, ROTOR, G, TILTROTOR_GEOM, TILTROTOR_AIRFOIL
from edgewise_bemt import EdgewiseBEMTSolver, EdgewiseFlightCondition

@dataclass
class AircraftState:
    airspeed_ms: float
    altitude_m: float
    pitch_deg: float
    nacelle_angle_deg: float
    rpm: float
    collective_deg: float
    theta_1c_deg: float = 0.0
    theta_1s_deg: float = 0.0
    elevator_deg: float = 0.0
    rudder_deg: float = 0.0
    aileron_deg: float = 0.0
    climb_rate_ms: float = 0.0
    wind_ms: float = 0.0
    gross_weight_kg: float = AIRCRAFT.gross_weight_kg
    dT_isa: float = 0.0

class TransitionAircraftModel:
    def __init__(self, radial_stations: int | None = None, n_azimuth: int | None = None):
        nr = radial_stations or ROTOR.n_stations
        na = n_azimuth or 72
        geom = RotorGeometry(**{**TILTROTOR_GEOM.__dict__, "n_stations": nr})
        self.rotor_solver = EdgewiseBEMTSolver(geom, TILTROTOR_AIRFOIL)
        self.n_azimuth = na

    @staticmethod
    def _q(V, h, dT):
        rho, *_ = Atmosphere(h, dT).properties()
        Veff = max(float(abs(V)), 1e-3)
        return rho, 0.5 * rho * Veff**2

    def wing_forces(self, V, h, alpha, dT=0.0):
        rho, q = self._q(V, h, dT)
        S = AIRCRAFT.wing_area_m2
        AR = AIRCRAFT.wing_span_m**2 / S
        CL = AIRCRAFT.wing_cl_alpha_per_rad * (alpha + np.radians(AIRCRAFT.wing_incidence_deg))
        CD = AIRCRAFT.wing_cd0 + CL**2 / (np.pi * AIRCRAFT.wing_oswald * AR)
        L = q * S * CL
        D = q * S * CD
        F = np.array([
            -D * np.cos(alpha) - L * np.sin(alpha),
            0.0,
            D * np.sin(alpha) - L * np.cos(alpha),
        ])
        Cm = AIRCRAFT.wing_cm0 + AIRCRAFT.wing_cma_per_rad * alpha
        My = q * S * AIRCRAFT.wing_mean_chord_m * Cm
        return F, np.array([0.0, My, 0.0]), {
            "CL": CL, "CD": CD, "L": L, "D": D, "q": q,
            "wing_stall": abs(CL) > AIRCRAFT.wing_cl_max,
            "alpha_deg": np.degrees(alpha), "rho": rho,
        }

    def tail_forces(self, V, h, alpha, elevator, rudder, dT=0.0):
        rho, q = self._q(V, h, dT)
        alpha_t = alpha + np.radians(AIRCRAFT.htail_incidence_deg) - 1.2 * np.radians(elevator)
        CLt = AIRCRAFT.htail_cl_alpha_per_rad * alpha_t
        CDt = AIRCRAFT.htail_cd0 + 0.02 * CLt**2
        Lt = q * AIRCRAFT.htail_area_m2 * AIRCRAFT.htail_efficiency * CLt
        Dt = q * AIRCRAFT.htail_area_m2 * CDt
        Ft = np.array([
            -Dt * np.cos(alpha) - Lt * np.sin(alpha),
            0.0,
            Dt * np.sin(alpha) - Lt * np.cos(alpha),
        ])
        # +X forward, +Z down. Positive upward tail lift gives Fz<0 and
        # r x F therefore produces +MY for a tail behind the CG.
        Mt = np.array([0.0, Lt * AIRCRAFT.htail_arm_m, 0.0])
        dr = np.radians(rudder)
        CY = AIRCRAFT.vtail_cy_rudder_per_rad * dr
        Cn = AIRCRAFT.vtail_cn_rudder_per_rad * dr
        Fy = q * AIRCRAFT.vtail_area_m2 * CY
        Mz = q * AIRCRAFT.vtail_area_m2 * AIRCRAFT.vtail_arm_m * Cn
        return Ft + np.array([0.0, Fy, 0.0]), Mt + np.array([0.0, 0.0, Mz]), {
            "CL": CLt, "CD": CDt, "L": Lt, "D": Dt, "Fy": Fy,
            "Cn": Cn, "alpha_deg": np.degrees(alpha_t), "rho": rho,
        }

    def fuselage_drag(self, V, h, alpha, dT=0.0):
        rho, q = self._q(V, h, dT)
        Aeq = AIRCRAFT.fuselage_flat_plate_m2 + 2.0 * AIRCRAFT.nacelle_flat_plate_each_m2
        D = q * Aeq
        return np.array([-D * np.cos(alpha), 0.0, D * np.sin(alpha)]), {"D": D, "q": q, "rho": rho}

    def rotor_pair(self, state: AircraftState):
        x = AIRCRAFT.rotor_hub_x_m
        y = AIRCRAFT.rotor_half_span_m
        z = AIRCRAFT.rotor_hub_z_m
        positions = [np.array([x, +y, z]), np.array([x, -y, z])]
        rotation_signs = [1.0, -1.0]
        results = []
        for pos, sign in zip(positions, rotation_signs):
            flight = EdgewiseFlightCondition.from_body_forward_speed(
                state.rpm, state.collective_deg, state.airspeed_ms,
                theta_1c_deg=state.theta_1c_deg, theta_1s_deg=state.theta_1s_deg,
                altitude=state.altitude_m, dT_isa=state.dT_isa,
                n_azimuth=self.n_azimuth, nacelle_angle_deg=state.nacelle_angle_deg,
                hub_position_body=tuple(pos), rotation_sign=sign,
            )
            results.append(self.rotor_solver.solve(flight))

        F = sum((r["F_body"] for r in results), np.zeros(3))
        M = sum((r["M_hub_body"] + np.cross(pos, r["F_body"])
                 for r, pos in zip(results, positions)), np.zeros(3))
        return F, M, {
            "P_aero_W": float(sum(r["P"] for r in results)),
            "stall_span": float(max(r["stalled_span_fraction"] for r in results)),
            "stall_warning": bool(max(r["stalled_span_fraction"] for r in results) > ROTOR.stall_span_fraction_warning),
            "M_adv": float(max(r["M_adv_tip"] for r in results)),
            "reverse_flow_fraction": float(max(r["reverse_flow_fraction"] for r in results)),
            "reverse_flow_span_fraction": float(max(r["reverse_flow_span_fraction"] for r in results)),
            "rotor_results": results,
        }

    def loads(self, state: AircraftState, include_weight=True):
        V = max(float(abs(state.airspeed_ms)), 1e-3)
        flight_path = np.arcsin(np.clip(state.climb_rate_ms / V, -0.99, 0.99))
        pitch = np.radians(state.pitch_deg)
        alpha = pitch - flight_path
        Fw, Mw, wdat = self.wing_forces(V, state.altitude_m, alpha, state.dT_isa)
        Ft, Mt, tdat = self.tail_forces(V, state.altitude_m, alpha, state.elevator_deg, state.rudder_deg, state.dT_isa)
        Fd, ddat = self.fuselage_drag(V, state.altitude_m, alpha, state.dT_isa)
        Fr, Mr, rdat = self.rotor_pair(state)
        W = state.gross_weight_kg * G
        Fweight = np.array([-W * np.sin(pitch), 0.0, W * np.cos(pitch)]) if include_weight else np.zeros(3)
        _, q = self._q(V, state.altitude_m, state.dT_isa)
        M_ail = np.array([
            q * AIRCRAFT.wing_area_m2 * AIRCRAFT.wing_span_m * 0.08 * np.radians(state.aileron_deg),
            0.0,
            0.0,
        ])
        F = Fr + Fw + Ft + Fd + Fweight
        M = Mr + Mw + Mt + M_ail
        rho, *_ = Atmosphere(state.altitude_m, state.dT_isa).properties()
        P_required = rdat["P_aero_W"] / AIRCRAFT.drivetrain_efficiency
        P_available = AIRCRAFT.n_engines * AIRCRAFT.sea_level_power_per_engine_kW * 1000.0 * (rho / 1.225)
        return {
            "FX": float(F[0]), "FY": float(F[1]), "FZ": float(F[2]),
            "MX": float(M[0]), "MY": float(M[1]), "MZ": float(M[2]),
            "P_required_W": float(P_required), "P_available_W": float(P_available),
            "flight_path_deg": np.degrees(flight_path), "alpha_deg": np.degrees(alpha),
            "rotor_force": Fr, "rotor_moment": Mr, "wing_force": Fw,
            "wing_moment": Mw, "tail_force": Ft, "tail_moment": Mt,
            "drag_force": Fd, "weight_force": Fweight,
            "rotor": rdat, "wing": wdat, "tail": tdat, "drag": ddat,
        }
