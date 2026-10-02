from __future__ import annotations
from dataclasses import dataclass, asdict
import csv
import numpy as np
from aircraft_config import AIRCRAFT, ROTOR
from aircraft_model import TransitionAircraftModel, AircraftState

@dataclass(frozen=True)
class TrimCase:
    nacelle_angle_deg: float
    airspeed_ms: float
    altitude_m: float = 500.0
    gross_weight_kg: float = AIRCRAFT.gross_weight_kg
    dT_isa: float = 0.0
    wind_ms: float = 0.0
    climb_rate_ms: float = 0.0
    rpm: float | None = None

class SteadyTrimSolver:
    """Bounded Gauss-Newton steady-trim solver with all six load residuals.

    Unknowns are pitch, collective, lateral cyclic, longitudinal cyclic,
    elevator, rudder, and aileron.  The aileron variable is normally driven
    close to zero by the symmetric twin-rotor model but is retained so the
    solver has an actuator for the body-roll residual MX.
    """
    names=("pitch_deg","collective_deg","theta_1c_lateral_deg",
           "theta_1s_longitudinal_deg","elevator_deg","rudder_deg","aileron_deg")
    residual_names=("FX","FY","FZ","MX","MY","MZ")
    scales=np.array([15000.,5000.,15000.,10000.,10000.,10000.])
    fd_steps=np.array([0.25,0.25,0.25,0.25,0.5,0.5,0.5])
    step_limits=np.array([2.5,2.5,2.,2.,4.,4.,3.])
    def __init__(self,radial_stations=10,n_azimuth=8,max_iter=10,tol=3e-3):
        self.model=TransitionAircraftModel(radial_stations=radial_stations,n_azimuth=n_azimuth)
        self.max_iter=int(max_iter); self.tol=float(tol)
    def rpm_for(self,case):
        if case.rpm is not None:return float(case.rpm)
        return ROTOR.rpm_hover if case.nacelle_angle_deg>=60.0 else ROTOR.rpm_cruise
    def bounds(self):
        lo=np.array([-10.,ROTOR.collective_hover_deg_min,-ROTOR.cyclic_deg_limit,-ROTOR.cyclic_deg_limit,
                     -AIRCRAFT.elevator_deg_limit,-AIRCRAFT.rudder_deg_limit,-AIRCRAFT.aileron_deg_limit])
        hi=np.array([12.,ROTOR.collective_cruise_deg_max,ROTOR.cyclic_deg_limit,ROTOR.cyclic_deg_limit,
                     AIRCRAFT.elevator_deg_limit,AIRCRAFT.rudder_deg_limit,AIRCRAFT.aileron_deg_limit])
        return lo,hi
    def _state(self,c,x):
        return AircraftState(
            airspeed_ms=c.airspeed_ms, altitude_m=c.altitude_m, pitch_deg=x[0],
            nacelle_angle_deg=c.nacelle_angle_deg, rpm=self.rpm_for(c),
            collective_deg=x[1], theta_1c_deg=x[2], theta_1s_deg=x[3],
            elevator_deg=x[4], rudder_deg=x[5], aileron_deg=x[6],
            climb_rate_ms=c.climb_rate_ms, wind_ms=c.wind_ms,
            gross_weight_kg=c.gross_weight_kg, dT_isa=c.dT_isa)
    def loads(self,c,x): return self.model.loads(self._state(c,x))
    def residual(self,c,x):
        o=self.loads(c,x)
        return np.array([o[k] for k in self.residual_names],float)/self.scales
    @staticmethod
    def _clip(x,lo,hi):
        eps=1e-7
        return np.minimum(np.maximum(x,lo+eps),hi-eps)
    def solve(self,c,x0=None):
        lo,hi=self.bounds()
        x=self._clip(np.array(x0 if x0 is not None else [0.,23.,0.,0.,0.,0.,0.],float),lo,hi)
        best=None
        nfev=0
        for it in range(self.max_iter):
            r=self.residual(c,x); nfev += 1
            n=float(np.max(np.abs(r)))
            if best is None or n<best[0]: best=(n,x.copy(),r.copy())
            if n<self.tol: break
            J=np.zeros((6,7))
            for j,h in enumerate(self.fd_steps):
                xp=x.copy(); xm=x.copy()
                xp[j]=min(hi[j],x[j]+h); xm[j]=max(lo[j],x[j]-h)
                rp=self.residual(c,xp); rm=self.residual(c,xm); nfev += 2
                den=xp[j]-xm[j]
                J[:,j]=(rp-rm)/den if den>1e-12 else 0.0
            try: dx=np.linalg.lstsq(J,-r,rcond=1e-8)[0]
            except np.linalg.LinAlgError: break
            dx=np.clip(dx,-self.step_limits,self.step_limits)
            accepted=False; scale=1.0
            for _ in range(7):
                xn=self._clip(x+scale*dx,lo,hi)
                rn=self.residual(c,xn); nfev += 1
                nn=float(np.max(np.abs(rn)))
                if nn <= n*(1-1e-4*scale)+1e-8:
                    x=xn; accepted=True; break
                scale*=0.5
            if not accepted:
                # Take a tiny deterministic trust-region step so the solver
                # still records progress/termination instead of hanging.
                x=self._clip(x+0.1*dx,lo,hi)
        max_scaled,xb,rb=best
        o=self.loads(c,xb); rot=o["rotor"];wing=o["wing"]
        sat=bool(np.any(np.isclose(xb,lo,atol=2e-3)) or np.any(np.isclose(xb,hi,atol=2e-3)))
        numerical=bool(max_scaled<self.tol)
        physical=bool(
            numerical and o["P_required_W"]<=o["P_available_W"] and
            rot["M_adv"]<=ROTOR.tip_mach_limit and
            rot["stall_span"]<=ROTOR.stall_span_fraction_limit and
            rot["reverse_flow_fraction"]<=1e-12 and
            (not wing["wing_stall"] or c.nacelle_angle_deg>60.0 and c.airspeed_ms<25.0) and not sat
        )
        return {
            "case":asdict(c),"x":xb,"names":self.names,"success":numerical,
            "optimizer_success":numerical,"message":"bounded finite-difference Gauss-Newton",
            "nfev":nfev,"iterations":it+1,"residual_norm":float(np.linalg.norm(rb)),
            "max_scaled_residual":float(max_scaled),"loads":np.array([o[k] for k in self.residual_names]),
            "loads_dict":o,"physical_feasible":physical,"control_saturated":sat,
            "bounds":(lo,hi),"state":self._state(c,xb),
        }

def save_trim_result_csv(results,path):
    fields=['nacelle_deg','V_ms','altitude_m','gross_weight_kg',*SteadyTrimSolver.names,'RPM',
            'FX_N','FY_N','FZ_N','MX_Nm','MY_Nm','MZ_Nm','P_required_kW','P_available_kW',
            'stall_span_pct','stall_warning_pct','stall_limit_pct','M_adv','reverse_flow_pct',
            'wing_stall','control_saturated','success','physical_feasible','max_scaled_residual']
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(fields)
        for r in results:
            c=r['case'];x=r['x'];o=r['loads_dict'];rot=o['rotor']
            w.writerow([c['nacelle_angle_deg'],c['airspeed_ms'],c['altitude_m'],c['gross_weight_kg'],*x,
                        r['state'].rpm,*r['loads'],o['P_required_W']/1000,o['P_available_W']/1000,
                        rot['stall_span']*100,ROTOR.stall_span_fraction_warning*100,
                        ROTOR.stall_span_fraction_limit*100,rot['M_adv'],rot['reverse_flow_fraction']*100,
                        o['wing']['wing_stall'],r['control_saturated'],r['success'],r['physical_feasible'],r['max_scaled_residual']])
