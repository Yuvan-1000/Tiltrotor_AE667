"""Mission Planner v2 for the Milestone-2 transition demonstrations.

The planner is a thin state/mass/fuel layer around the same aircraft model used
by the steady-trim calculation. It supports scheduled state variables and
scheduled controls, records every time step, and stops at the first violated
hard constraint. Trim residual is logged at every step and can be promoted to a
hard constraint for online-trim operation.
"""
from __future__ import annotations
from dataclasses import dataclass
import csv
import numpy as np
from aircraft_config import AIRCRAFT, ROTOR
from aircraft_model import TransitionAircraftModel, AircraftState

RESIDUAL_SCALES=np.array([30000.,3000.,30000.,10000.,20000.,20000.])


def clamp(value,lo,hi): return max(lo,min(hi,value))

class PiecewiseSchedule:
    def __init__(self,knots):
        if not knots: raise ValueError("schedule cannot be empty")
        self.knots=sorted((float(t),float(v)) for t,v in knots)
    def __call__(self,t):
        if len(self.knots)==1:return self.knots[0][1]
        ts=np.array([k[0] for k in self.knots]);vs=np.array([k[1] for k in self.knots])
        return float(np.interp(float(t),ts,vs))

def make_linear_schedule(start,end,duration): return PiecewiseSchedule([(0.0,start),(float(duration),end)])

class TrimControlLookup:
    """Interpolate controls over a supplied trim table."""
    def __init__(self,csv_path):
        with open(csv_path,newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
        if not rows: raise ValueError('trim table is empty')
        self.rows=rows
        self.grid={(float(r['nacelle_deg']),float(r['V_ms'])):r for r in rows}
    def controls(self,gamma_deg,V_ms,rpm=None):
        speeds=sorted({k[1] for k in self.grid})
        Vn=min(speeds,key=lambda x:abs(x-V_ms))
        angles=sorted({k[0] for k in self.grid if abs(k[1]-Vn)<1e-8})
        if len(angles)==1:
            r=self.grid[(angles[0],Vn)]
            return {k:float(r[k]) for k in ('pitch_deg','collective_deg','theta_1c_lateral_deg','theta_1s_longitudinal_deg','elevator_deg','rudder_deg')}
        vals={k:np.array([float(self.grid[(g,Vn)][k]) for g in angles]) for k in ('pitch_deg','collective_deg','theta_1c_lateral_deg','theta_1s_longitudinal_deg','elevator_deg','rudder_deg')}
        g=np.array(angles,float)
        return {k:float(np.interp(gamma_deg,g,vals[k])) for k in vals}

@dataclass
class Segment:
    name:str
    duration_s:float
    airspeed_ms:tuple|list|float
    altitude_m:tuple|list|float
    nacelle_angle_deg:tuple|list|float
    rpm:tuple|list|float
    climb_rate_ms:float=0.0
    wind_ms:float=0.0
    control_source:str='scheduled'

class MissionPlannerV2:
    def __init__(self,trim_csv=None,radial_stations=20,n_azimuth=16):
        self.model=TransitionAircraftModel(radial_stations=radial_stations,n_azimuth=n_azimuth)
        self.lookup=TrimControlLookup(trim_csv) if trim_csv else None
    def _schedule(self,value,duration):
        if isinstance(value,(list,tuple)):
            if len(value)==2:return make_linear_schedule(value[0],value[1],duration)
            return PiecewiseSchedule([(duration*i/(len(value)-1),value[i]) for i in range(len(value))])
        return PiecewiseSchedule([(0,value)])
    def _controls(self,source,gamma,V,rpm,segment,tau):
        if source=='trim_lookup':
            if self.lookup is None: raise ValueError('trim_lookup requested but no trim table supplied')
            c=self.lookup.controls(gamma,V,rpm)
            c.setdefault('aileron_deg',0.0)
            return c
        if source=='scheduled':
            raw=segment.get('controls')
            if not isinstance(raw,dict): raise ValueError('scheduled control source requires a controls dictionary')
            out={}
            for name,val in raw.items(): out[name]=self._schedule(val,float(segment['duration_s']))(tau)
            return out
        if source=='zero':
            return {'pitch_deg':0.0,'collective_deg':20.0,'theta_1c_lateral_deg':0.0,'theta_1s_longitudinal_deg':0.0,'elevator_deg':0.0,'rudder_deg':0.0,'aileron_deg':0.0}
        raise ValueError(f'unsupported control source {source!r}')
    def _active_constraint(self,state,loads,enforce_trim=False):
        rot=loads['rotor'];wing=loads['wing']
        scaled=np.array([loads[k] for k in ('FX','FY','FZ','MX','MY','MZ')])/RESIDUAL_SCALES
        if enforce_trim and np.max(np.abs(scaled))>0.02:return 'trim residual'
        if abs(loads['MX'])>100:return 'symmetry/roll moment'
        if rot['reverse_flow_fraction']>0:return 'reverse flow'
        if rot['stall_span']>ROTOR.stall_span_fraction_limit:return 'rotor stall'
        if rot['M_adv']>ROTOR.tip_mach_limit:return 'advancing-tip Mach'
        wing_stall_active = (state.nacelle_angle_deg <= 60.0) or (state.airspeed_ms >= 25.0)
        if wing_stall_active and wing['wing_stall']:return 'wing stall'
        if loads['P_required_W']>loads['P_available_W']:return 'power limitation'
        if state.collective_deg<ROTOR.collective_hover_deg_min or state.collective_deg>ROTOR.collective_cruise_deg_max:return 'collective limit'
        if abs(state.theta_1c_deg)>ROTOR.cyclic_deg_limit or abs(state.theta_1s_deg)>ROTOR.cyclic_deg_limit:return 'cyclic limit'
        if abs(state.elevator_deg)>AIRCRAFT.elevator_deg_limit or abs(state.rudder_deg)>AIRCRAFT.rudder_deg_limit:return 'control-surface limit'
        return ''
    def validate_mission(self, mission):
        if not isinstance(mission,dict) or not mission.get('segments'):
            raise ValueError('mission must be a dict containing a non-empty segments list')
        for i,seg in enumerate(mission['segments']):
            if float(seg.get('duration_s',0.0)) <= 0:
                raise ValueError(f'segment {i} has non-positive duration')
            for key in ('airspeed_ms','altitude_m','nacelle_angle_deg','rpm'):
                if key not in seg: raise ValueError(f'segment {i} missing {key}')
            if isinstance(seg['nacelle_angle_deg'],(list,tuple)):
                vals=seg['nacelle_angle_deg']
            else: vals=[seg['nacelle_angle_deg']]
            if any((v< AIRCRAFT.nacelle_min_deg or v> AIRCRAFT.nacelle_max_deg) for v in vals):
                raise ValueError(f'segment {i} nacelle angle outside [0,90] deg')
            if any(v <= 0 for v in (seg['rpm'] if isinstance(seg['rpm'],(list,tuple)) else [seg['rpm']])):
                raise ValueError(f'segment {i} RPM must be positive')

    def run(self,mission,dt=None):
        self.validate_mission(mission)
        dt=float(dt if dt is not None else mission.get('dt_s',1.0))
        if dt <= 0: raise ValueError('dt must be positive')
        fuel=float(mission.get('initial_fuel_kg',AIRCRAFT.fuel_capacity_kg))
        reserve=AIRCRAFT.fuel_capacity_kg*AIRCRAFT.reserve_fuel_fraction
        gross=float(mission.get('gross_weight_kg',AIRCRAFT.gross_weight_kg))
        wind_default=float(mission.get('wind_ms',0.0))
        t_global=0.0;history=[];failure=None
        for seg in mission['segments']:
            duration=float(seg['duration_s'])
            schV=self._schedule(seg['airspeed_ms'],duration);schH=self._schedule(seg['altitude_m'],duration)
            schG=self._schedule(seg['nacelle_angle_deg'],duration);schR=self._schedule(seg['rpm'],duration)
            climb=float(seg.get('climb_rate_ms',0.0));wind=float(seg.get('wind_ms',wind_default))
            source=seg.get('control_source','scheduled');enforce_trim=bool(seg.get('enforce_trim',False))
            nsteps=int(np.ceil(duration/dt))
            for istep in range(nsteps+1):
                tau=min(istep*dt,duration)
                V=schV(tau);h=schH(tau);gamma=schG(tau);rpm=schR(tau)
                controls=self._controls(source,gamma,V,rpm,seg,tau)
                state=AircraftState(
                    airspeed_ms=V,altitude_m=h,pitch_deg=controls.get('pitch_deg',0.0),
                    nacelle_angle_deg=gamma,rpm=rpm,collective_deg=controls['collective_deg'],
                    theta_1c_deg=controls.get('theta_1c_lateral_deg',0.0),
                    theta_1s_deg=controls.get('theta_1s_longitudinal_deg',0.0),
                    elevator_deg=controls.get('elevator_deg',0.0),rudder_deg=controls.get('rudder_deg',0.0),
                    aileron_deg=controls.get('aileron_deg',0.0),climb_rate_ms=climb,wind_ms=wind,
                    gross_weight_kg=gross,dT_isa=0.0,
                )
                loads=self.model.loads(state)
                active=self._active_constraint(state,loads,enforce_trim=enforce_trim)
                scaled=np.array([loads[k] for k in ('FX','FY','FZ','MX','MY','MZ')])/RESIDUAL_SCALES
                trim_res=float(np.max(np.abs(scaled)))
                feasible=(active=='' and fuel>=reserve-1e-9)
                power_kW=loads['P_required_W']/1000.0
                burn=0.0 if tau>=duration else AIRCRAFT.sfc_kg_per_kWh*power_kW*(dt/3600.0)
                record={
                    'segment':seg['name'],'time_s':t_global+tau,'altitude_m':h,'airspeed_ms':V,
                    'ground_speed_ms':V+wind,'nacelle_angle_deg':gamma,'rpm':rpm,'pitch_deg':state.pitch_deg,
                    'collective_deg':state.collective_deg,'theta_1c_lateral_deg':state.theta_1c_deg,
                    'theta_1s_longitudinal_deg':state.theta_1s_deg,'elevator_deg':state.elevator_deg,
                    'rudder_deg':state.rudder_deg,'FX_N':loads['FX'],'FY_N':loads['FY'],'FZ_N':loads['FZ'],
                    'MX_Nm':loads['MX'],'MY_Nm':loads['MY'],'MZ_Nm':loads['MZ'],
                    'P_required_kW':power_kW,'P_available_kW':loads['P_available_W']/1000.0,
                    'fuel_before_kg':fuel,'fuel_burn_kg':burn,'fuel_remaining_kg':max(0.0,fuel-burn),
                    'gross_weight_kg':gross,'stall_margin_pct':max(0.0,(ROTOR.stall_span_fraction_limit-loads['rotor']['stall_span'])*100.0),
                    'tip_mach_margin':ROTOR.tip_mach_limit-loads['rotor']['M_adv'],
                    'reverse_flow_pct':loads['rotor']['reverse_flow_fraction']*100.0,
                    'trim_residual':trim_res,'feasible':feasible,'active_constraint':active,
                }
                history.append(record)
                if not feasible and failure is None:
                    failure={'segment':seg['name'],'time_s':record['time_s'],'reason':active or 'fuel reserve'}
                    break
                fuel=max(0.0,fuel-burn);gross=max(0.0,gross-burn)
            t_global+=duration
            if failure is not None:break
        return {'success':failure is None,'failure':failure,'history':history,'final_fuel_kg':fuel,'final_gross_weight_kg':gross}

def save_history_csv(history,path):
    if not history:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0].keys()));w.writeheader();w.writerows(history)
