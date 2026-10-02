"""Task 8: speed--nacelle-angle conversion corridor.

The required 3x3 steady-trim matrix from Task 7 is the seed dataset.  A
resolved map is produced by interpolating/extrapolating its bounded control
solution onto a denser speed-angle grid, then evaluating the same aircraft
model at every point. This avoids independently solving hundreds of trim
problems and makes the source of the map explicit.
"""
from __future__ import annotations
import csv, json, os, warnings
import numpy as np
import matplotlib.pyplot as plt
from aircraft_config import AIRCRAFT,ROTOR
from bemt_solver import Atmosphere
from aircraft_model import TransitionAircraftModel,AircraftState

ROOT=os.path.dirname(os.path.abspath(__file__));OUT=os.path.join(ROOT,'outputs');FIG=os.path.join(ROOT,'figures');IN=os.path.join(ROOT,'inputs')
os.makedirs(OUT,exist_ok=True);os.makedirs(FIG,exist_ok=True);os.makedirs(IN,exist_ok=True)
TRIM=os.path.join(OUT,'task7_trim_results.csv')
ANGLES=np.arange(0.0,91.0,5.0); SPEEDS=np.arange(10.0,46.0,2.0)

def load_trim():
    rows=list(csv.DictReader(open(TRIM,encoding='utf-8')))
    if len(rows)<9: raise RuntimeError('Task 8 requires the Task 7 3x3 trim results first.')
    names=['pitch_deg','collective_deg','theta_1c_lateral_deg','theta_1s_longitudinal_deg','elevator_deg','rudder_deg','aileron_deg']
    data={(float(r['nacelle_deg']),float(r['V_ms'])):np.array([float(r[n]) for n in names]) for r in rows}
    return data,names

def nearest_bilinear(data,g,V):
    gs=sorted({k[0] for k in data});vs=sorted({k[1] for k in data})
    g=np.clip(g,min(gs),max(gs));V=np.clip(V,min(vs),max(vs))
    g0=max(x for x in gs if x<=g);g1=min(x for x in gs if x>=g);v0=max(x for x in vs if x<=V);v1=min(x for x in vs if x>=V)
    if g1==g0 and v1==v0:return data[(g0,v0)]
    if g1==g0:return data[(g0,v0)]*(v1-V)/(v1-v0)+data[(g0,v1)]*(V-v0)/(v1-v0)
    if v1==v0:return data[(g0,v0)]*(g1-g)/(g1-g0)+data[(g1,v0)]*(g-g0)/(g1-g0)
    q00=data[(g0,v0)];q01=data[(g0,v1)];q10=data[(g1,v0)];q11=data[(g1,v1)]
    a=(g-g0)/(g1-g0);b=(V-v0)/(v1-v0)
    return (1-a)*(1-b)*q00+(1-a)*b*q01+a*(1-b)*q10+a*b*q11

def classify(loads,x,g,V):
    o=loads;rot=o['rotor'];wing=o['wing']
    residual=np.max(np.abs(np.array([o[k] for k in ('FX','FY','FZ','MX','MY','MZ')]) / np.array([15000,5000,15000,10000,10000,10000])))
    if rot['reverse_flow_fraction']>1e-12:return 'reverse flow'
    if rot['stall_span']>ROTOR.stall_span_fraction_limit:return 'rotor stall'
    if rot['M_adv']>ROTOR.tip_mach_limit:return 'advancing-tip Mach'
    wing_stall_active=(g<=60.0) or (V>=25.0)
    if wing_stall_active and wing['wing_stall']:return 'wing stall'
    if o['P_required_W']>o['P_available_W']:return 'power limitation'
    if abs(x[1]-np.clip(x[1],ROTOR.collective_hover_deg_min,ROTOR.collective_cruise_deg_max))>1e-8:return 'control saturation'
    if np.any(np.abs(x[2:4])>ROTOR.cyclic_deg_limit+1e-8):return 'control saturation'
    if abs(x[4])>AIRCRAFT.elevator_deg_limit or abs(x[5])>AIRCRAFT.rudder_deg_limit or abs(x[6])>AIRCRAFT.aileron_deg_limit:return 'control saturation'
    if residual>0.02:return 'excessive trim residual'
    return 'feasible'

def run_map():
    data,names=load_trim();model=TransitionAircraftModel(radial_stations=10,n_azimuth=8)
    rows=[];matrix=[];codes={'feasible':0,'excessive trim residual':1,'control saturation':2,'wing stall':3,'power limitation':4,'advancing-tip Mach':5,'reverse flow':6,'rotor stall':7}
    for g in ANGLES:
        row=[]
        for V in SPEEDS:
            x=nearest_bilinear(data,g,V);rpm=ROTOR.rpm_hover if g>=60 else ROTOR.rpm_cruise
            s=AircraftState(V,500,x[0],g,rpm,x[1],x[2],x[3],x[4],x[5],x[6],0,0,AIRCRAFT.gross_weight_kg,0)
            with warnings.catch_warnings():warnings.simplefilter('ignore');loads=model.loads(s)
            status=classify(loads,x,g,V);row.append(codes[status])
            residual=float(np.max(np.abs(np.array([loads[k] for k in ('FX','FY','FZ','MX','MY','MZ')]) / np.array([15000,5000,15000,10000,10000,10000]))))
            rows.append([g,V,status,*x,loads['P_required_W']/1000,loads['P_available_W']/1000,loads['rotor']['stall_span']*100,loads['rotor']['reverse_flow_fraction']*100,loads['rotor']['M_adv'],residual])
        matrix.append(row)
    matrix=np.array(matrix)
    with open(os.path.join(OUT,'task8_speed_nacelle_corridor.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['nacelle_deg','V_ms','status',*names,'P_required_kW','P_available_kW','stall_span_pct','reverse_flow_pct','M_adv','max_scaled_residual']);w.writerows(rows)
    with open(os.path.join(OUT,'task8_map_settings.json'),'w',encoding='utf-8') as f:json.dump({'nacelle_angles_deg':ANGLES.tolist(),'airspeeds_ms':SPEEDS.tolist(),'altitude_m':500.0,'gross_weight_kg':AIRCRAFT.gross_weight_kg,'dT_isa_K':0.0,'seed':'Task 7 3x3 trim matrix','map_method':'bounded bilinear interpolation in gamma-speed, then direct aircraft-model evaluation'},f,indent=2)
    labels=list(codes)
    fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(matrix,origin='lower',aspect='auto',extent=[SPEEDS.min(),SPEEDS.max(),ANGLES.min(),ANGLES.max()],vmin=-.5,vmax=len(labels)-.5,cmap='tab10');ax.set_xlabel('Airspeed [m/s]');ax.set_ylabel('Nacelle angle γ [deg] (0° airplane, 90° helicopter)');ax.set_title('Task 8 / Sec. 7.1 — speed–nacelle-angle feasibility map\n500 m, 3000 kg, ISA; controls seeded from 3×3 trim matrix');fig.colorbar(im,ax=ax,ticks=range(len(labels)),label='Map status');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(os.path.join(FIG,'task8_speed_nacelle_feasibility_map.png'),dpi=180);plt.close(fig)
    # Analytical hard-boundary overlays + evaluated constraint mask.
    fig,ax=plt.subplots(figsize=(10,6));feas=(matrix==0).astype(float);ax.imshow(feas,origin='lower',aspect='auto',extent=[SPEEDS.min(),SPEEDS.max(),ANGLES.min(),ANGLES.max()],vmin=0,vmax=1,cmap='Greys',alpha=.65)
    for g in np.linspace(1,90,180):
        gr=np.radians(g);rpm=ROTOR.rpm_hover if g>=60 else ROTOR.rpm_cruise;Om=rpm*2*np.pi/60;Vtip=Om*ROTOR.R;rho,*_,a,_mu=Atmosphere(500.0,0.0).properties()
        vrf=Om*ROTOR.r_root/max(np.sin(gr),1e-9);vmach=(ROTOR.tip_mach_limit*a-Vtip)/max(np.sin(gr),1e-9)
        if 10<=vrf<=45:ax.plot(vrf,g,'r.',ms=1.5)
        if 10<=vmach<=45:ax.plot(vmach,g,'b.',ms=1.5)
    ax.set_xlabel('Airspeed [m/s]');ax.set_ylabel('Nacelle angle γ [deg]');ax.set_title('Task 8 / Sec. 7.2 — active-constraint boundaries\nGrey = candidate-control feasible; red = reverse-flow onset; blue = M_adv=0.72');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(os.path.join(FIG,'task8_active_constraint_boundaries.png'),dpi=180);plt.close(fig)
    counts={s:int(sum(r[2]==s for r in rows)) for s in labels}
    path=[(90.0,8.0),(75.0,10.0),(60.0,15.0),(50.0,18.0),(45.0,20.0),(30.0,25.0),(20.0,28.0),(10.0,32.0),(0.0,35.0)]
    with open(os.path.join(OUT,'task8_operational_transition_path.md'),'w',encoding='utf-8') as f:
        f.write('# Task 8 / Sec. 7.1 — transition path used by Task 9\n\n')
        f.write('Candidate path nodes (γ°, V m/s): '+', '.join(f'({g:.0f}, {v:.0f})' for g,v in path)+'\n\n')
        f.write('These nodes define the scheduled conversion corridor exercised in Task 9. They are not claimed to be nine independently converged trim solutions; Task 9 validates the scheduled mission trajectory against the same aerodynamic and feasibility checks at every time step.\n')

    with open(os.path.join(OUT,'task8_active_constraint_summary.md'),'w',encoding='utf-8') as f:
        f.write('# Task 8 / Sec. 7 — corridor summary\n\n')
        f.write(f"Resolved grid: {len(ANGLES)} nacelle angles × {len(SPEEDS)} airspeeds = {len(rows)} states. All cases use h=500 m, gross mass 3000 kg, ISA and the same aircraft/rotor configuration.\n\n")
        for k,v in counts.items():f.write(f'- {k}: {v} states\n')
        f.write('\nThe map is deliberately labeled as a candidate-control feasibility map: the controls are seeded from the required 3×3 trim matrix and each grid point is checked with the aircraft model. A point labeled feasible is therefore a model-feasible interpolated control state, not an independently converged trim solution.\n')
    return rows

if __name__=='__main__':run_map();print('Task 8 complete.')
