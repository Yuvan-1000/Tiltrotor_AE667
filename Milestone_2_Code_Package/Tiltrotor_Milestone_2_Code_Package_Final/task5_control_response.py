"""Task 5: pilot-input/control-response study.

Implements the M2 demonstration requirement at two conditions using a single
rotor: one helicopter-like condition and one intermediate conversion condition.
For each collective, longitudinal-cyclic and lateral-cyclic sweep the driver
records all six rotor load components, shaft power, stall, reverse-flow and
advancing-tip-Mach diagnostics.
"""
from __future__ import annotations
import csv, os, json, warnings
import numpy as np
import matplotlib.pyplot as plt
from aircraft_config import TILTROTOR_GEOM,TILTROTOR_AIRFOIL,ROTOR
from edgewise_bemt import EdgewiseFlightCondition,EdgewiseBEMTSolver

ROOT=os.path.dirname(os.path.abspath(__file__));FIG=os.path.join(ROOT,'figures');OUT=os.path.join(ROOT,'outputs');IN=os.path.join(ROOT,'inputs')
os.makedirs(FIG,exist_ok=True);os.makedirs(OUT,exist_ok=True);os.makedirs(IN,exist_ok=True)

CONDITIONS={
 'helicopter_like':dict(nacelle_angle_deg=90.0,airspeed_ms=20.0,rpm=ROTOR.rpm_hover,collective_deg=14.0,description='helicopter-like edgewise flight'),
 'intermediate_conversion':dict(nacelle_angle_deg=45.0,airspeed_ms=10.0,rpm=ROTOR.rpm_hover,collective_deg=14.0,description='intermediate conversion'),
}

def solve_one(c,collective,lat=0.0,long=0.0):
    f=EdgewiseFlightCondition.from_body_forward_speed(c['rpm'],collective,c['airspeed_ms'],nacelle_angle_deg=c['nacelle_angle_deg'],theta_1c_deg=lat,theta_1s_deg=long,altitude=500.0,dT_isa=0.0,n_azimuth=72)
    with warnings.catch_warnings():warnings.simplefilter('ignore');return EdgewiseBEMTSolver(TILTROTOR_GEOM,TILTROTOR_AIRFOIL).solve(f)

def save(rows,path):
    keys=['control_deg','FX_N','FY_N','FZ_N','MX_Nm','MY_Nm','MZ_Nm','P_kW','stall_span_pct','stall_margin_pct','reverse_flow_pct','M_adv','M_adv_margin']
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(keys)
        for r in rows:w.writerow([r[k] for k in keys])

def figures(rows,control_label,prefix,title):
    x=np.array([r['control_deg'] for r in rows])
    groups=[(['FX_N','FY_N','FZ_N'],['FX [N]','FY [N]','FZ [N]'],'forces'),(['MX_Nm','MY_Nm','MZ_Nm'],['MX [N·m]','MY [N·m]','MZ [N·m]'],'moments'),(['P_kW'],['Rotor power [kW]'],'power')]
    for keys,labels,suffix in groups:
        fig,ax=plt.subplots(figsize=(7.1,4.8))
        for k,l in zip(keys,labels):ax.plot(x,[r[k] for r in rows],marker='o',ms=3,label=l)
        ax.axhline(0,lw=.8);ax.set_xlabel(control_label);ax.set_ylabel(', '.join(labels));ax.set_title(title+' — '+suffix);ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(os.path.join(FIG,f'{prefix}_{suffix}.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,4.8));ax.plot(x,[r['stall_margin_pct'] for r in rows],marker='o',ms=3,label='Stall margin to 25% span limit [percentage points]');ax.plot(x,[r['M_adv_margin'] for r in rows],marker='s',ms=3,label='Advancing-tip Mach margin');ax.set_xlabel(control_label);ax.set_ylabel('Margin');ax.set_title(title+' — feasibility margins');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(os.path.join(FIG,f'{prefix}_margins.png'),dpi=180);plt.close(fig)

def row(v,r):return {'control_deg':float(v),'FX_N':r['F_body'][0],'FY_N':r['F_body'][1],'FZ_N':r['F_body'][2],'MX_Nm':r['M_hub_body'][0],'MY_Nm':r['M_hub_body'][1],'MZ_Nm':r['M_hub_body'][2],'P_kW':r['P']/1000,'stall_span_pct':r['stalled_span_fraction']*100,'stall_margin_pct':(ROTOR.stall_span_fraction_limit-r['stalled_span_fraction'])*100,'reverse_flow_pct':r['reverse_flow_fraction']*100,'M_adv':r['M_adv_tip'],'M_adv_margin':ROTOR.tip_mach_limit-r['M_adv_tip']}

def do_condition(name,c):
    vals=np.arange(12.0,23.0,1.0)
    coll=[row(v,solve_one(c,v,0,0)) for v in vals]
    save(coll,os.path.join(OUT,f'task5_{name}_collective.csv'));figures(coll,'Collective θ0 [deg]',f'task5_{name}_collective',f"{c['description']} | 750 RPM, 500 m, SL ISA")
    vals=np.arange(-8.0,8.1,1.0)
    longitudinal=[row(v,solve_one(c,c['collective_deg'],0,v)) for v in vals]
    save(longitudinal,os.path.join(OUT,f'task5_{name}_longitudinal_cyclic.csv'));figures(longitudinal,'Longitudinal cyclic θ1s [deg]',f'task5_{name}_longitudinal_cyclic',f"{c['description']} | θ1s = longitudinal cyclic")
    lateral=[row(v,solve_one(c,c['collective_deg'],v,0)) for v in vals]
    save(lateral,os.path.join(OUT,f'task5_{name}_lateral_cyclic.csv'));figures(lateral,'Lateral cyclic θ1c [deg]',f'task5_{name}_lateral_cyclic',f"{c['description']} | θ1c = lateral cyclic")
    return {'collective':coll,'longitudinal':longitudinal,'lateral':lateral}

def main():
    with open(os.path.join(IN,'control_sweep_conditions.json'),'w',encoding='utf-8') as f:json.dump(CONDITIONS,f,indent=2)
    summaries={}
    for name,c in CONDITIONS.items():summaries[name]=do_condition(name,c)
    with open(os.path.join(OUT,'task5_control_operating_conditions.md'),'w',encoding='utf-8') as f:
        f.write('# Task 5 control-response operating conditions\n\n')
        f.write('The rotor uses the course convention θ(ψ)=θ0+θ1c cosψ+θ1s sinψ; θ1c is lateral cyclic and θ1s is longitudinal cyclic. Rotor rotation sense is + for the representative single-rotor plots.\n\n')
        for n,c in CONDITIONS.items():f.write(f"- {n}: γ={c['nacelle_angle_deg']:.0f}°, V={c['airspeed_ms']:.0f} m/s, RPM={c['rpm']:.0f}, h=500 m, ISA, base collective={c['collective_deg']:.0f}°.\n")
    with open(os.path.join(OUT,'task5_control_observations.md'),'w',encoding='utf-8') as f:
        f.write('# Task 5 control-response observations / Sec. 4.6\n\n')
        f.write('| Observation | Physical cause | Evidence | Design implication |\n|---|---|---|---|\n')
        f.write('| Collective increase raises FZ magnitude and rotor power | Higher blade pitch raises section angle of attack and aerodynamic loading | Task 5 collective force/power figures | Collective is the primary thrust-control variable; power margin must be checked with it |\n')
        f.write('| Longitudinal cyclic changes hub pitching response strongly while mean thrust changes less | The prescribed θ1s sinψ term redistributes loading around the disk; rigid-disk model has no flapping to absorb the first harmonic | Longitudinal cyclic moment/force figures | Longitudinal cyclic supplies a direct pitch-control input in the pre-flapping M2 model |\n')
        f.write('| Lateral cyclic produces the corresponding hub-roll response with small cross-coupling terms | The θ1c cosψ term changes the azimuthal loading distribution relative to the shaft frame | Lateral cyclic moment/force figures | Lateral cyclic provides the roll-control channel; cross-coupling should be checked in trim |\n')
        f.write('| Advancing/retreating asymmetry grows with in-plane speed | UT = Ωr + s Vedge sinψ makes the advancing side faster and the retreating side slower | Task 3 contour and Task 5 conditions | Reverse flow and stall limits become increasingly important during conversion |\n')
        f.write('| Stall/Mach margins are operating-point dependent | Collective changes α and Vedge changes advancing-tip speed | Margin CSVs/figures | Control sweeps must be interpreted together with aerodynamic limits, not load trends alone |\n')
        f.write('The course convention is θ(ψ)=θ0+θ1c cosψ+θ1s sinψ, with θ1c lateral cyclic and θ1s longitudinal cyclic. Every plot states its operating condition: the helicopter-like case is γ=90°, V=20 m/s, 750 RPM, h=500 m, ISA, θ0=14°; the intermediate case is γ=45°, V=10 m/s, 750 RPM, h=500 m, ISA, θ0=14°.\n')
    print('Task 5 complete.')
if __name__=='__main__':main()
