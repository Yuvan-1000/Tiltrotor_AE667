"""Task 4: rotor forces and moments in rotor/body axes and about the CG.

This task-level driver makes the Task-2 rotor-load transformation explicit and
provides a compact check of counter-rotating torque cancellation on the twin
rotor aircraft.
"""
from __future__ import annotations
import csv, os, warnings
import numpy as np
import matplotlib.pyplot as plt
from aircraft_config import AIRCRAFT, ROTOR, TILTROTOR_GEOM, TILTROTOR_AIRFOIL
from edgewise_bemt import EdgewiseFlightCondition, EdgewiseBEMTSolver
from aircraft_model import AircraftState, TransitionAircraftModel

ROOT=os.path.dirname(os.path.abspath(__file__));FIG=os.path.join(ROOT,'figures');OUT=os.path.join(ROOT,'outputs')
os.makedirs(FIG,exist_ok=True);os.makedirs(OUT,exist_ok=True)

def run_conditions():
    solver=EdgewiseBEMTSolver(TILTROTOR_GEOM,TILTROTOR_AIRFOIL)
    cases=[
      ('helicopter_like',90.0,20.0,14.0,ROTOR.rpm_hover),
      ('intermediate_conversion',45.0,10.0,14.0,ROTOR.rpm_hover),
    ]
    rows=[]
    for name,g,V,theta,rpm in cases:
        for sign in (1,-1):
            hub_y=sign*AIRCRAFT.rotor_half_span_m
            f=EdgewiseFlightCondition.from_body_forward_speed(rpm,theta,V,nacelle_angle_deg=g,n_azimuth=72,hub_position_body=(AIRCRAFT.rotor_hub_x_m, hub_y, AIRCRAFT.rotor_hub_z_m),rotation_sign=sign)
            with warnings.catch_warnings():warnings.simplefilter('ignore');r=solver.solve(f)
            rows.append([name,sign,r['mu'],r['T'],r['Q'],r['P'],*r['F_rotor'],*r['M_hub_rotor'],*r['F_body'],*r['M_hub_body'],*r['M_body_reference'],r['M_adv_tip'],r['stalled_span_fraction'],r['reverse_flow_fraction']])
    with open(os.path.join(OUT,'task4_rotor_loads.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['condition','rotation_sign','mu','T_N','Q_Nm','P_W','Fx_R','Fy_R','Fz_R','Mx_R_hub','My_R_hub','Mz_R_hub','Fx_B','Fy_B','Fz_B','Mx_B_hub','My_B_hub','Mz_B_hub','Mx_B_CG','My_B_CG','Mz_B_CG','M_adv','stall_span','reverse_flow_fraction']);w.writerows(rows)
    model=TransitionAircraftModel(radial_stations=16,n_azimuth=16)
    states=[
      ('helicopter_like',90.0,20.0,0.0,14.0,ROTOR.rpm_hover),
      ('intermediate_conversion',45.0,10.0,0.0,14.0,ROTOR.rpm_hover),
    ]
    pair_rows=[]
    for name,g,V,pitch,theta,rpm in states:
        state=AircraftState(V,500,pitch,g,rpm,theta,0,0,0,0,0,0,0,AIRCRAFT.gross_weight_kg,0)
        loads=model.loads(state)
        pair_rows.append([name,g,V,*loads['rotor_force'],*loads['rotor_moment'],loads['rotor']['P_aero_W'],loads['rotor']['M_adv'],loads['rotor']['stall_span'],loads['rotor']['reverse_flow_fraction']])
    with open(os.path.join(OUT,'task4_twin_rotor_loads.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['condition','nacelle_deg','airspeed_ms','FX_rotor_pair_N','FY_rotor_pair_N','FZ_rotor_pair_N','MX_about_CG_Nm','MY_about_CG_Nm','MZ_about_CG_Nm','P_aero_pair_W','M_adv_max','stall_span_max','reverse_flow_fraction_max']);w.writerows(pair_rows)
    # Mz cancellation and body/CG force transformation plot.
    x=np.arange(2)
    fig,ax=plt.subplots(figsize=(7,4.6))
    labels=[]; plus=[]; minus=[]
    for cond in ['helicopter_like','intermediate_conversion']:
        rr=[r for r in rows if r[0]==cond]; plus.append(rr[0][20] if rr[0][1]==1 else rr[1][20]); minus.append(rr[1][20] if rr[0][1]==1 else rr[0][20]); labels.append(cond.replace('_',' '))
    ax.bar(x-.18,plus,width=.36,label='+ rotation');ax.bar(x+.18,minus,width=.36,label='− rotation');ax.axhline(0,lw=1);ax.set_xticks(x,labels);ax.set_ylabel('Rotor hub yaw moment Mz [N·m]');ax.set_title('Task 4 — counter-rotation reaction torque | 750 RPM, 500 m, ISA; γ=90°, V=20 and γ=45°, V=10 m/s');ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(os.path.join(FIG,'task4_counter_rotation_torque.png'),dpi=180);plt.close(fig)
    with open(os.path.join(OUT,'task4_force_moment_notes.md'),'w',encoding='utf-8') as f:
        f.write('# Task 4 — rotor forces and moments\n\n')
        f.write('The edgewise solver returns forces/moments in rotor coordinates, transforms them into aircraft body coordinates using the nacelle direction-cosine matrix, and then forms the moment about the CG as M_CG = M_hub + r_hub × F. The twin-rotor aircraft uses opposite rotation signs, so the steady shaft reaction torque cancels in the symmetric pair.\n')
    return rows

if __name__=='__main__':run_conditions();print('Task 4 complete.')
