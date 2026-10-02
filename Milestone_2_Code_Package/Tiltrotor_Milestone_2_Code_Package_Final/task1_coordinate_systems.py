"""Task 1: coordinate systems, transformations, sign conventions, and assumptions.

Outputs support M2 report Sections 1.1-1.3. All numerical aircraft/rotor values
are imported from aircraft_config.py so report and code cannot drift apart.
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib.pyplot as plt
from aircraft_config import AIRCRAFT, ROTOR, config_dict, save_json

ROOT=os.path.dirname(os.path.abspath(__file__))
FIG=os.path.join(ROOT,"figures"); OUT=os.path.join(ROOT,"outputs"); IN=os.path.join(ROOT,"inputs")
os.makedirs(FIG,exist_ok=True); os.makedirs(OUT,exist_ok=True); os.makedirs(IN,exist_ok=True)

def arrow(ax,p,q,label,offset=(0.03,0.03)):
    ax.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='->',lw=2))
    ax.text(q[0]+offset[0],q[1]+offset[1],label,fontsize=9)

def make_coordinate_figure():
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for ax in axs.flat:
        ax.set_aspect('equal'); ax.axis('off')
    ax=axs[0,0]; ax.set_title('Inertial / navigation frame I (NED)')
    arrow(ax,(0,0),(0.9,0),'$+X_I$ North'); arrow(ax,(0,0),(0,-0.9),'$+Z_I$ Down'); arrow(ax,(0,0),(0.55,0.6),'$+Y_I$ East')
    ax.text(-0.95,-0.9,'Origin: earth-fixed reference\nNED convention; +Z down',fontsize=9)
    ax=axs[0,1]; ax.set_title('Aircraft body frame B')
    arrow(ax,(0,0),(0.9,0),'$+X_B$ forward'); arrow(ax,(0,0),(0.0,-0.9),'$+Z_B$ down'); arrow(ax,(0,0),(0.55,0.6),'$+Y_B$ starboard')
    ax.text(-0.95,-0.9,'Origin: aircraft CG\nright-handed; +Z down',fontsize=9)
    ax=axs[0,2]; ax.set_title('Rotor-shaft frame R')
    g=np.radians(50); z=np.array([0.75*np.cos(g),-0.75*np.sin(g)]); x=np.array([-0.75*np.sin(g),-0.75*np.cos(g)]); y=np.array([0.55,0.35])
    arrow(ax,(0,0),z,'$+z_R$ thrust'); arrow(ax,(0,0),x,'$+x_R$'); arrow(ax,(0,0),y,'$+y_R$')
    ax.text(-1.0,-0.9,'Nacelle angle $\\gamma$ rotates shaft: 90° helicopter,\n0° airplane. Rotor +z is thrust direction.',fontsize=9)
    ax=axs[1,0]; ax.set_title('Tip-path plane / blade azimuth')
    t=np.linspace(0,2*np.pi,200); ax.plot(np.cos(t),np.sin(t),lw=1.3)
    arrow(ax,(0,0),(0.8,0),'$+x_{TPP}$'); arrow(ax,(0,0),(0,0.8),'$+y_{TPP}$')
    ax.text(-0.95,-0.9,'Rigid-disk model: $\\beta=0$ so TPP = shaft plane.\n$\\psi$ increases with rotor rotation; ψ=90° is advancing.',fontsize=9)
    ax=axs[1,1]; ax.set_title('Blade-element local frame')
    arrow(ax,(0,0),(0.85,0),'$+t$ tangential'); arrow(ax,(0,0),(0.0,0.85),'$+n$ shaft-normal'); arrow(ax,(0,0),(-0.6,0.35),'$+r$ radial')
    ax.text(-0.95,-0.9,'$U_T$: in-plane velocity; $U_P$: through-disk velocity.\n$\\phi=atan2(U_P,U_T)$; $\\alpha=\\theta-\\phi$.',fontsize=9)
    ax=axs[1,2]; ax.set_title('Transformation chain to aircraft loads')
    boxes=[('Earth / I','attitude','Body / B'),('Body / B','nacelle rotation γ','Rotor / R'),('Rotor / R','azimuth ψ + blade kinematics','element'),('Rotor forces + moments','DCM $C_{BR}$','Body loads'),('Hub loads','r × F','Moments about CG')]
    y=1.0
    for a,l,b in boxes:
        ax.text(-0.95,y,a,ha='left',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.35',fc='white',ec='black'))
        ax.text(0,y,l,ha='center',va='center',fontsize=8)
        ax.text(0.95,y,b,ha='right',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.35',fc='white',ec='black'))
        y-=0.43
    fig.suptitle('Milestone 2 Task 1 — coordinate systems and transformation conventions',fontsize=15)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(FIG,'task1_coordinate_frames.png'),dpi=180); plt.close(fig)

def write_outputs():
    rows=[
      ('1.1','Inertial frame','NED: +X_I forward, +Y_I right, +Z_I down','NED inertial convention; altitude is measured along -Z_I.'),
      ('1.1','Aircraft body frame','Origin at CG; +X_B forward, +Y_B starboard, +Z_B down','All aircraft loads and moments use this frame.'),
      ('1.1','Rotor-shaft frame','+z_R along rotor thrust; γ=90° helicopter and γ=0° airplane','Rotor force/moment is transformed with a direction-cosine matrix.'),
      ('1.1','Tip-path plane','TPP = shaft plane because β=0','Explicit rigid-disk approximation; no flapping dynamics.'),
      ('1.1','Blade-element frame','r/t/n local radial, tangential and shaft-normal directions','Used to convert sectional lift/drag into thrust/tangential loads.'),
      ('1.1','Azimuth','ψ=0 reference blade; ψ increases in rotor rotation sense; ψ=90° advancing','Same convention is used in code, plots, and lecture pitch law.'),
      ('1.1','Cyclic pitch','θ=θ0+θ1c cosψ+θ1s sinψ; θ1c lateral, θ1s longitudinal','Matches AE 667 Blade Dynamics lecture convention.'),
      ('1.2','Flapping','Rigid disk, β=0','Allowed approximation for this milestone; removes flapping velocity term.'),
      ('1.2','Reverse flow','U_T<0 flagged; default policy excludes those elements from forward-flow integration','Avoids extrapolating forward-flow polar into reverse flow.'),
      ('1.2','Stall','|α|>12° flagged; 5% span warning and 25% span hard screening limit','Linear airfoil model cannot model post-stall physics.'),
      ('1.2','Compressibility','Advancing-tip Mach monitored with M_adv ≤ 0.72','No high-Mach correction applied to adopted linear polar.'),
      ('1.2','Inflow','M2 handout: λ_i/λ_i,Glauert = 1 + [(4/3)(μ/λ_G)/(1.2+μ/λ_G)](r/R)cosψ','λ_G is total inflow ratio; only induced component is azimuthally redistributed.'),
      ('1.3','Mass budget',f'{AIRCRAFT.empty_weight_kg:.0f} empty + {AIRCRAFT.max_payload_kg:.0f} payload + {AIRCRAFT.fuel_capacity_kg:.0f} fuel = {AIRCRAFT.gross_weight_kg:.0f} kg','Single source of truth fixes the M1 grading inconsistency.'),
      ('1.3','Wing',f'S={AIRCRAFT.wing_area_m2:.1f} m², b={AIRCRAFT.wing_span_m:.1f} m, AR={AIRCRAFT.wing_span_m**2/AIRCRAFT.wing_area_m2:.2f}, incidence={AIRCRAFT.wing_incidence_deg:.1f}°','Preliminary lift/drag model for transition.'),
      ('1.3','Empennage',f'HT S={AIRCRAFT.htail_area_m2:.1f} m², arm={AIRCRAFT.htail_arm_m:.1f} m; VT S={AIRCRAFT.vtail_area_m2:.1f} m², arm={AIRCRAFT.vtail_arm_m:.1f} m','Provides pitch/yaw control authority for trim.'),
      ('1.3','Fuselage/nacelles',f'Equivalent flat plate={AIRCRAFT.fuselage_flat_plate_m2+2*AIRCRAFT.nacelle_flat_plate_each_m2:.2f} m²','Parasite drag only; rotor wake/interference omitted.'),
      ('1.3','Rotor placement',f'Hub r_B = ({AIRCRAFT.rotor_hub_x_m:.2f}, ±{AIRCRAFT.rotor_half_span_m:.2f}, {AIRCRAFT.rotor_hub_z_m:.2f}) m','CG placement moment r×F is included.'),
      ('1.3','Control limits',f'Collective {ROTOR.collective_hover_deg_min:.0f}–{ROTOR.collective_cruise_deg_max:.0f}°, cyclic ±{ROTOR.cyclic_deg_limit:.0f}°, elevator/rudder ±{AIRCRAFT.elevator_deg_limit:.0f}°, aileron ±{AIRCRAFT.aileron_deg_limit:.0f}°','Used consistently by trim and mission feasibility checks.'),
    ]
    with open(os.path.join(OUT,'task1_assumptions.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['Section','Item','Value / method','Justification / impact']); w.writerows(rows)
    with open(os.path.join(OUT,'task1_coordinate_conventions.md'),'w',encoding='utf-8') as f:
        f.write('# Task 1 — Coordinate systems, transformations and conventions\n\n')
        f.write('The aircraft uses a right-handed NED inertial frame and body frame. Positive pitch is nose-up. Rotor +z_R points in the thrust direction. The nacelle angle γ is 90° in helicopter mode and 0° in airplane mode. The positive azimuth direction follows the rotor rotation sense and ψ=90° is the advancing side for the representative rotor.\n\n')
        f.write('The course cyclic convention is θ(ψ)=θ0+θ1c cosψ+θ1s sinψ. In the AE 667 lecture θ1c denotes lateral cyclic and θ1s denotes longitudinal cyclic.\n\n')
        f.write('The rotor-frame force vector is transformed to body axes with C_BR. The hub moment reported about the aircraft CG is M_CG = C_BR M_hub,R + r_hub × F_body. For the symmetric twin-rotor aircraft, the two rotors use opposite rotation signs so shaft reaction torque cancels in steady flight.\n')
    save_json(os.path.join(IN,'aircraft_config.json'))

if __name__=='__main__':
    make_coordinate_figure(); write_outputs(); print('Task 1 complete.')
