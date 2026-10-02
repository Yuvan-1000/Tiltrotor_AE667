"""Task 6: preliminary wing, fuselage/nacelle and empennage model.

Generates the updated aircraft schematic and the model/configuration tables
needed by M2 Sections 5.1-5.5. The numerical model lives in aircraft_model.py;
this driver documents and exercises it.
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib.pyplot as plt
from aircraft_config import AIRCRAFT, ROTOR, TILTROTOR_GEOM
from aircraft_model import TransitionAircraftModel, AircraftState

ROOT=os.path.dirname(os.path.abspath(__file__));FIG=os.path.join(ROOT,'figures');OUT=os.path.join(ROOT,'outputs')
os.makedirs(FIG,exist_ok=True);os.makedirs(OUT,exist_ok=True)

def schematic():
    fig,axs=plt.subplots(1,2,figsize=(13,5.8))
    ax=axs[0];ax.set_aspect('equal');ax.set_xlim(-6.5,6.5);ax.set_ylim(-5.0,5.0);ax.axis('off');ax.set_title('Task 6 / Sec. 5.1 — updated plan view')
    ax.add_patch(plt.Rectangle((-4,-1),8,2,fill=False,lw=2));
    for y in (AIRCRAFT.rotor_half_span_m,-AIRCRAFT.rotor_half_span_m):
        ax.add_patch(plt.Circle((0,y),ROTOR.R,fill=False,linestyle='--',lw=1.2));ax.plot([0,0],[y-.35,y+.35],lw=4)
    ax.add_patch(plt.Polygon([[0,1],[3.5,3.0],[4.8,2.4],[1.5,.3]],closed=True,fill=False,lw=2));ax.add_patch(plt.Polygon([[0,-1],[3.5,-3],[4.8,-2.4],[1.5,-.3]],closed=True,fill=False,lw=2))
    ax.scatter([AIRCRAFT.cg_x_m],[AIRCRAFT.cg_y_m],s=40);ax.text(.15,.15,'CG');ax.text(-5.9,-4.5,f'Fuselage ≈10 m; wing span {AIRCRAFT.wing_span_m:.1f} m; rotor diameter {2*ROTOR.R:.1f} m',fontsize=8);ax.text(-5.9,-4.82,f'Rotor hubs: x={AIRCRAFT.rotor_hub_x_m:.2f} m, y=±{AIRCRAFT.rotor_half_span_m:.2f} m; CG at ({AIRCRAFT.cg_x_m:.1f},{AIRCRAFT.cg_y_m:.1f},{AIRCRAFT.cg_z_m:.1f}) m',fontsize=8)
    ax.annotate('',xy=(5.5,0),xytext=(3.8,0),arrowprops=dict(arrowstyle='->'));ax.text(5.55,.1,'+X_B forward',fontsize=9);ax.annotate('',xy=(0,4.4),xytext=(0,2.8),arrowprops=dict(arrowstyle='->'));ax.text(.1,4.4,'+Y_B starboard',fontsize=9)
    ax=axs[1];ax.set_aspect('equal');ax.set_xlim(-6,6);ax.set_ylim(-3.5,4.2);ax.axis('off');ax.set_title('Side view — nacelle conversion')
    ax.add_patch(plt.Rectangle((-3.8,-.8),7.2,1.6,fill=False,lw=2));ax.plot([-1.8,2.6],[.8,.8],lw=3);ax.plot([2.6,4.4],[.8,2.2],lw=2);ax.plot([2.6,4.2],[.8,-.3],lw=2)
    origin=(0,.8)
    for gamma,ls,label in [(90,'-','γ=90° helicopter'),(45,'--','γ=45° conversion'),(0,':','γ=0° airplane')]:
        g=np.radians(gamma);L=2.25;dx=L*np.cos(g);dz=-L*np.sin(g);ax.plot([origin[0],dx],[origin[1],origin[1]+dz],ls=ls,lw=2);ax.text(dx+.15,origin[1]+dz,label,fontsize=8)
    ax.scatter([AIRCRAFT.cg_x_m],[0],s=40);ax.text(.15,.15,'CG');ax.text(-5.5,-2.8,'γ rotates the rotor thrust axis from upward to forward',fontsize=8);ax.text(-5.5,-3.1,'Wing, horizontal tail, vertical tail, rudder/elevator/aileron modeled; rotor wake interference omitted',fontsize=8)
    fig.tight_layout();fig.savefig(os.path.join(FIG,'task6_updated_aircraft_schematic.png'),dpi=180);plt.close(fig)

def tables():
    with open(os.path.join(OUT,'task6_2_changes_from_m1.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['Change','M1 baseline','M2 value','Motivating issue','Expected benefit']);w.writerows([
          ['Rotor flow model','Axisymmetric axial BEMT','Azimuth-resolved edgewise BEMT','M2 forward-flight requirement','Advancing/retreating loading'],
          ['Nacelle state','End states only','Continuous 0–90°','M2 conversion requirement','Conversion corridor'],
          ['Rotor-to-body loads','Rotor thrust/torque only','3-axis force + moment + r×F about CG','Aircraft trim requirement','Full aircraft equilibrium'],
          ['Airframe aero','Cruise L/D proxy 8.5','Explicit wing + tail + fuselage/nacelle drag','Transition trim','Lift sharing and moments'],
          ['Mass source','Narrative/code could drift','Single configuration: 1950+550+500=3000 kg','M1 feedback','Reproducibility'],
        ])
    with open(os.path.join(OUT,'task6_3_updated_rotor_design.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['Parameter','M1 baseline','M2 value','Units / note']);w.writerows([
          ['Radius',ROTOR.R,ROTOR.R,'m'],['Root cut-out',ROTOR.r_root,ROTOR.r_root,'m'],['Blades',3,ROTOR.B,'-'],['Root chord',ROTOR.chord_root,ROTOR.chord_root,'m'],['Taper',ROTOR.taper_ratio,ROTOR.taper_ratio,'c_tip/c_root'],['Twist','0 to -12°','0 to -12°','deg'],['RPM','750 heli / 650 airplane','750 heli / 650 airplane; transition schedule user-controlled','RPM'],['Collective limits','2–26° hover; 5–35° cruise','same','deg'],['Cyclic limits','not used','±8°','deg'],['Stall criterion','±12° flag','±12° flag; 5% warning / 25% hard screen','deg/span'],['Tip-Mach limit','M1 screening','0.72','-']])
    with open(os.path.join(OUT,'task6_4_wing_empennage.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['Component','Area / span','Aspect ratio / model','Incidence','Location / control']);w.writerows([
          ['Main wing',f'{AIRCRAFT.wing_area_m2:.1f} m² / {AIRCRAFT.wing_span_m:.1f} m',f'AR={AIRCRAFT.wing_span_m**2/AIRCRAFT.wing_area_m2:.2f}; CLα={AIRCRAFT.wing_cl_alpha_per_rad:.2f}/rad; parabolic CD','2°','main wing / aileron'],
          ['Horizontal tail',f'{AIRCRAFT.htail_area_m2:.1f} m² / {AIRCRAFT.htail_span_m:.1f} m',f'CLα={AIRCRAFT.htail_cl_alpha_per_rad:.2f}/rad','-1°',f'aft / elevator, arm {AIRCRAFT.htail_arm_m:.1f} m'],
          ['Vertical tail',f'{AIRCRAFT.vtail_area_m2:.1f} m² / —','linear CYδr/Cnδr','0°',f'aft / rudder, arm {AIRCRAFT.vtail_arm_m:.1f} m'],
          ['Fuselage + nacelles',f'{AIRCRAFT.fuselage_flat_plate_m2+2*AIRCRAFT.nacelle_flat_plate_each_m2:.2f} m² eq. flat plate','parasite drag only','—','near CG / no wake model'],
        ])
    with open(os.path.join(OUT,'task6_5_mass_control_limits.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['Quantity','Value','Units','Note']);w.writerows([
          ['Gross mass',AIRCRAFT.gross_weight_kg,'kg','1950 empty + 550 payload + 500 fuel'],['Empty mass',AIRCRAFT.empty_weight_kg,'kg','design estimate'],['Payload',AIRCRAFT.max_payload_kg,'kg','maximum payload'],['Fuel',AIRCRAFT.fuel_capacity_kg,'kg','10% reserve'],['CG',f'({AIRCRAFT.cg_x_m:.2f},{AIRCRAFT.cg_y_m:.2f},{AIRCRAFT.cg_z_m:.2f})','m','body axes'],['Nacelle angle',f'{AIRCRAFT.nacelle_min_deg}–{AIRCRAFT.nacelle_max_deg}','deg','airplane→helicopter'],['Nacelle rate',AIRCRAFT.nacelle_rate_deg_s,'deg/s','example transition rate'],['Rotor RPM',f'{AIRCRAFT.rpm_min:.0f}–{AIRCRAFT.rpm_max:.0f}','RPM','hard operating range'],['Collective',f'{ROTOR.collective_hover_deg_min:.0f}–{ROTOR.collective_cruise_deg_max:.0f}','deg','bounded trim input'],['Cyclic',f'±{ROTOR.cyclic_deg_limit:.0f}','deg','both cyclic terms'],['Elevator',f'±{AIRCRAFT.elevator_deg_limit:.0f}','deg','bounded trim input'],['Rudder',f'±{AIRCRAFT.rudder_deg_limit:.0f}','deg','bounded trim input'],['Aileron',f'±{AIRCRAFT.aileron_deg_limit:.0f}','deg','bounded trim input']])

def exercise_model():
    model=TransitionAircraftModel(radial_stations=20,n_azimuth=16)
    alphas=np.radians(np.linspace(-10,14,121));cls=[];cds=[]
    for a in alphas:
        F,M,d=model.wing_forces(60,500,a);cls.append(d['CL']);cds.append(d['CD'])
    fig,ax=plt.subplots(figsize=(7,4.7));ax.plot(np.degrees(alphas),cls,label='Wing $C_L$');ax.plot(np.degrees(alphas),cds,label='Wing $C_D$');ax.axhline(AIRCRAFT.wing_cl_max,linestyle='--',label=f'Wing $C_{{L,max}}$={AIRCRAFT.wing_cl_max:.2f}');ax.set_xlabel('Aircraft angle of attack α [deg]');ax.set_ylabel('Aerodynamic coefficient [-]');ax.set_title('Task 6 / Sec. 5.4 — preliminary wing aerodynamic model | V=60 m/s, h=500 m, ISA');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(os.path.join(FIG,'task6_wing_model.png'),dpi=180);plt.close(fig)
    with open(os.path.join(OUT,'task6_airframe_assumptions.md'),'w',encoding='utf-8') as f:
        f.write('# Task 6 airframe-model assumptions\n\nWing, horizontal tail, vertical tail and equivalent parasite-drag elements see the prescribed aircraft airspeed and the current aircraft angle of attack. The model is steady and low-order. Rotor wake download/wing interference, fuselage sideforce, sideslip, dynamic stall and unsteady conversion aerodynamics are omitted and carried as Milestone-3 refinement items.\n')

def main():schematic();tables();exercise_model();print('Task 6 complete.')
if __name__=='__main__':main()
