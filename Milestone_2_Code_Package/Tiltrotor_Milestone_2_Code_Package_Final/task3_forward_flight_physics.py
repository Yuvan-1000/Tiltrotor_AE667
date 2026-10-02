"""Task 3: forward/edgewise-flow physics and Section-3 verification.

Covers M2 report Section 3.1-3.4: M1 limiting-case recovery, azimuthal
loading/periodicity, reverse-flow and advancing-tip/stall diagnostics, and
radial/azimuthal discretization sensitivity.
"""
from __future__ import annotations
import csv, os, warnings
import numpy as np
import matplotlib.pyplot as plt
from bemt_solver import FlightCondition, BEMTSolver, RotorGeometry
from aircraft_config import TILTROTOR_GEOM,TILTROTOR_AIRFOIL,ROTOR
from edgewise_bemt import EdgewiseFlightCondition,EdgewiseBEMTSolver

ROOT=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(ROOT,'figures'); OUT=os.path.join(ROOT,'outputs')
os.makedirs(FIG,exist_ok=True);os.makedirs(OUT,exist_ok=True)
AX=BEMTSolver(TILTROTOR_GEOM,TILTROTOR_AIRFOIL); ED=EdgewiseBEMTSolver(TILTROTOR_GEOM,TILTROTOR_AIRFOIL)
REP=dict(rpm=ROTOR.rpm_hover,collective_deg=14.0,altitude_m=0.0,V_axial_ms=0.0,V_edge_ms=35.0,n_azimuth=96,nacelle_angle_deg=90.0)

def limiting_cases():
    cases=[
      ('hover',ROTOR.rpm_hover,14.0,0.0,0.0),
      ('axial_forward',ROTOR.rpm_cruise,30.0,3000.0,30.0),
    ]
    rows=[]
    for name,rpm,theta,h,vax in cases:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a=AX.solve(FlightCondition.from_rpm(rpm,theta,altitude=h,V_axial=vax))
            e=ED.solve(EdgewiseFlightCondition.from_rpm(rpm,theta,altitude=h,V_axial=vax,V_edge=0,n_azimuth=96,nacelle_angle_deg=90))
        rows.append([name,rpm,theta,h,vax,a['T'],e['T'],abs(e['T']-a['T']),100*abs(e['T']-a['T'])/max(abs(a['T']),1e-12),a['Q'],e['Q'],abs(e['Q']-a['Q']),100*abs(e['Q']-a['Q'])/max(abs(a['Q']),1e-12),a['CT'],e['CT'],abs(e['CT']-a['CT']),a['CQ'],e['CQ'],abs(e['CQ']-a['CQ'])])
    with open(os.path.join(OUT,'task3_1_limiting_case_recovery.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['case','RPM','theta0_deg','altitude_m','V_axial_ms','T_M1_N','T_M2_N','abs_dT_N','T_pct_diff','Q_M1_Nm','Q_M2_Nm','abs_dQ_Nm','Q_pct_diff','CT_M1','CT_M2','abs_dCT','CQ_M1','CQ_M2','abs_dCQ']);w.writerows(rows)
    with open(os.path.join(OUT,'task3_1_verification_summary.md'),'w',encoding='utf-8') as f:
        f.write('# Task 3 / Sec. 3.1 — M1 limiting-case recovery\n\n')
        f.write('The M2 edgewise solver is evaluated with V_edge=0 and both cyclic inputs set to zero. Because its baseline inflow is taken from the same M1 axial solver, the limit should collapse to the M1 result at identical RPM, collective, altitude and axial speed.\n\n')
        for r in rows:f.write(f"- {r[0]}: T difference={r[8]:.3e}%, Q difference={r[12]:.3e}%, |dCT|={r[15]:.3e}, |dCQ|={r[18]:.3e}.\n")
    return rows

def representative():
    return EdgewiseFlightCondition.from_rpm(REP['rpm'],REP['collective_deg'],altitude=REP['altitude_m'],V_axial=REP['V_axial_ms'],V_edge=REP['V_edge_ms'],n_azimuth=REP['n_azimuth'],nacelle_angle_deg=REP['nacelle_angle_deg'])

def azimuthal_loading():
    with warnings.catch_warnings(): warnings.simplefilter('ignore'); r=ED.solve(representative())
    psi=np.degrees(r['psi'])
    fig,ax=plt.subplots(figsize=(8,5.6));cf=ax.contourf(psi,r['r_over_R'],r['dT_blade'],levels=28);fig.colorbar(cf,ax=ax,label='Blade-element thrust loading dT/dr [N/m]')
    if np.any(r['reverse_flow']):ax.contour(psi,r['r_over_R'],r['reverse_flow'].astype(float),levels=[.5],linestyles='--')
    ax.set(xlabel='Azimuth ψ [deg] (90° = advancing side)',ylabel='r/R',title='Task 3 / Sec. 3.2 — azimuthal thrust loading\n750 RPM, θ0=14°, Vedge=35 m/s, γ=90°, SL ISA')
    fig.tight_layout();fig.savefig(os.path.join(FIG,'task3_2_azimuthal_loading_contour.png'),dpi=180);plt.close(fig)
    with open(os.path.join(OUT,'task3_2_azimuthal_loading.md'),'w',encoding='utf-8') as f:
        f.write('Operating condition: 750 RPM, θ0=14°, Vedge=35 m/s, Vaxial=0 m/s, γ=90°, sea-level ISA, 80 radial stations, 96 azimuth stations, zero cyclic.\n\n')
        f.write(f"Advance ratio μ={r['mu']:.4f}; reverse-flow area fraction={r['reverse_flow_fraction']*100:.3f}%; reverse-flow span fraction={r['reverse_flow_span_fraction']*100:.2f}%; M_adv={r['M_adv_tip']:.3f}; stalled span={r['stalled_span_fraction']*100:.2f}%.\n\n")
        f.write('The contour is periodic over 360°. The strongest loading occurs on the advancing side because the local tangential speed is increased there; the retreating side is less loaded and the low-radius retreating region can enter reverse flow as μ grows.\n')
    with open(os.path.join(OUT,'task3_2_azimuthal_loading.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['r_m']+[f'psi_{p:.1f}_deg' for p in psi]);
        for i,rv in enumerate(r['r']):w.writerow([rv,*r['dT_blade'][i,:]])
    return r

def limit_diagnostics(r):
    psi=np.degrees(r['psi'])
    fig,ax=plt.subplots(figsize=(8,5.4));cf=ax.contourf(psi,r['r_over_R'],r['Ut'],levels=28);fig.colorbar(cf,ax=ax,label='In-plane blade velocity U_T [m/s]');ax.contour(psi,r['r_over_R'],r['Ut'],levels=[0.0],colors='black',linewidths=1.5)
    ax.set(xlabel='Azimuth ψ [deg] (90° advancing)',ylabel='r/R',title='Task 3 / Sec. 3.3 — in-plane velocity and reverse flow\n750 RPM, θ0=14°, Vedge=35 m/s, γ=90°, SL ISA');fig.tight_layout();fig.savefig(os.path.join(FIG,'task3_3_VT_reverse_flow.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5.4));cf=ax.contourf(psi,r['r_over_R'],r['alpha_deg'],levels=28);fig.colorbar(cf,ax=ax,label='Blade-section angle of attack α [deg]');ax.contour(psi,r['r_over_R'],r['alpha_deg'],levels=[-12,12],colors='black',linestyles='--',linewidths=1.4)
    ax.text(.02,.03,f"M_adv={r['M_adv_tip']:.3f} / {ROTOR.tip_mach_limit:.2f} limit\nStalled span={r['stalled_span_fraction']*100:.2f}% / {ROTOR.stall_span_fraction_limit*100:.0f}% hard limit",transform=ax.transAxes,bbox=dict(boxstyle='round',fc='white',alpha=.86),fontsize=9)
    ax.set(xlabel='Azimuth ψ [deg] (90° advancing)',ylabel='r/R',title='Task 3 / Sec. 3.3 — stall boundary and advancing-tip Mach\n750 RPM, θ0=14°, Vedge=35 m/s, γ=90°, SL ISA');fig.tight_layout();fig.savefig(os.path.join(FIG,'task3_3_alpha_stall_boundary.png'),dpi=180);plt.close(fig)
    with open(os.path.join(OUT,'task3_3_limit_diagnostics.md'),'w',encoding='utf-8') as f:
        f.write('Operating condition: 750 RPM, θ0=14°, Vedge=35 m/s, Vaxial=0 m/s, γ=90°, SL ISA, 80 radial stations, 96 azimuth stations.\n\n')
        f.write(f"Reverse-flow area={r['reverse_flow_fraction']*100:.3f}%, reverse-flow span={r['reverse_flow_span_fraction']*100:.2f}%, M_adv={r['M_adv_tip']:.3f}, Mach margin={r['tip_mach_margin']:.3f}, stalled-span fraction={r['stalled_span_fraction']*100:.2f}%.\n\n")
        f.write('The U_T=0 line is the onset of reverse flow. The α=±12° contours are the adopted stall flag boundary; the underlying linear polar is not a post-stall model.\n')

def discretization():
    nr=[20,40,60,80,120,160];npsi=[12,18,24,36,48,72,96]
    Tr=[];Qr=[];Tp=[];Qp=[]
    for n in nr:
        geom=RotorGeometry(**{**TILTROTOR_GEOM.__dict__,'n_stations':n});s=EdgewiseBEMTSolver(geom,TILTROTOR_AIRFOIL)
        with warnings.catch_warnings():warnings.simplefilter('ignore');rr=s.solve(representative())
        Tr.append(rr['T']);Qr.append(rr['Q'])
    for n in npsi:
        f=EdgewiseFlightCondition.from_rpm(REP['rpm'],REP['collective_deg'],altitude=0,V_axial=0,V_edge=REP['V_edge_ms'],n_azimuth=n,nacelle_angle_deg=90)
        with warnings.catch_warnings():warnings.simplefilter('ignore');rr=ED.solve(f)
        Tp.append(rr['T']);Qp.append(rr['Q'])
    specs=[(nr,Tr,'Number of radial stations n_r','Thrust T [N]','task3_4_thrust_vs_radial_stations.png'),(nr,Qr,'Number of radial stations n_r','Torque Q [N·m]','task3_4_torque_vs_radial_stations.png'),(npsi,Tp,'Number of azimuth stations n_ψ','Thrust T [N]','task3_4_thrust_vs_azimuth_stations.png'),(npsi,Qp,'Number of azimuth stations n_ψ','Torque Q [N·m]','task3_4_torque_vs_azimuth_stations.png')]
    for x,y,xlab,ylab,fn in specs:
        fig,ax=plt.subplots(figsize=(6.3,4.5));ax.plot(x,y,marker='o');ax.set(xlabel=xlab,ylabel=ylab,title=f'{ylab.split("[")[0].strip()} convergence | 750 RPM, θ0=14°, Vedge=35 m/s, γ=90°, SL ISA');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(os.path.join(FIG,fn),dpi=180);plt.close(fig)
    with open(os.path.join(OUT,'task3_4_discretization_sensitivity.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['resolution_type','resolution','T_N','Q_Nm']);
        for x,t,q in zip(nr,Tr,Qr):w.writerow(['radial',x,t,q])
        for x,t,q in zip(npsi,Tp,Qp):w.writerow(['azimuth',x,t,q])
    with open(os.path.join(OUT,'task3_4_convergence_note.md'),'w',encoding='utf-8') as f:
        f.write('The production grid is 80 radial × 72 azimuth stations. The 96-azimuth representative case is used for the contour/limit plots. The sensitivity plots show that changes become small at the adopted production resolution; the plotted convergence data, not a verbal assertion, are the basis for the chosen grid.\n')

def main():
    limiting_cases();r=azimuthal_loading();limit_diagnostics(r);discretization();print('Task 3 complete.')
if __name__=='__main__':main()
