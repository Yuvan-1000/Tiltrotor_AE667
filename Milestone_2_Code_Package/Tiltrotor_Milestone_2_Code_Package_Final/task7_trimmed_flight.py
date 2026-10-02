"""Task 7: steady trimmed-flight analysis (M2 Sections 2.2 and 6.1-6.4).

Runs the required 3x3 matrix: three nacelle angles spanning helicopter,
conversion and airplane mode, with three airspeeds at each angle. Records all
six aircraft loads, controls, power and feasibility status, and documents a
failed/rejected trim case.
"""
from __future__ import annotations
import csv, os, warnings
import numpy as np
import matplotlib.pyplot as plt
from trim_solver import SteadyTrimSolver,TrimCase,save_trim_result_csv
from aircraft_config import AIRCRAFT,ROTOR

ROOT=os.path.dirname(os.path.abspath(__file__));OUT=os.path.join(ROOT,'outputs');FIG=os.path.join(ROOT,'figures');IN=os.path.join(ROOT,'inputs')
os.makedirs(OUT,exist_ok=True);os.makedirs(FIG,exist_ok=True);os.makedirs(IN,exist_ok=True)
ANGLES=[90.0,45.0,0.0];SPEEDS=[20.0,30.0,40.0];ALT=500.0

def classify(r):
    o=r['loads_dict'];rot=o['rotor'];wing=o['wing']
    if not r['success']:
        if rot['stall_span']>ROTOR.stall_span_fraction_limit:return 'no trim before rotor-stall constraint'
        if r['control_saturated']:return 'no trim / control saturation'
        return 'numerical or insufficient trim authority'
    if r['control_saturated']:return 'control saturation'
    if rot['reverse_flow_fraction']>1e-12:return 'reverse-flow'
    if rot['stall_span']>ROTOR.stall_span_fraction_limit:return 'rotor stall'
    if rot['M_adv']>ROTOR.tip_mach_limit:return 'advancing-tip Mach'
    if wing['wing_stall']:return 'wing stall'
    if o['P_required_W']>o['P_available_W']:return 'power limitation'
    return 'feasible'

def main():
    solver=SteadyTrimSolver(radial_stations=10,n_azimuth=8,max_iter=10,tol=3e-3)
    results=[]; status_rows=[]
    for gamma in ANGLES:
        previous=None
        for V in SPEEDS:
            case=TrimCase(gamma,V,altitude_m=ALT,gross_weight_kg=AIRCRAFT.gross_weight_kg,dT_isa=0.0,wind_ms=0.0,climb_rate_ms=0.0)
            with warnings.catch_warnings():warnings.simplefilter('ignore');r=solver.solve(case,x0=previous)
            status=classify(r);r['status']=status;results.append(r)
            if r['success']:previous=r['x']
            status_rows.append([gamma,V,status,r['success'],r['physical_feasible'],r['max_scaled_residual'],*r['x'],*r['loads'],r['loads_dict']['P_required_W']/1000,r['loads_dict']['P_available_W']/1000,r['loads_dict']['rotor']['stall_span']*100,r['loads_dict']['rotor']['M_adv']])
    save_trim_result_csv(results,os.path.join(OUT,'task7_trim_results.csv'))
    with open(os.path.join(OUT,'task7_trim_status_matrix.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['nacelle_deg','V_ms','status','numerical_trim','physical_feasible','max_scaled_residual',*solver.names,*solver.residual_names,'P_required_kW','P_available_kW','stall_span_pct','M_adv']);w.writerows(status_rows)
    with open(os.path.join(OUT,'task7_required_trim_matrix.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['Nacelle angle [deg]','Airspeed 1 [m/s]','Airspeed 2 [m/s]','Airspeed 3 [m/s]']);[w.writerow([g,*SPEEDS]) for g in ANGLES]
    with open(os.path.join(IN,'trim_cases.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['nacelle_deg','airspeed_ms','altitude_m','gross_weight_kg','dT_isa_K','wind_ms','climb_rate_ms']);[w.writerow([g,V,ALT,AIRCRAFT.gross_weight_kg,0.0,0.0,0.0]) for g in ANGLES for V in SPEEDS]
    failed=next((r for r in results if not r['physical_feasible']),results[0])
    with open(os.path.join(OUT,'task7_failed_trim_case.md'),'w',encoding='utf-8') as f:
        f.write('# Task 7 / Sec. 6.3 failed/rejected trim case\n\n')
        f.write(f"Case: γ={failed['case']['nacelle_angle_deg']:.0f}°, V={failed['case']['airspeed_ms']:.0f} m/s, h={failed['case']['altitude_m']:.0f} m, gross mass={failed['case']['gross_weight_kg']:.0f} kg, ISA.\n\n")
        f.write(f"Numerical trim: {failed['success']}; physical feasibility: {failed['physical_feasible']}; classification: **{failed['status']}**.\n\n")
        o=failed['loads_dict'];f.write(f"Maximum scaled residual = {failed['max_scaled_residual']:.4g}; rotor stalled span = {o['rotor']['stall_span']*100:.2f}%; M_adv={o['rotor']['M_adv']:.3f}; power={o['P_required_W']/1000:.1f}/{o['P_available_W']/1000:.1f} kW; control saturated={failed['control_saturated']}.\n")
    # 3x3 grid of success/feasibility and six-load residual magnitudes.
    Z=np.full((3,3),np.nan);R=np.zeros((3,3))
    for i,g in enumerate(ANGLES):
        for j,V in enumerate(SPEEDS):
            r=next(x for x in results if x['case']['nacelle_angle_deg']==g and x['case']['airspeed_ms']==V);Z[i,j]=1 if r['physical_feasible'] else (0.5 if r['success'] else 0);R[i,j]=r['max_scaled_residual']
    fig,ax=plt.subplots(figsize=(7.2,5.2));im=ax.imshow(Z,origin='lower',extent=[0.5,3.5,-.5,2.5],vmin=0,vmax=1,aspect='auto');ax.set_xticks([1,2,3],SPEEDS);ax.set_yticks(range(3),ANGLES);ax.set_xlabel('Airspeed [m/s]');ax.set_ylabel('Nacelle angle γ [deg]');ax.set_title('Task 7 — 3×3 trim matrix status | h=500 m, 3000 kg, ISA');
    for i,g in enumerate(ANGLES):
        for j,V in enumerate(SPEEDS):
            r=next(x for x in results if x['case']['nacelle_angle_deg']==g and x['case']['airspeed_ms']==V);label='F' if r['physical_feasible'] else ('N' if r['success'] else 'X');ax.text(j+1,i,label,ha='center',va='center',fontsize=12,fontweight='bold')
    ax.set_xlim(.5,3.5);ax.set_ylim(-.5,2.5);fig.colorbar(im,ax=ax,ticks=[0,.5,1],label='0 failed, 0.5 mathematical only, 1 physical');fig.tight_layout();fig.savefig(os.path.join(FIG,'task7_trim_matrix_status.png'),dpi=180);plt.close(fig)
    # Trend plot: separate axes keep units explicit and avoid mixing degrees and kW.
    fig,axs=plt.subplots(3,1,figsize=(8.5,10),sharex=True)
    markers=['o','s','^']; styles=['-','--','-.']
    for g in ANGLES:
        rr=[r for r in results if r['case']['nacelle_angle_deg']==g and r['success']]
        if not rr: continue
        v=np.array([r['case']['airspeed_ms'] for r in rr],float)
        pitch=np.array([r['x'][0] for r in rr],float)
        col=np.array([r['x'][1] for r in rr],float)
        power=np.array([r['loads_dict']['P_required_W']/1000 for r in rr],float)
        axs[0].plot(v,pitch,marker=markers[0],linestyle=styles[0],label=f'γ={g:.0f}°')
        axs[1].plot(v,col,marker=markers[1],linestyle=styles[1],label=f'γ={g:.0f}°')
        axs[2].plot(v,power,marker=markers[2],linestyle=styles[2],label=f'γ={g:.0f}°')
    axs[0].set_ylabel('Pitch attitude [deg]'); axs[1].set_ylabel('Collective [deg]'); axs[2].set_ylabel('Power required [kW]')
    axs[2].set_xlabel('Airspeed [m/s]')
    for ax in axs: ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Task 7 / Sec. 6.4 — trim-control and power trends | h=500 m, 3000 kg, ISA')
    fig.tight_layout();fig.savefig(os.path.join(FIG,'task7_trim_trends.png'),dpi=180);plt.close(fig)
    with open(os.path.join(OUT,'task7_physical_interpretation.md'),'w',encoding='utf-8') as f:
        f.write('# Task 7 / Sec. 6.4 — physical interpretation of trim trends\n\n')
        f.write('As γ decreases from helicopter toward airplane mode, the rotor thrust direction rotates progressively into the body-forward direction. The wing therefore carries a larger fraction of the required vertical force at higher airspeed, while rotor collective and pitch attitude shift to satisfy the longitudinal and vertical force balance. Longitudinal cyclic is available in the model to create a first-harmonic hub pitching response; tail elevator supplies the remaining aircraft pitching-moment trim. The exact values depend on the low-order wing/tail model and the rigid-disk rotor approximation.\n\n')
        f.write('A numerically converged state is not automatically a feasible state: rotor stall, reverse flow, Mach, wing-stall, power and control bounds are checked after convergence. This distinction is used in the conversion map and mission planner.\n')
    print('Task 7 complete.')
if __name__=='__main__':main()
