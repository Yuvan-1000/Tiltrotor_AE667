"""Task 2: updated algorithms and logic-flow diagrams.

M2 report Section 2 asks for clear explanations of three tool chains:
2.1 edgewise rotor estimator, 2.2 trim solver, and 2.3 Mission Planner v2.
This script generates the three diagrams and a written step-by-step explanation.
"""
from __future__ import annotations
import os
import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(ROOT,'figures'); OUT=os.path.join(ROOT,'outputs')
os.makedirs(FIG,exist_ok=True);os.makedirs(OUT,exist_ok=True)

def flow(title,steps,filename,footnote):
    fig,ax=plt.subplots(figsize=(10,12));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,14)
    ax.set_title(title,fontsize=15,pad=14)
    y=13.2; gap=1.0 if len(steps)<=11 else 0.85
    for i,step in enumerate(steps,1):
        ax.text(5,y,f'{i}. {step}',ha='center',va='center',fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.45',fc='white',ec='black',lw=1.1),wrap=True)
        if i<len(steps):
            ax.annotate('',xy=(5,y-gap*.46),xytext=(5,y-gap*.22),arrowprops=dict(arrowstyle='->',lw=1.1))
        y-=gap
    ax.text(.45,.35,footnote,fontsize=8.5,ha='left',va='bottom',wrap=True)
    fig.tight_layout();fig.savefig(os.path.join(FIG,filename),dpi=180);plt.close(fig)

def main():
    flow('Task 2.1 — Edgewise-flight performance estimator',[
      'Read user inputs: rotor geometry, airfoil, RPM, θ0, θ1c, θ1s, altitude, ISA offset, body speed, nacelle angle, hub location',
      'Atmosphere → ρ, a, μ; resolve body-forward speed into V_edge and V_axial using nacelle angle γ',
      'Create radial r-grid and azimuth ψ-grid; evaluate chord c(r) and twist θ_tw(r)',
      'Call validated M1 axial BEMT at the same Ω, θ0, altitude and V_axial → baseline induced inflow λi0(r)',
      'Form total inflow ratio λG = λ̄i0 + V_axial/(ΩR); apply M2 Glauert/Drees correction to the induced part',
      'Blade kinematics at every (r,ψ): UT = Ωr + s V_edge sinψ and UP = V_axial + vi(r,ψ)',
      'Course pitch law: θ = θ0 + θ1c cosψ + θ1s sinψ; then φ = atan2(UP,UT), α = θ−φ',
      'Airfoil model → Cl, Cd and stall flag; compute sectional thrust, tangential force and torque',
      'Apply reverse-flow policy (UT<0 flagged/excluded), evaluate local/advancing-tip Mach and stalled-span fraction',
      'Azimuth-average and radial-integrate loads → T, Q, P and rotor-frame 3-axis force/moment',
      'Transform rotor loads to body axes; add hub placement moment rCG→hub × F; return diagnostics and convergence status',
    ],'task2_1_edgewise_flow_diagram.png','The first four stages reuse the M1 backend. The M2 extension is the (r,ψ) blade-kinematics/inflow/pitch calculation followed by sectional-load integration and body-axis transformation. This separation keeps the aerodynamic backend reusable by trim and mission analysis.')
    flow('Task 2.2 — Steady trim solver',[
      'Read a fixed flight state: nacelle angle γ, airspeed, altitude, gross mass, ISA offset, wind/climb rate and RPM policy',
      'Initialize bounded controls: pitch, collective, lateral cyclic θ1c, longitudinal cyclic θ1s, elevator, rudder, aileron',
      'Evaluate TransitionAircraftModel: two counter-rotating edgewise rotors + wing + tail + fuselage/nacelle drag + weight',
      'Build six-dimensional residual vector [FX, FY, FZ, MX, MY, MZ] and nondimensionalize with fixed force/moment scales',
      'Finite-difference the residual Jacobian, solve the bounded Gauss-Newton minimum-norm step, then apply backtracking',
      'Repeat until max scaled residual < tolerance, bounds stall progress, or the iteration limit is reached',
      'After numerical convergence, apply physical checks: rotor stall, reverse flow, advancing-tip Mach, wing stall, power and control saturation',
      'Store converged/rejected controls, all six loads, power, margins and reason for any failure for Task 7 and Task 8',
    ],'task2_2_trim_solver_flow_diagram.png','The solver is deterministic and bounded. Numerical trim means the six load residuals are small; physical feasibility is a separate test so a mathematically balanced state cannot be mistaken for an acceptable operating point.')
    flow('Task 2.3 — Mission Planner v2',[
      'Read a user-defined segment list: duration, airspeed schedule, altitude schedule, nacelle schedule, RPM schedule and controls/trim source',
      'At every Δt, evaluate atmosphere and wind convention; update current airspeed, ground speed, altitude and nacelle angle',
      'Get controls either from an explicit schedule or from a supplied trim-table lookup',
      'Build AircraftState and evaluate the same aircraft model used by steady trim',
      'Compute power required/available, fuel burn, gross-mass update and reserve-fuel status',
      'Check rotor stall, reverse flow, advancing-tip Mach, wing stall, power, collective/cyclic/surface limits and trim residual when enabled',
      'Write one history row with states, controls, loads, power, fuel and feasibility at each time step',
      'Stop at the first hard failure or advance to the next segment; report first violated constraint and final mass/fuel',
    ],'task2_3_mission_planner_flow_diagram.png','Mission Planner v2 owns time, mass and fuel; the aircraft/rotor model owns aerodynamics. That interface is the key to reproducing the same physics consistently in both steady trim and transition-mission tests.')
    text='# Task 2 algorithm explanation\n\n'
    text+='## 2.1 Edgewise estimator\n'
    text+='The estimator starts with user inputs and the ISA atmosphere, then resolves the aircraft forward speed into through-disk and in-plane components from the nacelle angle. The validated M1 axial BEMT provides a per-station induced-inflow baseline. The M2 handout relation is applied using lambda_G = lambda_bar_i + V_axial/(Omega R), where lambda_G is total inflow ratio and the corrected quantity is the induced component. Each (r,psi) element then uses azimuth-dependent tangential velocity, non-uniform induced velocity and the course cyclic-pitch law. Sectional lift/drag is resolved into thrust and tangential force, reverse-flow/stall/Mach checks are applied, and the loads are integrated over a revolution before being transformed to body axes.\n\n'
    text+='## 2.2 Trim solver\n'
    text+='The steady solver treats the flight condition as fixed and adjusts seven bounded pilot/attitude variables. Six residuals are evaluated directly from the aircraft model. A finite-difference Jacobian gives a Gauss-Newton step, with clipping and backtracking to keep the iteration inside the adopted bounds. Numerical success and physical feasibility are reported separately.\n\n'
    text+='## 2.3 Mission Planner v2\n'
    text+='The planner is a generic segment/time-step engine. All mission schedules come from JSON input rather than hard-coded mission logic. At each time step it calls the same aircraft model, burns fuel from shaft-power demand, reduces gross mass, and checks the adopted aerodynamic, power and control constraints. Both outbound and inbound demonstrations use the same aircraft configuration.\n'
    open(os.path.join(OUT,'task2_algorithm_explanations.md'),'w',encoding='utf-8').write(text)
    print('Task 2 complete.')
if __name__=='__main__':main()
