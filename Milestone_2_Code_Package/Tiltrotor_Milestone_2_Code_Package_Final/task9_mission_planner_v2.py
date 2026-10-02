"""Task 9: Mission Planner v2 transition tests.

Creates user-editable JSON inputs and runs both required conversions using the
same aircraft configuration:
  1) takeoff/initial climb -> hover-to-airplane conversion -> airplane hold
  2) airplane hold -> airplane-to-hover conversion -> hover descent

All required time histories are saved: altitude, airspeed, ground speed,
nacelle angle, rotor RPM, collective/cyclic controls, power required/available,
fuel and stall/tip-Mach margins.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib.pyplot as plt
from aircraft_config import AIRCRAFT,ROTOR
from mission_planner_v2 import MissionPlannerV2,save_history_csv

ROOT=os.path.dirname(os.path.abspath(__file__));OUT=os.path.join(ROOT,'outputs');FIG=os.path.join(ROOT,'figures');IN=os.path.join(ROOT,'inputs')
os.makedirs(OUT,exist_ok=True);os.makedirs(FIG,exist_ok=True);os.makedirs(IN,exist_ok=True)

NODES=[
    (90.0,8.0,750.0,0.0,24.0),
    (75.0,10.0,750.0,0.0,24.6),
    (60.0,15.0,750.0,8.0,24.0),
    (50.0,18.0,650.0,8.0,27.0),
    (45.0,20.0,650.0,7.0,28.0),
    (30.0,25.0,650.0,7.0,28.0),
    (20.0,28.0,650.0,7.0,28.0),
    (10.0,32.0,650.0,7.0,30.0),
    (0.0,35.0,650.0,7.0,28.0),
]

def segment_between(a,b,name):
    g0,V0,r0,p0,c0=a;g1,V1,r1,p1,c1=b
    return {'name':name,'duration_s':4.0,'airspeed_ms':[V0,V1],'altitude_m':[500.0,500.0],
            'nacelle_angle_deg':[g0,g1],'rpm':[r0,r1],'climb_rate_ms':0.0,'wind_ms':0.0,
            'control_source':'scheduled','enforce_trim':False,
            'controls':{'pitch_deg':[p0,p1],'collective_deg':[c0,c1],
                        'theta_1c_lateral_deg':[0,0],'theta_1s_longitudinal_deg':[0,0],
                        'elevator_deg':[0,0],'rudder_deg':[0,0],'aileron_deg':[0,0]}}

def build_missions():
    outbound_segments=[
      {'name':'takeoff_and_climb','duration_s':10.0,'airspeed_ms':[5.0,8.0],'altitude_m':[0.0,500.0],
       'nacelle_angle_deg':[90.0,90.0],'rpm':[750.0,750.0],'climb_rate_ms':2.0,'wind_ms':0.0,
       'control_source':'scheduled','enforce_trim':False,
       'controls':{'pitch_deg':[0,0],'collective_deg':[24,24],'theta_1c_lateral_deg':[0,0],'theta_1s_longitudinal_deg':[0,0],
                   'elevator_deg':[0,0],'rudder_deg':[0,0],'aileron_deg':[0,0]}}
    ]
    for i,(a,b) in enumerate(zip(NODES[:-1],NODES[1:]),1):outbound_segments.append(segment_between(a,b,f'hover_to_airplane_{i:02d}'))
    outbound_segments.append({'name':'airplane_mode_hold','duration_s':5.0,'airspeed_ms':35.0,'altitude_m':500.0,'nacelle_angle_deg':0.0,'rpm':650.0,
       'climb_rate_ms':0.0,'wind_ms':0.0,'control_source':'scheduled','enforce_trim':False,
       'controls':{'pitch_deg':7.0,'collective_deg':28.0,'theta_1c_lateral_deg':0.0,'theta_1s_longitudinal_deg':0.0,'elevator_deg':0.0,'rudder_deg':0.0,'aileron_deg':0.0}})
    common={'initial_fuel_kg':AIRCRAFT.fuel_capacity_kg,'gross_weight_kg':AIRCRAFT.gross_weight_kg,'wind_ms':0.0,'dt_s':1.0}
    outbound=dict(common,segments=outbound_segments)
    inbound_nodes=list(reversed(NODES))
    inbound_segments=[]
    inbound_segments.append({'name':'airplane_mode_hold','duration_s':5.0,'airspeed_ms':35.0,'altitude_m':500.0,'nacelle_angle_deg':0.0,'rpm':650.0,
       'climb_rate_ms':0.0,'wind_ms':0.0,'control_source':'scheduled','enforce_trim':False,
       'controls':{'pitch_deg':7.0,'collective_deg':28.0,'theta_1c_lateral_deg':0.0,'theta_1s_longitudinal_deg':0.0,'elevator_deg':0.0,'rudder_deg':0.0,'aileron_deg':0.0}})
    for i,(a,b) in enumerate(zip(inbound_nodes[:-1],inbound_nodes[1:]),1):inbound_segments.append(segment_between(a,b,f'airplane_to_hover_{i:02d}'))
    inbound_segments.append({'name':'hover_descent','duration_s':10.0,'airspeed_ms':[8.0,5.0],'altitude_m':[500.0,0.0],
       'nacelle_angle_deg':[90.0,90.0],'rpm':[750.0,750.0],'climb_rate_ms':-2.0,'wind_ms':0.0,
       'control_source':'scheduled','enforce_trim':False,
       'controls':{'pitch_deg':[0,0],'collective_deg':[24,24],'theta_1c_lateral_deg':[0,0],'theta_1s_longitudinal_deg':[0,0],
                   'elevator_deg':[0,0],'rudder_deg':[0,0],'aileron_deg':[0,0]}})
    inbound=dict(common,segments=inbound_segments)
    for name,m in [('outbound',outbound),('inbound',inbound)]:
        with open(os.path.join(IN,f'example_transition_{name}.json'),'w',encoding='utf-8') as f:json.dump(m,f,indent=2)
    return outbound,inbound

def plot_history(history,name):
    t=np.array([h['time_s'] for h in history],float)
    fig,axs=plt.subplots(5,1,figsize=(8.8,12),sharex=True)
    state_items=[
        ('altitude_m','Altitude [m]'),('airspeed_ms','Airspeed [m/s]'),
        ('ground_speed_ms','Ground speed [m/s]'),('nacelle_angle_deg','Nacelle angle [deg]'),
        ('rpm','Rotor RPM')]
    for ax,(key,label) in zip(axs,state_items):
        ax.plot(t,[h[key] for h in history],marker='o',ms=2)
        ax.set_ylabel(label);ax.grid(alpha=.25)
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Task 9 / Sec. 8.2 — {name} transition states | 3000 kg start, ISA; see mission input for schedules')
    fig.tight_layout();fig.savefig(os.path.join(FIG,f'task9_{name}_states.png'),dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.8,4.8))
    for key,label in [('pitch_deg','Pitch [deg]'),('collective_deg','Collective [deg]'),
                      ('theta_1s_longitudinal_deg','Longitudinal cyclic θ1s [deg]'),
                      ('theta_1c_lateral_deg','Lateral cyclic θ1c [deg]')]:
        ax.plot(t,[h[key] for h in history],marker='o',ms=2,label=label)
    ax.set_xlabel('Time [s]');ax.set_ylabel('Control / attitude angle [deg]')
    ax.set_title(f'Task 9 / Sec. 8.2 — {name} transition controls')
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(os.path.join(FIG,f'task9_{name}_controls.png'),dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.8,4.8))
    ax.plot(t,[h['P_required_kW'] for h in history],label='Power required [kW]')
    ax.plot(t,[h['P_available_kW'] for h in history],label='Power available [kW]')
    ax.set_xlabel('Time [s]');ax.set_ylabel('Power [kW]')
    ax.set_title(f'Task 9 / Sec. 8.2 — {name} transition power')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(os.path.join(FIG,f'task9_{name}_power.png'),dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.8,4.8))
    ax.plot(t,[h['fuel_remaining_kg'] for h in history],label='Fuel remaining [kg]')
    ax.set_xlabel('Time [s]');ax.set_ylabel('Fuel remaining [kg]')
    ax.set_title(f'Task 9 / Sec. 8.2 — {name} transition fuel')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(os.path.join(FIG,f'task9_{name}_fuel.png'),dpi=180);plt.close(fig)

    fig,axs=plt.subplots(3,1,figsize=(8.8,8.4),sharex=True)
    axs[0].plot(t,[h['stall_margin_pct'] for h in history],marker='o',ms=2);axs[0].set_ylabel('Stall margin [percentage points]')
    axs[1].plot(t,[h['tip_mach_margin'] for h in history],marker='o',ms=2);axs[1].set_ylabel('Mach margin [-]')
    axs[2].plot(t,[h['reverse_flow_pct'] for h in history],marker='o',ms=2);axs[2].set_ylabel('Reverse-flow area [%]');axs[2].set_xlabel('Time [s]')
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle(f'Task 9 / Sec. 8.2 — {name} transition aerodynamic margins')
    fig.tight_layout();fig.savefig(os.path.join(FIG,f'task9_{name}_margins.png'),dpi=180);plt.close(fig)

def run(name,mission):
    planner=MissionPlannerV2(radial_stations=12,n_azimuth=12)
    result=planner.run(mission,dt=mission.get('dt_s',1.0));save_history_csv(result['history'],os.path.join(OUT,f'task9_{name}_history.csv'));plot_history(result['history'],name)
    with open(os.path.join(OUT,f'task9_{name}_summary.md'),'w',encoding='utf-8') as f:
        f.write(f'# Task 9 / Sec. 8 — {name} conversion\n\nSuccess: **{result["success"]}**.\n\n')
        f.write(f'Initial fuel: {AIRCRAFT.fuel_capacity_kg:.1f} kg; final fuel: {result["final_fuel_kg"]:.2f} kg; final gross mass: {result["final_gross_weight_kg"]:.2f} kg.\n')
        if result['failure']:f.write(f'First failure: {result["failure"]}\n')
        else:
            f.write(f"Minimum stall margin: {min(h['stall_margin_pct'] for h in result['history']):.2f} percentage points; minimum Mach margin: {min(h['tip_mach_margin'] for h in result['history']):.4f}; maximum power required: {max(h['P_required_kW'] for h in result['history']):.1f} kW.\n")
    return result

def main():
    a,b=build_missions();ra=run('outbound',a);rb=run('inbound',b)
    assert ra['success'] and rb['success'],(ra['failure'],rb['failure'])
    print('Task 9 complete: outbound and inbound missions passed.')
if __name__=='__main__':main()
