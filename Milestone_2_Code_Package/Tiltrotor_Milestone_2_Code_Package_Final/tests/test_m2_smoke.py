"""Fast smoke/regression checks for the Milestone-2 extension."""
from __future__ import annotations
import csv, json, os, sys, warnings
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
from aircraft_config import AIRCRAFT, ROTOR, TILTROTOR_GEOM, TILTROTOR_AIRFOIL
from bemt_solver import BEMTSolver, FlightCondition
from edgewise_bemt import EdgewiseBEMTSolver, EdgewiseFlightCondition, rotor_to_body_matrix
from aircraft_model import TransitionAircraftModel, AircraftState
from mission_planner_v2 import MissionPlannerV2
from trim_solver import SteadyTrimSolver, TrimCase


def test_frames():
    C=rotor_to_body_matrix(37.0)
    assert np.allclose(C.T@C,np.eye(3),atol=1e-12)
    assert np.isclose(np.linalg.det(C),1.0,atol=1e-12)


def test_mu_zero():
    ax=BEMTSolver(TILTROTOR_GEOM,TILTROTOR_AIRFOIL)
    ed=EdgewiseBEMTSolver(TILTROTOR_GEOM,TILTROTOR_AIRFOIL)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for rpm,th,h,vax in [(750,14,0,0),(650,30,3000,30)]:
            a=ax.solve(FlightCondition.from_rpm(rpm,th,altitude=h,V_axial=vax))
            e=ed.solve(EdgewiseFlightCondition.from_rpm(rpm,th,altitude=h,V_axial=vax,V_edge=0,n_azimuth=24))
            assert abs(e['CT']-a['CT'])<1e-8
            assert abs(e['CQ']-a['CQ'])<1e-8


def test_twin_torque_cancellation():
    model=TransitionAircraftModel(radial_stations=12,n_azimuth=16)
    s=AircraftState(20,500,0,90,750,14,0,0,0,0,0,0,0,AIRCRAFT.gross_weight_kg,0)
    o=model.loads(s)
    assert abs(o['MZ'])<1e-6


def test_trim_outputs():
    solver=SteadyTrimSolver(radial_stations=10,n_azimuth=8,max_iter=6,tol=5e-3)
    r=solver.solve(TrimCase(45,30,500,AIRCRAFT.gross_weight_kg))
    assert len(r['loads'])==6
    assert len(r['x'])==7


def test_mission_examples():
    for name in ('outbound','inbound'):
        with open(os.path.join(ROOT,'inputs',f'example_transition_{name}.json'),encoding='utf-8') as f:m=json.load(f)
        planner=MissionPlannerV2(radial_stations=10,n_azimuth=10)
        r=planner.run(m,dt=1.0)
        assert r['success'],(name,r['failure'])
        assert r['final_fuel_kg']>=AIRCRAFT.fuel_capacity_kg*AIRCRAFT.reserve_fuel_fraction-1e-6
        assert r['final_gross_weight_kg']<=AIRCRAFT.gross_weight_kg


def test_range_shape():
    p=os.path.join(ROOT,'outputs','task10_m1_range_feedback_sanity.md')
    assert os.path.isfile(p)
    txt=open(p,encoding='utf-8').read()
    assert 'curve decreases at the high-speed end' in txt


def main():
    test_frames();test_mu_zero();test_twin_torque_cancellation();test_trim_outputs();test_mission_examples();test_range_shape()
    print('M2 smoke/regression tests passed.')

if __name__=='__main__':main()
