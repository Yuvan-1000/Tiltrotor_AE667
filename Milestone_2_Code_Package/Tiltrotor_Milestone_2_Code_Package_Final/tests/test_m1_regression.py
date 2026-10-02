"""Regression tests for the unchanged Milestone-1 axial BEMT backend."""
from __future__ import annotations
import csv, os, sys, warnings
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bemt_solver import BEMTSolver, FlightCondition
from aircraft_config import TILTROTOR_GEOM, TILTROTOR_AIRFOIL


def main() -> None:
    expected_path=os.path.join(ROOT,"data","m1_regression_expected.csv")
    with open(expected_path,encoding="utf-8") as f: rows=list(csv.DictReader(f))
    solver=BEMTSolver(TILTROTOR_GEOM,TILTROTOR_AIRFOIL)
    for row in rows:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r=solver.solve(FlightCondition.from_rpm(float(row["rpm"]),float(row["collective_deg"]),
                                                     altitude=float(row["altitude_m"]),
                                                     V_axial=float(row["V_axial_ms"])))
        for key, actual in [("T_N",r["T"]),("Q_Nm",r["Q"]),("P_W",r["P"]),
                            ("CT",r["CT"]),("CQ",r["CQ"]),("CP",r["CP"])]:
            target=float(row[key]); tol=2e-10*max(1.0,abs(target))
            assert abs(actual-target)<=tol,(row["case"],key,actual,target)
    print("M1 regression tests passed.")

if __name__=="__main__":main()
