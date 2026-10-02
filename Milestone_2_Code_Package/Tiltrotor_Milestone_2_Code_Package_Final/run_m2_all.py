"""Run every Milestone-2 task driver in report order.

The driver intentionally follows the M2 PDF order, not any project-track
assignment split. Task 7 produces the trim matrix consumed by Task 8, and
Task 9 creates the editable transition mission inputs used by its demonstrations.
"""
from __future__ import annotations
import importlib
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
TASKS = [
    "task1_coordinate_systems",
    "task2_updated_algorithms",
    "task3_forward_flight_physics",
    "task4_rotor_forces_moments",
    "task5_control_response",
    "task6_aircraft_aerodynamic_model",
    "task7_trimmed_flight",
    "task8_conversion_corridor",
    "task9_mission_planner_v2",
    "task10_design_refinement",
]

def main() -> None:
    print("=== AE667 Milestone 2: run Tasks 1–10 in PDF order ===")
    for name in TASKS:
        path = os.path.join(ROOT, name + ".py")
        print(f"\n--- {name} ---")
        subprocess.run([sys.executable, path], cwd=ROOT, check=True)
    print("\n=== All Milestone-2 task drivers completed ===")

if __name__ == "__main__":
    main()
