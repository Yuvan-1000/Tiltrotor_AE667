"""Audit the Milestone-2 code package against the supplied submission requirements."""
from __future__ import annotations
import csv
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIRED_FILES = [
    "README.md", "requirements.txt", "environment.yml", "REFERENCES.md",
    "ACKNOWLEDGEMENT.md", "M2_TASK_MAP.md", "M2_REQUIREMENTS_CHECKLIST.md",
    "M2_MODEL_LIMITATIONS.md", "M1_BASELINE_REFERENCE.md",
    "aircraft_config.py", "aircraft_model.py", "bemt_solver.py", "edgewise_bemt.py",
    "trim_solver.py", "mission_planner_v2.py", "run_m2_all.py",
    *[f"task{i:1d}_" for i in []],
]
TASK_SCRIPTS = [
    "task1_coordinate_systems.py", "task2_updated_algorithms.py",
    "task3_forward_flight_physics.py", "task4_rotor_forces_moments.py",
    "task5_control_response.py", "task6_aircraft_aerodynamic_model.py",
    "task7_trimmed_flight.py", "task8_conversion_corridor.py",
    "task9_mission_planner_v2.py", "task10_design_refinement.py",
]
EXPECTED_OUTPUTS = [
    "task1_assumptions.csv", "task2_algorithm_explanations.md",
    "task3_1_limiting_case_recovery.csv", "task3_2_azimuthal_loading.csv",
    "task3_4_discretization_sensitivity.csv", "task4_rotor_loads.csv",
    "task4_twin_rotor_loads.csv", "task5_helicopter_like_collective.csv",
    "task5_intermediate_conversion_collective.csv", "task6_2_changes_from_m1.csv",
    "task6_3_updated_rotor_design.csv", "task6_4_wing_empennage.csv",
    "task6_5_mass_control_limits.csv", "task7_required_trim_matrix.csv",
    "task7_trim_results.csv", "task7_trim_status_matrix.csv",
    "task8_speed_nacelle_corridor.csv", "task9_outbound_history.csv",
    "task9_inbound_history.csv", "task10_design_refinement.csv",
    "task10_milestone3_plan.csv",
]
EXPECTED_FIGURES = [
    "task1_coordinate_frames.png", "task2_1_edgewise_flow_diagram.png",
    "task2_2_trim_solver_flow_diagram.png", "task2_3_mission_planner_flow_diagram.png",
    "task3_2_azimuthal_loading_contour.png", "task3_3_VT_reverse_flow.png",
    "task3_3_alpha_stall_boundary.png", "task3_4_thrust_vs_radial_stations.png",
    "task3_4_torque_vs_radial_stations.png", "task3_4_thrust_vs_azimuth_stations.png",
    "task3_4_torque_vs_azimuth_stations.png", "task4_counter_rotation_torque.png",
    "task6_updated_aircraft_schematic.png", "task7_trim_matrix_status.png",
    "task8_speed_nacelle_feasibility_map.png", "task8_active_constraint_boundaries.png",
    "task9_outbound_states.png", "task9_inbound_states.png",
    "task10_cruise_range_sanity.png",
]

def check() -> int:
    missing = []
    for f in REQUIRED_FILES + TASK_SCRIPTS:
        if not os.path.isfile(os.path.join(ROOT, f)):
            missing.append(f)
    for d in ("inputs", "data", "figures", "outputs", "tests"):
        if not os.path.isdir(os.path.join(ROOT, d)):
            missing.append(d + "/")
    for f in EXPECTED_OUTPUTS:
        if not os.path.isfile(os.path.join(ROOT, "outputs", f)):
            missing.append("outputs/" + f)
    for f in EXPECTED_FIGURES:
        if not os.path.isfile(os.path.join(ROOT, "figures", f)):
            missing.append("figures/" + f)
    for f in ["inputs/aircraft_config.json", "inputs/control_sweep_conditions.json",
              "inputs/trim_cases.csv", "inputs/example_transition_outbound.json",
              "inputs/example_transition_inbound.json", "data/m1_regression_expected.csv",
              "data/knight_hefner_table1.csv"]:
        if not os.path.isfile(os.path.join(ROOT, f)):
            missing.append(f)
    if missing:
        print("AUDIT FAILED")
        for f in missing: print(" -", f)
        return 1
    print("AUDIT PASSED: M2 package structure, required examples/data, figures and outputs are present.")
    return 0

if __name__ == "__main__":
    raise SystemExit(check())
