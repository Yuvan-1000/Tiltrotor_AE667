"""
task10_feasibility_demo.py
================================================================================
Milestone-1 Task 10 / Demonstration Cases: "Mission test: complete one
feasible payload mission and one deliberately infeasible mission, explaining
the first violated constraint."

Uses mission_planner.py (Task 9) unchanged -- Task 10's feasibility checks
(fuel, power margin, stall, tip Mach, collective/RPM bounds) are already
built into MissionPlanner._step(); this script just demonstrates them on two
concrete missions and prints/saves a clear report of what happened, when,
and why.
================================================================================
"""

import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mission_planner import MissionPlanner
from task5_tiltrotor_design import AIRCRAFT

warnings.filterwarnings("ignore")

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ================================================================================
# Mission A: FEASIBLE round-trip payload mission
#   sized using the validated numbers from Tasks 6/7/9:
#     - takeoff weight 2700 kg (below the 3205 kg max-hover-GW-at-SL found
#       in Task 6, leaving climb/cruise power margin)
#     - climb/descent at 1.5 m/s (found sustainable in Task 9 dev testing)
#     - cruise at 22 m/s (inside the ~27.5 m/s trim-feasible envelope found
#       in Task 9's cruise-range study)
# ================================================================================
def feasible_mission():
    mp = MissionPlanner(gross_weight_kg=2700.0, fuel_kg=AIRCRAFT["fuel_capacity_kg"], dt_s=30.0)
    segments = [
        {"type": "hover", "duration_s": 60, "altitude_m": 0.0},
        {"type": "climb", "target_altitude_m": 300.0, "climb_rate_ms": 1.5},
        {"type": "cruise", "distance_km": 30.0, "altitude_m": 300.0, "airspeed_ms": 22.0},
        {"type": "payload", "delta_kg": -150.0},     # drop 150 kg payload
        {"type": "loiter", "duration_s": 120, "altitude_m": 300.0},
        {"type": "cruise", "distance_km": 30.0, "altitude_m": 300.0, "airspeed_ms": 22.0},
        {"type": "descent", "target_altitude_m": 0.0, "descent_rate_ms": 1.5},
        {"type": "hover", "duration_s": 30, "altitude_m": 0.0},
    ]
    summary = mp.run(segments)

    print("=== Task 10, Mission A (FEASIBLE) -- design test case ===")
    print(f"Segments: {[s['type'] for s in segments]}")
    print(f"Completed: {summary['completed']}")
    print(f"Total time: {summary['final_time_s']/60:.1f} min, "
          f"total distance: {summary['final_distance_km']:.1f} km")
    print(f"Fuel remaining: {summary['final_fuel_kg']:.1f} kg "
          f"(reserve = {mp.reserve_fuel_kg:.1f} kg, margin = "
          f"{summary['final_fuel_kg']-mp.reserve_fuel_kg:.1f} kg)")
    print(f"Final gross weight: {summary['final_gross_weight_kg']:.1f} kg")

    _plot_mission(mp, "Mission A (feasible)", "task10_missionA_feasible")
    _write_log_csv(mp, "task10_missionA_feasible_log.csv")
    return mp, summary


# ================================================================================
# Mission B: DELIBERATELY INFEASIBLE mission
#   Same starting point as Mission A (2700 kg, a feasible takeoff weight),
#   but this time the aircraft picks up a payload mid-mission that is far
#   too heavy (+900 kg -- pushing gross weight to 3520 kg, above both the
#   3000 kg design MTOW and the 3205 kg max hover weight found in Task 6).
#   This is a more illustrative failure than an overweight takeoff: the
#   early hover/climb/cruise segments succeed normally, and the mission
#   only becomes infeasible once the oversized pickup happens -- exactly
#   the kind of segment/time/reason identification Task 10 asks for.
# ================================================================================
def infeasible_mission():
    mp = MissionPlanner(gross_weight_kg=2700.0, fuel_kg=AIRCRAFT["fuel_capacity_kg"], dt_s=30.0)
    segments = [
        {"type": "hover", "duration_s": 60, "altitude_m": 0.0},
        {"type": "climb", "target_altitude_m": 300.0, "climb_rate_ms": 1.5},
        {"type": "cruise", "distance_km": 30.0, "altitude_m": 300.0, "airspeed_ms": 22.0},
        {"type": "descent", "target_altitude_m": 0.0, "descent_rate_ms": 1.5},
        {"type": "hover", "duration_s": 30, "altitude_m": 0.0},
        {"type": "payload", "delta_kg": +900.0},     # oversized pickup -> infeasible from here on
        {"type": "climb", "target_altitude_m": 300.0, "climb_rate_ms": 1.5},
        {"type": "cruise", "distance_km": 30.0, "altitude_m": 300.0, "airspeed_ms": 22.0},
        {"type": "descent", "target_altitude_m": 0.0, "descent_rate_ms": 1.5},
        {"type": "hover", "duration_s": 30, "altitude_m": 0.0},
    ]
    summary = mp.run(segments)

    print("\n=== Task 10, Mission B (DELIBERATELY INFEASIBLE) -- design test case ===")
    print(f"Takeoff weight: 2700 kg (feasible, same as Mission A) -- but this mission")
    print(f"picks up an oversized +900 kg payload mid-route (-> 3520 kg gross, above the")
    print(f"3000 kg design MTOW and the 3205 kg Task-6 max-hover-GW-at-SL)")
    print(f"Completed: {summary['completed']}")
    if summary["failure"]:
        f = summary["failure"]
        print(f"FIRST VIOLATED CONSTRAINT:")
        print(f"  Segment : {f['segment']}")
        print(f"  Time    : {f['time_s']:.1f} s ({f['time_s']/60:.1f} min)")
        print(f"  Reason  : {f['reason']}")
    else:
        print("(Unexpectedly completed -- adjust the overweight test case.)")

    _plot_mission(mp, "Mission B (deliberately infeasible)", "task10_missionB_infeasible")
    _write_log_csv(mp, "task10_missionB_infeasible_log.csv")
    return mp, summary


# ================================================================================
# Plot & log helpers
# ================================================================================
def _plot_mission(mp, title, fname_prefix):
    t_min = [row["t_s"] / 60.0 for row in mp.log if "gross_weight_kg" in row]
    gw = [row["gross_weight_kg"] for row in mp.log if "gross_weight_kg" in row]
    alt = [row.get("altitude_m", None) for row in mp.log if "gross_weight_kg" in row]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(t_min, gw, "-o", ms=3, color="tab:blue")
    axes[0].set_xlabel("Time [min]"); axes[0].set_ylabel("Gross weight [kg]")
    axes[0].set_title("Gross weight vs time"); axes[0].grid(alpha=0.3)

    axes[1].plot(t_min, alt, "-o", ms=3, color="tab:orange")
    axes[1].set_xlabel("Time [min]"); axes[1].set_ylabel("Altitude [m]")
    axes[1].set_title("Altitude profile"); axes[1].grid(alpha=0.3)

    if mp.failed:
        axes[0].axvline(mp.failure.time_s / 60.0, color="red", ls="--", lw=1.5, label="mission failure")
        axes[1].axvline(mp.failure.time_s / 60.0, color="red", ls="--", lw=1.5, label="mission failure")
        axes[0].legend(); axes[1].legend()

    fig.suptitle(f"Task 10 -- {title}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{fname_prefix}.png"), dpi=160)
    plt.close(fig)


def _write_log_csv(mp, fname):
    keys = set()
    for row in mp.log:
        keys.update(row.keys())
    keys = sorted(keys)
    with open(os.path.join(OUT_DIR, fname), "w") as f:
        f.write(",".join(keys) + "\n")
        for row in mp.log:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    mpA, summaryA = feasible_mission()
    mpB, summaryB = infeasible_mission()

    with open(os.path.join(OUT_DIR, "task10_summary.md"), "w") as f:
        f.write("# Task 10 -- Mission feasibility demonstration summary\n\n")
        f.write("## Mission A (feasible)\n")
        f.write(f"- Completed: {summaryA['completed']}\n")
        f.write(f"- Total time: {summaryA['final_time_s']/60:.1f} min\n")
        f.write(f"- Total distance: {summaryA['final_distance_km']:.1f} km\n")
        f.write(f"- Fuel remaining: {summaryA['final_fuel_kg']:.1f} kg "
                f"(reserve={mpA.reserve_fuel_kg:.1f} kg)\n\n")
        f.write("## Mission B (deliberately infeasible)\n")
        f.write(f"- Completed: {summaryB['completed']}\n")
        if summaryB["failure"]:
            fb = summaryB["failure"]
            f.write(f"- First violated constraint: segment='{fb['segment']}', "
                    f"time={fb['time_s']:.1f} s, reason: {fb['reason']}\n")

    print(f"\nPlots written to {FIG_DIR}/task10_mission[AB]_*.png")
    print(f"Logs written to {OUT_DIR}/task10_mission[AB]_*_log.csv")
    print(f"Summary written to {OUT_DIR}/task10_summary.md")
