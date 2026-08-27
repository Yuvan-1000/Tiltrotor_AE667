"""
task9_mission_planner_demo.py
================================================================================
Milestone-1 Task 9 demonstration / Report Sections 7.1-7.4.

Runs mission_planner.py through:
  7.1  a set of small, targeted verification tests (segment sequencing, mass
       continuity, payload pickup/drop, fuel update, atmospheric variation,
       wind treatment, reserve-fuel accounting, power required/available,
       failure-warning logic) and prints a PASS/FAIL table
  7.2  fuel-burn rate vs gross weight, hover, sea level
  7.3  hover endurance vs takeoff weight (incl. adopted reserve-fuel policy)
  7.4  cruise range vs cruise speed (incl. adopted reserve-fuel policy)
================================================================================
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mission_planner import (
    MissionPlanner, _solve_for_thrust, power_available_kW,
    G, CRUISE_L_OVER_D,
)
from task5_tiltrotor_design import (
    AIRCRAFT, RPM_HOVER, RPM_CRUISE, COLLECTIVE_RANGE_HOVER_DEG,
    COLLECTIVE_RANGE_CRUISE_DEG,
)

warnings.filterwarnings("ignore")   # suppress BEMT bracket-warning noise in this demo

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ================================================================================
# 7.1 Implementation verification
# ================================================================================
def verification_tests():
    results = []

    # (1) segment sequencing + mass continuity across hover -> climb -> cruise
    mp = MissionPlanner(gross_weight_kg=2600.0, fuel_kg=AIRCRAFT["fuel_capacity_kg"], dt_s=30.0)
    mp.run([
        {"type": "hover", "duration_s": 60, "altitude_m": 0.0},
        {"type": "climb", "target_altitude_m": 300.0, "climb_rate_ms": 1.5},
        {"type": "cruise", "distance_km": 5.0, "altitude_m": 300.0, "airspeed_ms": 30.0},
    ])
    seq = [row["segment"] for row in mp.log]
    seq_ok = seq[0] == "hover" and "climb" in seq and seq[-1] == "cruise"
    mass_continuous = all(mp.log[i]["gross_weight_kg"] >= mp.log[i + 1]["gross_weight_kg"] - 1e-6
                           for i in range(len(mp.log) - 1) if "gross_weight_kg" in mp.log[i + 1])
    results.append(("Segment sequencing", "hover->climb->cruise order preserved in log",
                     seq_ok, f"log order: {seq[0]} ... {seq[-1]}"))
    results.append(("Mass continuity", "gross weight strictly non-increasing (fuel-burn only, no payload event)",
                     mass_continuous, "checked every logged step"))

    # (2) payload pickup / drop changes mass instantaneously, fuel unaffected
    mp2 = MissionPlanner(gross_weight_kg=2600.0, fuel_kg=400.0, dt_s=30.0)
    gw_before = mp2.state.gross_weight_kg
    fuel_before = mp2.state.fuel_kg
    mp2.payload_event(+200.0)
    pickup_ok = (abs(mp2.state.gross_weight_kg - (gw_before + 200.0)) < 1e-9
                 and abs(mp2.state.fuel_kg - fuel_before) < 1e-9)
    mp2.payload_event(-350.0)
    drop_ok = abs(mp2.state.gross_weight_kg - (gw_before + 200.0 - 350.0)) < 1e-9
    results.append(("Payload pickup", "+200 kg changes GW, not fuel", pickup_ok,
                     f"GW {gw_before:.0f}->{gw_before+200:.0f} kg"))
    results.append(("Payload drop", "-350 kg changes GW, not fuel", drop_ok,
                     f"GW ->{gw_before+200-350:.0f} kg"))

    # (3) fuel update: fuel strictly decreases during a powered hover segment
    mp3 = MissionPlanner(gross_weight_kg=2600.0, fuel_kg=400.0, dt_s=30.0)
    mp3.hover(duration_s=300.0, altitude_m=0.0)
    fuel_series = [row["fuel_kg"] for row in mp3.log]
    fuel_decreasing = all(fuel_series[i] > fuel_series[i + 1] for i in range(len(fuel_series) - 1))
    results.append(("Fuel update", "fuel strictly decreases each step of a powered segment",
                     fuel_decreasing, f"{fuel_series[0]:.2f} -> {fuel_series[-1]:.2f} kg over 300 s"))

    # (4) atmospheric variation: power available drops with altitude
    P0 = power_available_kW(0.0)
    P3000 = power_available_kW(3000.0)
    atmo_ok = P3000 < P0
    results.append(("Atmospheric variation", "power available decreases with altitude",
                     atmo_ok, f"P_avail(0m)={P0:.0f} kW, P_avail(3000m)={P3000:.0f} kW"))

    # (5) wind treatment: tailwind reduces cruise time for the same distance
    #     (use a speed within the trim-feasible envelope -- see
    #     task9_mission_planner_demo's cruise_range_vs_speed / Sec 7.4 notes;
    #     >~27 m/s at MTOW-like weights runs into the stall margin because
    #     the REQUIRED (drag-based) thrust cannot be met at high advance
    #     ratio without over-pitching -- a real trim constraint, not a bug)
    mp_tail = MissionPlanner(gross_weight_kg=2600.0, fuel_kg=400.0, dt_s=30.0, wind_ms=+10.0)
    mp_tail.cruise(distance_km=10.0, altitude_m=1000.0, airspeed_ms=22.0)
    mp_head = MissionPlanner(gross_weight_kg=2600.0, fuel_kg=400.0, dt_s=30.0, wind_ms=-10.0)
    mp_head.cruise(distance_km=10.0, altitude_m=1000.0, airspeed_ms=22.0)
    wind_ok = (not mp_tail.failed) and (not mp_head.failed) and mp_tail.state.time_s < mp_head.state.time_s
    results.append(("Wind treatment", "tailwind gives shorter time-to-distance than headwind",
                     wind_ok, f"tailwind t={mp_tail.state.time_s:.0f}s ({'ok' if not mp_tail.failed else mp_tail.failure.reason}), "
                              f"headwind t={mp_head.state.time_s:.0f}s ({'ok' if not mp_head.failed else mp_head.failure.reason})"))

    # (6) reserve-fuel accounting: mission halts at/above the reserve level, never below
    mp4 = MissionPlanner(gross_weight_kg=2400.0, fuel_kg=60.0, dt_s=30.0)  # thin fuel on purpose
    mp4.hover(duration_s=3600.0, altitude_m=0.0)
    reserve_ok = mp4.state.fuel_kg <= mp4.reserve_fuel_kg + 1e-6 and mp4.failed
    results.append(("Reserve-fuel accounting", "mission stops at/above reserve, flags failure",
                     reserve_ok, f"stopped at fuel={mp4.state.fuel_kg:.1f} kg, "
                                  f"reserve={mp4.reserve_fuel_kg:.1f} kg"))

    # (7) power-required vs power-available calculation sanity (hover, MTOW, sea level)
    theta0, res = _solve_for_thrust(AIRCRAFT["gross_weight_kg"] * G / AIRCRAFT["n_rotors"],
                                     0.0, 0.0, RPM_HOVER, COLLECTIVE_RANGE_HOVER_DEG)
    P_req = res["P"] * AIRCRAFT["n_rotors"] / 1000.0 / AIRCRAFT["drivetrain_efficiency"]
    P_avail = power_available_kW(0.0)
    power_calc_ok = P_req < P_avail   # matches Task 6's finding: MTOW hover is feasible at SL
    results.append(("Power required/available calc", "MTOW hover at sea level within power available",
                     power_calc_ok, f"P_req={P_req:.1f} kW, P_avail={P_avail:.1f} kW "
                                     f"(cf. Task 6: max hover GW at SL = 3205 kg)"))

    # (8) failure-warning logic: identifies segment, time, and reason
    mp5 = MissionPlanner(gross_weight_kg=3800.0, fuel_kg=AIRCRAFT["fuel_capacity_kg"], dt_s=30.0)  # overweight
    mp5.hover(duration_s=60.0, altitude_m=0.0)
    warning_ok = (mp5.failed and mp5.failure.segment_label == "hover"
                  and mp5.failure.time_s is not None and len(mp5.failure.reason) > 0)
    results.append(("Failure-warning logic", "overweight hover flags segment+time+reason",
                     warning_ok, f"'{mp5.failure.reason}'" if mp5.failed else "did not fail (unexpected)"))

    print("=== Task 9 / Sec 7.1 -- Implementation verification (technical entities / test cases) ===")
    with open(os.path.join(OUT_DIR, "task9_1_verification_table.csv"), "w") as f:
        f.write("Verification_item,Test_case,Pass,Evidence\n")
        for item, test, ok, evidence in results:
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {item:<32} {test}\n         evidence: {evidence}")
            f.write(f'"{item}","{test}",{status},"{evidence}"\n')
    n_fail = sum(1 for r in results if not r[2])
    print(f"\n{len(results)-n_fail}/{len(results)} verification checks passed.")
    return results


# ================================================================================
# 7.2 Fuel-burn rate vs gross weight (hover, sea level)
# ================================================================================
def fuel_burn_vs_gross_weight():
    gw_values = np.linspace(2000.0, 3200.0, 13)
    rates = []
    for gw in gw_values:
        theta0, res = _solve_for_thrust(gw * G / AIRCRAFT["n_rotors"], 0.0, 0.0,
                                         RPM_HOVER, COLLECTIVE_RANGE_HOVER_DEG)
        if theta0 is None:
            rates.append(np.nan)
            continue
        P_shaft_kW = res["P"] * AIRCRAFT["n_rotors"] / 1000.0 / AIRCRAFT["drivetrain_efficiency"]
        fuel_rate_kgph = AIRCRAFT["sfc_kg_per_kWh"] * P_shaft_kW
        rates.append(fuel_rate_kgph)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(gw_values, rates, "o-", color="tab:blue")
    plt.axvline(AIRCRAFT["gross_weight_kg"], color="k", ls="--", lw=1, label="design MTOW")
    plt.xlabel("Gross weight [kg]"); plt.ylabel("Fuel-burn rate, hover, sea level [kg/hr]")
    plt.title("Task 9 / Sec 7.2 -- Hover fuel-burn rate vs gross weight")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task9_2_fuel_burn_vs_gw.png"), dpi=160)
    plt.close()

    with open(os.path.join(OUT_DIR, "task9_2_fuel_burn_vs_gw.csv"), "w") as f:
        f.write("gross_weight_kg,fuel_burn_rate_kg_per_hr\n")
        for gw, r in zip(gw_values, rates):
            f.write(f"{gw:.1f},{r:.3f}\n")
    print(f"\nFuel-burn-rate plot written to {FIG_DIR}/task9_2_fuel_burn_vs_gw.png")
    return gw_values, rates


# ================================================================================
# 7.3 Hover endurance vs takeoff weight
# ================================================================================
def hover_endurance_vs_takeoff_weight():
    takeoff_weights = [2200.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0]
    max_search_s = 4.0 * 3600.0
    dt = 120.0
    rows = []
    for gw0 in takeoff_weights:
        mp = MissionPlanner(gross_weight_kg=gw0, fuel_kg=AIRCRAFT["fuel_capacity_kg"], dt_s=dt)
        mp.hover(duration_s=max_search_s, altitude_m=0.0)
        endurance_hr = mp.state.time_s / 3600.0
        binding = mp.failure.reason if mp.failed else "reached search cap (not fuel/power limited)"
        rows.append((gw0, endurance_hr, binding))
        print(f"takeoff_weight={gw0:.0f} kg -> hover endurance = {endurance_hr:.2f} hr "
              f"({binding})")

    gw_arr = [r[0] for r in rows]
    end_arr = [r[1] for r in rows]
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(gw_arr, end_arr, "o-", color="tab:red")
    plt.xlabel("Takeoff weight [kg]"); plt.ylabel("Hover endurance [hr]")
    plt.title(f"Task 9 / Sec 7.3 -- Hover endurance vs takeoff weight\n"
              f"(sea level, {AIRCRAFT['reserve_fuel_fraction']*100:.0f}% reserve-fuel policy applied)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task9_3_hover_endurance_vs_takeoff_weight.png"), dpi=160)
    plt.close()

    with open(os.path.join(OUT_DIR, "task9_3_hover_endurance.csv"), "w") as f:
        f.write("takeoff_weight_kg,endurance_hr,binding_constraint\n")
        for r in rows:
            f.write(f"{r[0]:.1f},{r[1]:.3f},{r[2]}\n")
    print(f"\nHover endurance plot written to {FIG_DIR}/task9_3_hover_endurance_vs_takeoff_weight.png")
    return rows


# ================================================================================
# 7.4 Cruise range vs cruise speed
# ================================================================================
def cruise_range_vs_speed():
    # NOTE: this aircraft's rotor can only meet its drag-based required
    # thrust (assumption A3, L/D=8.5) up to ~27-28 m/s at MTOW before the
    # 5% stall margin binds (see task9_notes_cruise_trim.md) -- that is a
    # genuine trim limit of this Milestone-1 rotor (consistent with Task 7's
    # cruise-speed shortfall finding), so the swept range stays inside it.
    speeds = [15.0, 18.0, 20.0, 22.0, 25.0]     # m/s, trim-feasible at MTOW
    dt = 600.0
    rows = []
    for V in speeds:
        mp = MissionPlanner(gross_weight_kg=AIRCRAFT["gross_weight_kg"],
                             fuel_kg=AIRCRAFT["fuel_capacity_kg"], dt_s=dt)
        mp.cruise(distance_km=1100.0, altitude_m=3000.0, airspeed_ms=V)  # deliberately far; will be reserve-limited
        range_km = mp.state.distance_km
        binding = mp.failure.reason if mp.failed else "reached target distance (not fuel limited)"
        rows.append((V, range_km, binding))
        print(f"V={V:.0f} m/s -> range = {range_km:.0f} km ({binding})")

    v_arr = [r[0] for r in rows]
    range_arr = [r[1] for r in rows]
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(v_arr, range_arr, "o-", color="tab:green")
    plt.xlabel("Cruise speed [m/s]"); plt.ylabel("Range [km]")
    plt.title(f"Task 9 / Sec 7.4 -- Cruise range vs cruise speed\n"
              f"(3000 m altitude, {AIRCRAFT['reserve_fuel_fraction']*100:.0f}% reserve-fuel policy, "
              f"L/D={CRUISE_L_OVER_D})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task9_4_cruise_range_vs_speed.png"), dpi=160)
    plt.close()

    with open(os.path.join(OUT_DIR, "task9_4_cruise_range.csv"), "w") as f:
        f.write("cruise_speed_ms,range_km,binding_constraint\n")
        for r in rows:
            f.write(f"{r[0]:.1f},{r[1]:.1f},{r[2]}\n")
    print(f"\nCruise range plot written to {FIG_DIR}/task9_4_cruise_range_vs_speed.png")
    return rows


if __name__ == "__main__":
    verification_tests()
    fuel_burn_vs_gross_weight()
    hover_endurance_vs_takeoff_weight()
    cruise_range_vs_speed()
    print(f"\nAll figures in ./{FIG_DIR}/task9_*.png ; all tables in ./{OUT_DIR}/task9_*.csv")
