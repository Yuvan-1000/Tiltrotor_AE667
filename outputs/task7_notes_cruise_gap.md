# Task 7 technical note: cruise-speed requirement vs. demonstrated capability

NOTE: this feasible cruise speed (36.6 m/s = 71 kt) is well below the 110 m/s (214 kt) mission requirement stated in Task 5. Root cause: this rotor's twist (-12 deg, chosen in Task 5 for hover numerical robustness -- see that file's docstring) does not carry enough built-in washout to keep the inboard stations unstalled at the high advance ratio a 110 m/s cruise would require at this RPM. A production-
representative fix (more washout, e.g. -30 to -40 deg as on the XV-15, paired with a re-derived, higher hover collective schedule) is exactly the kind of rotor redesign flagged for Milestone 2 in Sec 7.5 -- do not silently raise AIRCRAFT['demonstrated_cruise_speed_ms'] without re-running Tasks 6-7.
