# Validation and regression data

`knight_hefner_table1.csv` contains the digitized Knight & Hefner (1937) Table-I nondimensional hover data used in the Milestone-1 validation package and retained here for continuity.

`m1_regression_expected.csv` contains numerical reference values from the supplied Milestone-1 designed rotor configuration. `tests/test_m1_regression.py` checks the M1 axial BEMT backend against those values so that the M2 edgewise extension does not silently alter the validated M1 solver.
