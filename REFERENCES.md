# References (consolidated -- paste/reformat onto Slide 37)

Re-verify each of these against the primary source yourselves before citing
in your submitted report; this list only records what this code package
drew numbers from and where to find them again.

1. Knight, M., and Hefner, R. A., "Static Thrust Analysis of the Lifting
   Airscrew," NACA Technical Note No. 626, Dec. 1937. Full scan (public
   domain, NASA Technical Reports Server):
   https://ntrs.nasa.gov/api/citations/19930081433/downloads/19930081433.pdf
   -- source of the Task 3 validation dataset (Table I, 2-blade rotor,
   sigma=0.0424) and the linear-airfoil coefficients (a0=5.75, Cd_min=0.0113,
   eps=1.25) given in the Milestone-1 handout.

2. Bell XV-15 Tilt Rotor Research Aircraft geometry (radius, chord, solidity,
   twist) -- cited via Wang, Y. et al., "Assessment of Detached Eddy
   Simulation for Tiltrotor Performance," arXiv:2201.11560 (2022), which
   reproduces the XV-15 rotor design literature values used in Task 8.

3. XV-15 rotor speed schedule (589 RPM hover / 517 RPM cruise) and design
   gross weight range -- DTIC report ADA123857 ("XV-15 Tilt Rotor Flight
   Test..."-class documents) and the Smithsonian National Air and Space
   Museum XV-15 aircraft record (airandspace.si.edu).

4. XV-15 full-scale hover-test rotor tip Mach range (0.60-0.73) --
   Shinoda, P. / Betzina, M., full-scale XV-15 hover-test reports, NASA
   Ames Outdoor Aerodynamic Research Facility (OARF), NASA Technical Reports
   Server.

5. Bell-Boeing V-22 Osprey geometry (11.58 m / 38 ft rotor diameter, 3
   blades, twin rotors) and MTOW (~27,443 kg / 60,500 lb VTOL) -- V-22
   Osprey fact sheets, globalsecurity.org and man.fas.org "V-22 Osprey"
   reference pages.

6. "The V-22 Tilt-Rotor Large-Scale Rotor Performance" large-scale
   powered-model wind-tunnel test report -- source of the V-22 isolated-
   rotor maximum hover Figure of Merit (0.808) used in Task 8, as cited by
   multiple secondary CFD-validation papers referencing the same test.

7. MathWorks Aerospace Blockset documentation, "Computation of Thrust and
   Torque" (combined blade-element/momentum-theory equations, Prandtl tip
   loss): https://www.mathworks.com/help/aeroblks/computation-of-thrust-and-torque.html
   -- used during development to independently cross-check the BEMT
   solver's governing equations (see `dev_notes_bemt_quirks.md`).

8. Software: Python 3, NumPy, SciPy (`scipy.optimize.brentq` for all
   root-finding), Matplotlib (all figures).
