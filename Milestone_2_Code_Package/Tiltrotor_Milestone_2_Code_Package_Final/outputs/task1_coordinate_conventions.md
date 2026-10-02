# Task 1 — Coordinate systems, transformations and conventions

The aircraft uses a right-handed NED inertial frame and body frame. Positive pitch is nose-up. Rotor +z_R points in the thrust direction. The nacelle angle γ is 90° in helicopter mode and 0° in airplane mode. The positive azimuth direction follows the rotor rotation sense and ψ=90° is the advancing side for the representative rotor.

The course cyclic convention is θ(ψ)=θ0+θ1c cosψ+θ1s sinψ. In the AE 667 lecture θ1c denotes lateral cyclic and θ1s denotes longitudinal cyclic.

The rotor-frame force vector is transformed to body axes with C_BR. The hub moment reported about the aircraft CG is M_CG = C_BR M_hub,R + r_hub × F_body. For the symmetric twin-rotor aircraft, the two rotors use opposite rotation signs so shaft reaction torque cancels in steady flight.
