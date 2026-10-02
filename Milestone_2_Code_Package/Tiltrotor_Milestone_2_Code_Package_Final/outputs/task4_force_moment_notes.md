# Task 4 — rotor forces and moments

The edgewise solver returns forces/moments in rotor coordinates, transforms them into aircraft body coordinates using the nacelle direction-cosine matrix, and then forms the moment about the CG as M_CG = M_hub + r_hub × F. The twin-rotor aircraft uses opposite rotation signs, so the steady shaft reaction torque cancels in the symmetric pair.
