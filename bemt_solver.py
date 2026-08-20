import numpy as np

class HoverBEMT:
    def __init__(self, R, B, RPM, rho=1.225, N_segments=50):
        self.R = R
        self.B = B
        self.Omega = RPM * (2 * np.pi / 60.0) # Convert RPM to rad/s
        self.rho = rho
        self.N = N_segments
        
        # Create radial segments (avoiding exact 0 at root for mathematical stability)
        self.r = np.linspace(0.1 * R, R, self.N)
        self.dr = self.r[1] - self.r[0]

    def get_aero_coeffs(self, alpha):
        # Knight & Hefner Airfoil Polars
        Cl = 5.75 * alpha
        Cd = 0.0113 + 1.25 * (alpha**2)
        return Cl, Cd

    def solve(self, chord_dist, theta_dist):
        T_total = 0.0
        Q_total = 0.0
        
        # Loop through each blade element
        for i in range(self.N):
            r_loc = self.r[i]
            c = chord_dist[i]
            theta = theta_dist[i]
            
            Ut = self.Omega * r_loc
            vi = 0.0 # Initial guess for induced velocity
            k = 0.3  # Relaxation factor for stability
            
            # Iterative solver for induced velocity
            for iteration in range(100):
                phi = np.arctan(vi / Ut)
                alpha = theta - phi
                
                Cl, Cd = self.get_aero_coeffs(alpha)
                U = np.sqrt(Ut**2 + vi**2)
                
                # Axial force coefficient
                Cy = Cl * np.cos(phi) - Cd * np.sin(phi)
                
                # Momentum Theory Right Hand Side (Hover: Vc = 0, F = 1)
                RHS = (self.B * c * (U**2) * Cy) / (8 * np.pi * r_loc)
                
                # Calculate new induced velocity (sqrt of RHS since Vc = 0)
                if RHS < 0:
                    vi_new = 0 # Catch for negative thrust anomalies
                else:
                    vi_new = np.sqrt(RHS)
                    
                # Check for convergence
                if abs(vi_new - vi) < 1e-5:
                    vi = vi_new
                    break
                    
                # Relaxation update
                vi = (1 - k) * vi + k * vi_new
            
            # -----------------------------------------
            # Calculate final forces for this element
            # -----------------------------------------
            phi = np.arctan(vi / Ut)
            alpha = theta - phi
            Cl, Cd = self.get_aero_coeffs(alpha)
            U = np.sqrt(Ut**2 + vi**2)
            
            dL = 0.5 * self.rho * (U**2) * c * Cl * self.dr
            dD = 0.5 * self.rho * (U**2) * c * Cd * self.dr
            
            dT = self.B * (dL * np.cos(phi) - dD * np.sin(phi))
            dQ = self.B * r_loc * (dL * np.sin(phi) + dD * np.cos(phi))
            
            T_total += dT
            Q_total += dQ
            
        # Total Power (P = Omega * Q)
        P_total = self.Omega * Q_total
        
        return T_total, Q_total, P_total

# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # 1. Define Test Parameters
    RADIUS = 2.0      # meters
    BLADES = 2
    RPM = 1200
    N_SEGMENTS = 50
    
    # 2. Initialize the Solver
    solver = HoverBEMT(R=RADIUS, B=BLADES, RPM=RPM, N_segments=N_SEGMENTS)
    
    # 3. Define Blade Geometry (Constant chord and twist for this quick test)
    chord_distribution = np.full(N_SEGMENTS, 0.15)       # 15cm chord everywhere
    pitch_distribution = np.full(N_SEGMENTS, np.radians(8)) # 8 degrees pitch everywhere
    
    # 4. Run the Solver
    Thrust, Torque, Power = solver.solve(chord_distribution, pitch_distribution)
    
    # 5. Print Results
    print(f"--- BEMT Hover Results ---")
    print(f"Total Thrust: {Thrust:.2f} N")
    print(f"Total Torque: {Torque:.2f} Nm")
    print(f"Total Power:  {Power / 1000:.2f} kW")