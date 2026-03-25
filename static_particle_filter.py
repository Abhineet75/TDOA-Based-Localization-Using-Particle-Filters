import numpy as np



# Point class for both transmitter and receiver

class Point:
    def __init__(self, role="Unknown", x=0.0, y=0.0, z=0.0):
        self.role = role
        self.X = x
        self.Y = y
        self.Z = z  

    def euc_dist(self, other):
        return np.sqrt((self.X - other.X)**2 + (self.Y - other.Y)**2 + (self.Z - other.Z)**2)



class StaticParticleFilterTDOA:


    # Class Constructor
    def __init__(self, receivers, base_variance, num_particles=1000, c_r=0.01, c_q=0.01):

        self.receivers = receivers
        self.num_particles = num_particles
        self.C_R = c_r
        self.C_Q = c_q
        self.base_variance = base_variance
        
        # Initialising the particles in a 1x1x1 cube
        # Uniformly distributed Transmitters
        self.particles = [
            Point("Transmitter", np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            for _ in range(num_particles)
        ]
 
    def _build_base_covariance_matrix(self):

        """
        Building the (N-1)x(N-1) covariance matrix anchored to receiver 0
        """

        # The diagonals 
        
        n_measurements = len(self.receivers) - 1
        R_matrix = np.full((n_measurements, n_measurements), self.base_variance)
        np.fill_diagonal(R_matrix, 2 * self.base_variance)
        
        return R_matrix

    def _get_h_vector(self, particle):
        """
        Computes the predicted h(P_i) vector for a single particle relative to Receiver 0.
        used for mahalnobis distance
        """
        dist_0 = particle.euc_dist(self.receivers[0])
        h_vec = []
        for i in range(1, len(self.receivers)):
            # Geometric path difference relative to reference receiver 0
            h_vec.append(particle.euc_dist(self.receivers[i]) - dist_0)
        return np.array(h_vec).reshape(-1, 1)

    def run_filter(self, delta_d, true_tx_coords, iterations=100):
        """
        
        delta_d: A numpy column vector of shape (N-1, 1) containing observed TDOA path differences.
        """
        R_base = self._build_base_covariance_matrix()
        error_history = []



        for k in range(1, iterations + 1):
            # Step 2: Choose jittering constants for this iteration
            Q_bar = max(self.C_Q / (k**2), 0.00001)
            
            # R_bar = R + (C_R / k^2) 
            jitter_matrix = np.eye(len(self.receivers) - 1) * (self.C_R / (k**2))
            R_bar = R_base + jitter_matrix
            R_inv = np.linalg.inv(R_bar)
            
            weights = np.zeros(self.num_particles)
            
            # particle weights using Mahalanobis distance
            for i, p in enumerate(self.particles):
                h_p = self._get_h_vector(p)
                innovation = delta_d - h_p
                
                # Matrix multiplication: (Δd - h(P))^T * R_inv * (Δd - h(P))
                mahalanobis_sq = innovation.T @ R_inv @ innovation
                
                # get likelihood
                weights[i] = np.exp(-0.5 * mahalanobis_sq[0, 0]) 
                
            # Normalize to get probability
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(self.num_particles) / self.num_particles

            # giving an estimate of position
            ex = sum(weights[i] * self.particles[i].X for i in range(self.num_particles))
            ey = sum(weights[i] * self.particles[i].Y for i in range(self.num_particles))
            ez = sum(weights[i] * self.particles[i].Z for i in range(self.num_particles))
            
            print(f"Iteration {k:03d} | Estimate: ({ex:.4f}, {ey:.4f}, {ez:.4f})")

            current_error = np.sqrt((ex - true_tx_coords[0])**2 + (ey - true_tx_coords[1])**2 + (ez - true_tx_coords[2])**2)
            error_history.append(current_error)

            # Resampling step
            indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=weights)
            
        
            std_dev = np.sqrt(Q_bar)
            new_particles = []
            
            for idx in indices:
                old_p = self.particles[idx]
             
                nx = np.clip(old_p.X + np.random.normal(0, std_dev), 0, 1)
                ny = np.clip(old_p.Y + np.random.normal(0, std_dev), 0, 1)
                nz = np.clip(old_p.Z + np.random.normal(0, std_dev), 0, 1)
                new_particles.append(Point("Particle", nx, ny, nz))
                
            self.particles = new_particles
            
        return Point("Estimate", ex, ey, ez), error_history


# --- Execution Example ---
if __name__ == "__main__":
    receivers = [
        Point("Receiver", 0.86, 0.79, 0.40), # Receiver 0 (Reference)
        Point("Receiver", 0.24, 0.10, 0.25),
        Point("Receiver", 0.50, 0.11, 0.52),
        Point("Receiver", 0.89, 0.38, 0.07)
    ]

    # Your D_NP matrix had Receiver 0 on the first row/column.
    # We extract the path differences relative to Receiver 0: d_10, d_20, d_30
    # From your D_NP array: D_NP[1][0], D_NP[2][0], D_NP[3][0]
    measured_delta_d = np.array([
        [-0.23448966], 
        [-0.16878072], 
        [-0.15686472]
    ])

    pf = StaticParticleFilterTDOA(
        receivers=receivers,  
        base_variance=0.001, 
        num_particles=1000
    )
    
    final_estimate = pf.run_filter(delta_d=measured_delta_d, iterations=50)
