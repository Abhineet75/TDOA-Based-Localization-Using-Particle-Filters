import numpy as np
import matplotlib.pyplot as plt

# Import your classes from the main file
from Positioning.static_particle_filter import Point, StaticParticleFilterTDOA

# --- Helper Function ---
def generate_true_delta_d(tx_coords, receivers):
    """
    Generates the true (N-1)x1 column vector of TDOA measurements 
    anchored to receiver 0, simulating ideal hardware.
    """
    tx_arr = np.array(tx_coords)
    # Distance from Transmitter to Reference Receiver 0
    dist_0 = np.linalg.norm(tx_arr - np.array([receivers[0].X, receivers[0].Y, receivers[0].Z]))
    
    delta_d = []
    for i in range(1, len(receivers)):
        dist_i = np.linalg.norm(tx_arr - np.array([receivers[i].X, receivers[i].Y, receivers[i].Z]))
        delta_d.append([dist_i - dist_0])
        
    return np.array(delta_d)


# --- Simulation Setup ---
TRUE_TX = [0.46, 0.39, 0.19]
ITERATIONS = 150
NUM_PARTICLES = 1000
BASE_VARIANCE = 0.001

# Define the Geometries (Using your Point class)
geometries = {
    "Max Volume (Corners)": [
        Point("Rx0", 0.0, 0.0, 0.0), 
        Point("Rx1", 1.0, 1.0, 0.0), 
        Point("Rx2", 1.0, 0.0, 1.0), 
        Point("Rx3", 0.0, 1.0, 1.0)
    ],
    "Dome (Ground + 1 Elevated)": [
        Point("Rx0", 0.0, 0.0, 0.0), 
        Point("Rx1", 1.0, 0.0, 0.0), 
        Point("Rx2", 0.0, 1.0, 0.0), 
        Point("Rx3", 0.5, 0.5, 0.2)
    ],
    "Tight Cluster (Poor Baseline)": [
        Point("Rx0", 0.0, 0.0, 0.0), 
        Point("Rx1", 0.1, 0.0, 0.0), 
        Point("Rx2", 0.0, 0.1, 0.0), 
        Point("Rx3", 0.0, 0.0, 0.1)
    ],
    "Coplanar (XY Plane)": [
        Point("Rx0", 0.0, 0.0, 0.0), 
        Point("Rx1", 1.0, 0.0, 0.0), 
        Point("Rx2", 0.0, 1.0, 0.0), 
        Point("Rx3", 1.0, 1.0, 0.0)
    ],

    "The Line (Collinear X-axis)": [
        # The classic failure mode. Will fail to resolve Y and Z.
        Point("Rx0", 0.0, 0.0, 0.0), 
        Point("Rx1", 0.33, 0.0, 0.0), 
        Point("Rx2", 0.66, 0.0, 0.0), 
        Point("Rx3", 1.0, 0.0, 0.0)
    ],
    "The Wall (Coplanar XZ Plane)": [
        # Simulates receivers mounted flat on a wall. 
        # Will perfectly find X and Z, but will fail to tell if Y is positive or negative.
        Point("Rx0", 0.0, 0.0, 0.0), 
        Point("Rx1", 1.0, 0.0, 0.0), 
        Point("Rx2", 0.0, 0.0, 1.0), 
        Point("Rx3", 1.0, 0.0, 1.0)
    ],
    "The Origin Corner (Orthogonal Axes)": [
        # One at the origin, the others stretching down the X, Y, and Z axes.
        # A very solid, standard baseline setup.
        Point("Rx0", 0.0, 0.0, 0.0), 
        Point("Rx1", 1.0, 0.0, 0.0), 
        Point("Rx2", 0.0, 1.0, 0.0), 
        Point("Rx3", 0.0, 0.0, 1.0)
    ],
    "The Perfect Tetrahedron": [
        # Mathematically optimal 3D spread. This maximizes the volume
        # enclosed by the receivers, yielding the sharpest TDOA intersections.
        Point("Rx0", 0.0, 0.0, 0.0), 
        Point("Rx1", 1.0, 0.0, 0.0), 
        Point("Rx2", 0.5, np.sqrt(3)/2, 0.0), 
        Point("Rx3", 0.5, np.sqrt(3)/6, np.sqrt(2/3))
    ]


}

# --- Run Tests ---
results = {}

print(f"Target Transmitter Location: {TRUE_TX}")
print("-" * 40)

for name, rx_list in geometries.items():
    print(f"Testing: {name}")
    
    # 1. Generate the true measurements for this specific geometry
    measured_delta_d = generate_true_delta_d(TRUE_TX, rx_list)
    
    # 2. Instantiate the filter
    pf = StaticParticleFilterTDOA(
        receivers=rx_list, 
        base_variance=BASE_VARIANCE, 
        num_particles=NUM_PARTICLES
    )
    
    # 3. Run the filter and capture the error history
    final_est, error_hist = pf.run_filter(
        delta_d=measured_delta_d, 
        true_tx_coords=TRUE_TX, 
        iterations=ITERATIONS
    )
    
    results[name] = error_hist
    print(f"Final Estimate: ({final_est.X:.4f}, {final_est.Y:.4f}, {final_est.Z:.4f})")
    print(f"Final Error: {error_hist[-1]:.4f}\n")



# --- Plotting the Results ---
plt.figure(figsize=(14, 8)) 

# Get a colormap that has a large number of distinct colors (tab20 has 20)
colormap = plt.get_cmap('tab20')

# Use enumerate to dynamically assign a color to each geometry, no matter how many there are
for i, (name, error_hist) in enumerate(results.items()):
    # The modulo (%) ensures that if you test more than 20 geometries, the colors safely loop
    color = colormap(i % 20) 
    
    plt.plot(range(1, ITERATIONS + 1), error_hist, label=name, linewidth=2.5, color=color, alpha=0.85)

plt.title("Particle Filter Convergence by Receiver Geometry", fontsize=16, fontweight='bold')
plt.xlabel("Simulation Iteration (k)", fontsize=12)
plt.ylabel("Euclidean Error (m)", fontsize=12)
plt.ylim(-0.05, 1.0) # Lock Y-axis to see the contrast clearly
plt.grid(True, linestyle='--', alpha=0.6)

# Move the legend outside the graph so it doesn't block the data curves
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=11)
plt.tight_layout()

plt.show()
