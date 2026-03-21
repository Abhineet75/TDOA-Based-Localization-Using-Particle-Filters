Here is the updated README with the new section on geometric testing integrated right before the Usage and Dependencies sections. 

***

# TDOA-Based Static Transmitter Localization using Simulated Annealing Particle Filters

## Overview

This repository contains a robust Python implementation of a Time Difference of Arrival (TDOA) localization algorithm. It utilizes a Simulated Annealing Particle Filter (SAPF) to accurately estimate the spatial coordinates of a static radio frequency (RF) transmitter using a network of distributed receivers. 

This work is based on the contents of the paper: *POSITIONING USING TIME-DIFFERENCE OF ARRIVAL MEASUREMENTS*, by Fredrik Gustafsson and Fredrik Gunnarsson, and marks my first effort in an end-to-end research to implementation, which I did with the help of one of my signal processing professors at IIT Delhi.  

Standard particle filters suffer heavily from particle impoverishment when tracking static targets in highly non-convex TDOA cost functions. This project solves that by dynamically modulating the measurement covariance and process noise over time, allowing the particles to effectively search the global space before converging on the Minimum Mean Square Error (MMSE) estimate.

## Core Mathematical Concepts

### 1. The N-1 Reference Anchor & Linear Independence

To prevent catastrophic singular matrix errors ($\det(R) = 0$) during the Mahalanobis distance calculation, this algorithm avoids feeding redundant relative time measurements into the covariance matrix.

For an $N$-receiver network, calculating the path difference between every possible pair yields $\binom{N}{2}$ measurements, which are linearly dependent. This implementation strictly anchors measurements to a single reference receiver (e.g., $Rx_0$), reducing the measurement vector to $N-1$ strictly independent variables. This guarantees a full-rank, invertible $(N-1) \times (N-1)$ covariance matrix.

### 2. Simulated Annealing Particle Filter (SAPF)

Because the target is static, standard kinematic process noise cannot be used to maintain particle diversity. Instead, the filter artificially manipulates noise levels as a function of the iteration step $k$:

- **Cost-Function Flattener (Measurement Jittering):**
    
    $$R_{bar} = R + \frac{C_R}{k^2}$$
    
    At early iterations, the base hardware measurement covariance ($R$) is artificially inflated by $C_R$. This "flattens" the local minima in the TDOA geometry, preventing the algorithm from over-trusting early measurements and getting trapped.
    
- **Search Radius (Position Jittering):**
    
    $$Q_{bar} = \max\left(\frac{C_Q}{k^2}, 0.00001\right)$$
    
    After resampling, particles are perturbed by $Q_{bar}$. At $k=1$, particles take large random walks to explore the grid. As $k \to \infty$, $Q_{bar}$ decays to a near-zero floor, shifting the system from global exploration to localized exploitation.
    

### 3. Bayesian Weighting & The MMSE Estimator

Particle weights are derived using the Mahalanobis distance to account for cross-correlation in the $N-1$ measurement vector. The unnormalized likelihood is calculated assuming Additive White Gaussian Noise (AWGN):

$$w_i^{raw} \propto \exp\left( -\frac{1}{2} (\Delta d - h(x_i))^T R_{bar}^{-1} (\Delta d - h(x_i)) \right)$$

Weights are then normalized to form a valid Posterior Probability Mass Function (PMF). The final spatial prediction is extracted by calculating the Expected Value of the posterior, which mathematically serves as the optimal Minimum Mean Square Error (MMSE) estimator:

$$\hat{x}_{MMSE} = \sum_{i=1}^{N_{particles}} w_i^{norm} \cdot x_i$$

## Geometric Dilution of Precision (GDOP) Testing

A dedicated testing suite (`geom_testing.py`) is included to evaluate the filter's performance across various physical receiver configurations. Because TDOA relies on the intersection of hyperboloids, poor geometric baselines can severely degrade accuracy regardless of algorithmic robustness. 

The simulation results highlight key failure modes and optimal setups:

* **The Perfect Tetrahedron & Max Volume (Corners):** Yields the fastest convergence and lowest steady-state error. Maximizing the enclosed 3D volume provides the sharpest mathematical intersections.
* **The Dome & Origin Corner:** Standard configurations that converge reliably, though with a slightly higher error floor compared to a mathematically perfect tetrahedron.
* **Coplanar (XY Plane / XZ Wall):** Demonstrates a predictable partial failure. The filter accurately resolves the two axes spanning the plane but fails on the orthogonal axis due to symmetrical ambiguity (e.g., unable to distinguish between $+Z$ and $-Z$).
* **Collinear (The Line):** Results in catastrophic tracking failure. The algorithm cannot resolve the rotational ambiguity around the axis of the receivers, often collapsing the particles into an infinite "ring" of zero-error rather than a single point.
* **Tight Cluster:** Simulates a poor baseline relative to the target distance. The true time differences become so infinitesimally small that the filter's artificial process noise dominates, resulting in high variance and a failure to settle.

## Usage & Tuning

The filter is encapsulated in an object-oriented structure to allow for easy hyperparameter tuning based on the physical realities of the hardware sensors and the scale of the deployment environment.

- `R`: Set this to the physical baseline variance of the TDOA sensors ($\sigma^2$).
- `C_Q`: Scale this based on the bounding box of the search space. $\sqrt{C_Q}$ should cover a meaningful percentage of the grid to ensure adequate early exploration.
- `C_R`: Tune this for convergence speed. Increase to force longer exploration; decrease if particles are failing to settle.
    
## Dependencies

- `numpy`: Matrix operations, linear algebra (inversions), and vectorized stochastic sampling.
- `matplotlib`: Visualization of geometric convergence curves.
    
## Author

**Abhineet Milind More**
