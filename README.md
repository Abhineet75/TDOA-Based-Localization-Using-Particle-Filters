# Static Particle Filter for Localization of Unknown Transmitter using TDOA Measurements


### Aim
Using a number of receivers (sensors), and the Time-Difference-of-Arrival measurements, we need to localize the transmitter of the signal. 

### Method
We use Monte Carlo Simulations to solve this problem – this is a variant of the Particle Filter approach, a Static Particle Filter. Resampling is done using the non-linear least squares methodology for assigning weights to different initial transmitter locations. 

### Implementation
This project is coded in Python. Mainly NumPy has been used for quick vectorized operations. 

