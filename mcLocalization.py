
# Static particle filter

import numpy as np
from Point import Point # I guess the whole class gets imported this way






# Hyperparameters
NUM_TRANSMITTERS = 1000  # Number of Transmitters being simulated, there is 1 finally
NUM_RECEIVERS = 4
WAVE_SPEED = 3


C_R = 0.01 # Jittering constant for measurement
C_Q = 0.01 # Jittering constant for position
R = 0.001  # Base measurement noise covariance

# TDOA_MEASUREMENTS[i][j] = time difference in measurement between i and j receivers

# We need to get these measurements in real life
# Empty square matrix for now
TDOA_MEASUREMENTS = [[0] * (NUM_RECEIVERS) for i in range(NUM_RECEIVERS)]
    
# for actual 
DEL = TDOA_MEASUREMENTS * WAVE_SPEED


# for testing
D = [[0] * (NUM_RECEIVERS) for i in range(NUM_RECEIVERS)]


transmitters = [] # list of points
receivers = []

for i in range(NUM_TRANSMITTERS):
    transmitters.append(Point("Transmitter"))

for i in range(NUM_RECEIVERS):
    receivers.append(Point("Receiver"))


receivers[0] = Point("Receiver", 0.86, 0.79, 0.40)
receivers[1] = Point("Receiver", 0.24, 0.10, 0.25)
receivers[2] = Point("Receiver", 0.50, 0.11, 0.52)
receivers[3] = Point("Receiver", 0.89, 0.38, 0.07)

# for r in receivers:
#     r.info()


"""
For each transmitter, for each pair of receivers, we need to calculate the squared
delta function

Okay but how do we actually get the tdoa measurements?
easy

Receiver : (0.86, 0.79, 0.40)
Receiver : (0.24, 0.10, 0.25)
Receiver : (0.50, 0.11, 0.52)
Receiver : (0.89, 0.38, 0.07)
Transmitter : (0.46, 0.39, 0.19)

np.array([0.46, 0.39, 0.19])

np.array([[0.86, 0.79, 0.40],[0.24, 0.10, 0.25],[0.50, 0.11, 0.52],[0.89, 0.38, 0.07]])

"""

D_NP = np.array([[-0, 0.23448966,  0.16878072,  0.15686472],
 [-0.23448966, -0, -0.06570894, -0.07762494],
 [-0.16878072,  0.06570894, -0   ,      -0.01191599],
 [-0.15686472 , 0.07762494 , 0.01191599, -0        ]])


print(D_NP)

# this gives the cost associated with the prediction that is the ttr_index'd transmitter
def delta_ttr(ttr_index):

    delta = 0

    for i in range(0, NUM_RECEIVERS):
        for j in range(i+1, NUM_RECEIVERS):
            H_P = (transmitters[ttr_index].euc_dist(receivers[i]) - transmitters[ttr_index].euc_dist(receivers[j])) 
            # D[i][j] = H_P
            # D[j][i] = D[i][j]
            #delta += (H_P - DEL[i][j])**2
            delta += (H_P - D[i][j])**2
    return delta




def get_weights(R_bar): # Renamed to avoid shadowing
    cost = [delta_ttr(i) for i in range(NUM_TRANSMITTERS)]
    # Use np.exp on the whole array for speed
    W = np.exp(-0.5 * np.array(cost) / R_bar)
    return W

def normalize(A):
    # Ensure it's a numpy array for vector math
    A = np.array(A)
    total = np.sum(A)
    if total == 0: # Safety check
        return np.ones(len(A)) / len(A)
    return A / total


SIMULATION_ROUNDS = 100

for k in range(1, SIMULATION_ROUNDS + 1):
    R_bar = R + (C_R / k**2)
    Q_bar = max(C_Q / k**2, 0.00001) # Floor to keep it moving
    std = np.sqrt(Q_bar)

    # 3a & 3b: Weights and Estimate
    w_list = get_weights(R_bar)
    w_list = normalize(w_list)

    ex = sum(w_list[i] * transmitters[i].X for i in range(NUM_TRANSMITTERS))
    ey = sum(w_list[i] * transmitters[i].Y for i in range(NUM_TRANSMITTERS))
    ez = sum(w_list[i] * transmitters[i].Z for i in range(NUM_TRANSMITTERS))

    
    print(f"Iteration {k} | Estimate: ({ex:.4f}, {ey:.4f}, {ez:.4f})")

    # 3c: Resample - Create a list of NEW Point objects immediately
    indices = np.random.choice(range(NUM_TRANSMITTERS), size=NUM_TRANSMITTERS, p=w_list)
    
    # We create a brand new list of Points to ensure total independence
    new_gen = []
    for idx in indices:
        p = transmitters[idx]
        nx = np.clip(p.X + np.random.normal(0, std), 0, 1)
        ny = np.clip(p.Y + np.random.normal(0, std), 0, 1)
        nz = np.clip(p.Z + np.random.normal(0, std), 0, 1)
        new_gen.append(Point("Transmitter", nx, ny, nz))
    
    # have a new set of points
    transmitters = new_gen





# better to write it in class format now to be able to modularize it and test it better.