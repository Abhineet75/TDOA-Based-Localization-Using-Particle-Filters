import numpy as np
import matplotlib.pyplot as plt
from poshelpers import gen_tdoa_measurements, gen_receiver_pos_arc

from Point import Point # I guess the whole class gets imported this way


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


# XZ plane: np.array([[0, 0, 0],[1, 0, 0],[0, 0, 1],[1, 0, 1]])
# XY plane: np.array([[0, 0, 0],[1, 0, 0],[1, 1, 0],[0, 1, 1]])
# Line on X-axis: np.array([[0, 0, 0],[0.25, 0, 0],[0.5, 0, 0],[0.75, 0, 0]])
# Tetrahedron around the point: np.array([[ 1.46,  1.39,  1.19],[ 1.46, -0.61, -0.81], [-0.54,  1.39, -0.81], [-0.54, -0.61,  1.19]])
"""
tetrahedron = np.array([
    [ 1.46,  1.39,  1.19],
    [ 1.46, -0.61, -0.81],
    [-0.54,  1.39, -0.81],
    [-0.54, -0.61,  1.19]
])
"""

RP1 = np.array([[0.86, 0.79, 0.40],[0.24, 0.10, 0.25],[0.50, 0.11, 0.52],[0.89, 0.38, 0.07]])
TX = np.array([0.46, 0.39, 0.19])

TDOA_X =  np.array([[-0, 0.23448966,  0.16878072,  0.15686472],
 [-0.23448966, -0, -0.06570894, -0.07762494],
 [-0.16878072,  0.06570894, -0   ,      -0.01191599],
 [-0.15686472 , 0.07762494 , 0.01191599, -0        ]])



class SPF:

    def __init__(self, n_trans=1000, n_rec=4, wave_speed=1, j_c_r=0.001, j_c_q=0.01, r_base=0.001, tdoa_meas = TDOA_X, receiver_positions= RP1, scale=1):

        self.NUM_TRANSMITTERS = n_trans
        self.NUM_RECEIVERS = n_rec
        self.WAVE_SPEED = wave_speed
        
        self.C_R = j_c_r
        self.C_Q = j_c_q
        self.R = r_base
        self.SCALE = scale

        self.TDOA_MEASUREMENTS = tdoa_meas
        self.RECEIVERS = receiver_positions

        self.TRANSMITTERS = np.zeros(shape = (self.NUM_TRANSMITTERS, 3))

        for i in range(self.NUM_TRANSMITTERS):
            self.TRANSMITTERS[i] = np.random.rand(3) * self.SCALE


    def euc_dist(A, B):
        # both have shape (3)
        # do it the numpy way
        return np.linalg.norm(A - B)

    def delta_tmtr(self, tmtr_index):
        delta = 0

        for i in range(0, self.NUM_RECEIVERS):
            for j in range(i+1, self.NUM_RECEIVERS):

                D_i = np.linalg.norm(self.TRANSMITTERS[tmtr_index] - self.RECEIVERS[i])
                D_j = np.linalg.norm(self.TRANSMITTERS[tmtr_index] - self.RECEIVERS[j])
                H_P = D_i - D_j

                # note in TDOA_MEASUREMENTS I actually have the distances
                delta += (H_P - (self.TDOA_MEASUREMENTS[i][j]))**2
        
                # D[i][j] = H_P
                # D[j][i] = D[i][j]
                #delta += (H_P - DEL[i][j])**2
        
        return delta


    def get_weights(self, R_bar): # Renamed to avoid shadowing
        
        cost = [self.delta_tmtr(i) for i in range(self.NUM_TRANSMITTERS)]
        # Use np.exp on the whole array for speed
        W = np.exp(-0.5 * np.array(cost) / R_bar)
        return W

    def normalize(self, A):
        # Ensure it's a numpy array for vector math
        A = np.array(A)
        total = np.sum(A)
        if total == 0: # Safety check
            return np.ones(len(A)) / len(A)
        return A / total




    def simulate(self, sim_rounds=100, make_tdoa=False, show_plot=True):

        if make_tdoa:
            self.TDOA_MEASUREMENTS = gen_tdoa_measurements(transmitter= TX, receivers=self.RECEIVERS, sos=1)

        # For graphing the error
        error = 0
        error_y = np.zeros(sim_rounds)
        error_x = np.arange(sim_rounds)

        SIMULATION_ROUNDS = sim_rounds

        # Starting the simulation

        for k in range(1, SIMULATION_ROUNDS + 1):
            R_bar = self.R + (self.C_R / k**2)
            Q_bar = self.C_Q / k**2 
            std = np.sqrt(Q_bar)

            
            w_list = self.get_weights(R_bar)
            w_list = self.normalize(w_list)



            ex = sum(w_list[i] * self.TRANSMITTERS[i][0] for i in range(self.NUM_TRANSMITTERS))
            ey = sum(w_list[i] * self.TRANSMITTERS[i][1] for i in range(self.NUM_TRANSMITTERS))
            ez = sum(w_list[i] * self.TRANSMITTERS[i][2] for i in range(self.NUM_TRANSMITTERS))

            ec = np.linalg.norm((np.array([0.46, 0.39, 0.19]) - np.array([ex, ey, ez])))
            error_y[k - 1] = ec
            error += ec

            
            print(f"Iteration {k} | Estimate: ({ex:.4f}, {ey:.4f}, {ez:.4f}) | Error : {ec}")

            # 3c: Resample - Create a list of NEW Point objects immediately
            indices = np.random.choice(range(self.NUM_TRANSMITTERS), size=self.NUM_TRANSMITTERS, p=w_list)
            
            new_gen = []

            for idx in indices:
                p = self.TRANSMITTERS[idx]
                nx = p[0] + np.random.normal(0, std)
                ny = p[1] + np.random.normal(0, std)
                nz = p[2] + np.random.normal(0, std)
                
                #nx, ny, nz = p[0], p[1], p[2]
                new_gen.append([nx, ny, nz])
            
            # have a new set of points
            self.TRANSMITTERS = np.array(new_gen)

        print(f"Total error: {error}")


        if show_plot:
            plt.plot(error_x, error_y)
            plt.show()

        return error_y



    # call the simulate function on different tests and see the final graph

tetrahedron = np.array([
    [ 1.46,  1.39,  1.19],
    [ 1.46, -0.61, -0.81],
    [-0.54,  1.39, -0.81],
    [-0.54, -0.61,  1.19]
])

test1 = SPF()

e1 = test1.simulate(300, make_tdoa=True, show_plot=False)

test2 = SPF(receiver_positions=np.array([[0, 0, 0],[1, 0, 0],[0, 0, 1],[1, 0, 1]]))

e2 = test2.simulate(300, make_tdoa=True, show_plot=False)

test3 = SPF(receiver_positions=np.array([[0, 0, 0],[0.25, 0, 0],[0.5, 0, 0],[0.75, 0, 0]]))

e3 = test3.simulate(300, make_tdoa=True, show_plot=False)

test4 = SPF(receiver_positions=tetrahedron)

e4 = test4.simulate(300, make_tdoa=True, show_plot=False)


plt.plot(e1, label="random")
plt.plot(e2, label="xz plane")
plt.plot(e3, label="x-axis")
plt.plot(e4, label="tetrahedron")
plt.legend()
plt.show()


# R1 = gen_receiver_pos_arc(num_receivers=4, angle=10, trans_pos=TX)

# test1 = SPF(receiver_positions=R1)
# test1.simulate(300, make_tdoa=True, show_plot=True)

"""

Currently, these receivers are situated in a cube.

- how will the accuracy really change over here? How do I quantify it?

The Random Geometry

1. Changing j_c_q doesn't change the convergence by too much, 0.004 to 0.006 ish
2. Past 100, the points hover about the same location
3. Converges within 10 rounds then its hovering

On the XZ-Plane, square with corner at the origin

1. It does not come close to the actual point.
2. The error keeps increasing, sharply first, and then its a straight line

Let's just define a rudimentary cost function as the squares.

Just so that this isn't a fluke, let's try another plane

XY plane:

1. It converges to about: Iteration 500 | Estimate: (0.7349, 0.4418, 0.7235)
2. This is not the earlier wrong convergence


Line on X-axis

1. Not predictable - converges to a random point or does not converge at all


Let me form a tetrahedron around the actual point

1. It converges, but not to the right point!
2. Oh bother, its not the right TDOA measurements how will it

3. Okay I need a script to give me the right TDOA measurements
4. Damn I wasted this time a bit. 

"""


# woohoo
# Now I can test accuracy and receiver geometry
# implementation wise, I will have to redo it using numpy but atleast I can start testing right now 
