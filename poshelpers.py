import numpy as np


# actually distance for now, not time
def gen_tdoa_measurements(transmitter=np.array([0.46, 0.39, 0.19]), receivers=np.array(np.array([[0.86, 0.79, 0.40],[0.24, 0.10, 0.25],[0.50, 0.11, 0.52],[0.89, 0.38, 0.07]])), sos=1):
    N = receivers.shape[0]
    tdoa = np.zeros((N, N))

    for i in range(N):
        for j in range(i+1):
            d1, d2 =  np.linalg.norm(transmitter - receivers[i]), np.linalg.norm(transmitter - receivers[j])            
            tdoa[i][j] = d1 - d2
            tdoa[j][i] = -tdoa[i][j]

    return tdoa


def gen_receiver_pos_arc(trans_pos=np.array([0,0,0]), angle=90, num_receivers=4, radius=1):

    # offset the final position by the position of the transmitter
    # On XZ plane. If one, then on the x-axis.  (x, 0, z)
    # x = rcos theta, z = rsin theta

    step = angle / (num_receivers - 1)

    recs = np.zeros(shape=(num_receivers, 3))

    for i in range(0, num_receivers):
        theta = i * step
        recs[i][0] = radius * np.cos((theta * np.pi / 180))
        recs[i][2] = radius * np.sin((theta * np.pi / 180))



    return recs + np.array([1, 0.5, 1])


"""

Okay, now I can create test cases and compare.

"""


