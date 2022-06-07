import numpy as np
import matplotlib.pyplot as plt
import DMnode as DM

##################################################
# GLOBAL VARIABLES
##################################################

# Hyperparameters optimized
ML = 1
N = 1
T = 0.0
S = 3
alpha = 0.02

##################################################
# NETWORK MODEL
##################################################
Mask = np.ones((ML, N))
u = DM.DM_node(T, S, alpha)


def DMClassifier(Input):
    L = len(Input)
    Input_ex = np.zeros((N, L * ML))
    for j in range(N):
        Input_ex[j, :] = np.dot(Input.reshape((-1, 1)),
                                Mask[:, [j]].T).reshape((1, -1))

    memout = np.zeros((N, L * ML))
    Vm = 0
    for i in range(L * ML):
        memout[:, i], Vm = u.test(Input_ex[:, i], Vm)

    neuout = np.zeros((N * ML, L))
    for i in range(L):
        neuout[:, i] = memout[:, i * ML:(i + 1) * ML].reshape((-1, 1))[:, 0]

    return neuout


##################################################
# DATA PROCESSING
##################################################
def DataGen():
    cycle = 1
    p = list(range(0, 255, 1)) + list(range(255, 0, -1))
    Input = (np.array(cycle * p) / 255 - 0.5) * 2
    return Input


##################################################
# MAIN
##################################################
if __name__ == '__main__':

    # LOAD DATA
    Input = DataGen()

    # I-V
    Output = DMClassifier(Input)
    Output = Output[0, :]

    # PLOT
    plt.figure()
    plt.plot(Input, Output, c='b')
    plt.xticks(np.arange(-1, 1.2, 0.5))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.xlim(-1.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()
