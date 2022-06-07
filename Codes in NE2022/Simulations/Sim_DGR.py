import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import DMnode as DM


##################################################
# GLOBAL VARIABLES
##################################################

# Task
N_CLASSES = 4
SEED = 18

# Hyperparameters optimized
ML = 8
N = 8
T = -0.05
S = 0.9
alpha = 0.03


##################################################
# NETWORK MODEL
##################################################
Mask = io.loadmat('DGRpara.mat')['Mask']
u = DM.DM_node(T, S, alpha)


def DMClassifier(Input):
    L = len(Input[0, :])
    Input_ex = np.zeros((3*N, L*ML))
    for j in range(N):
        ax = np.dot(Input[[0], :].T, Mask[:, [j]].T).reshape((1, -1))
        ay = np.dot(Input[[1], :].T, Mask[:, [j]].T).reshape((1, -1))
        az = np.dot(Input[[2], :].T, Mask[:, [j]].T).reshape((1, -1))
        Input_ex[j*3:(j+1)*3, :] = np.vstack([ax, ay, az])

    memout = np.zeros((3*N, L*ML))
    Vm = 0
    for i in range(L*ML):
        memout[:, i], Vm = u.test(Input_ex[:, i], Vm)

    neuout = np.zeros((3*N*ML, L))
    for i in range(L):
        neuout[:, i] = memout[:, i*ML:(i+1)*ML].reshape((-1, 1))[:, 0]

    return neuout


##################################################
# DATA PROCESSING
##################################################
def DataProcess(x, y):
    Input = x.reshape((1, -1, 3))[0, :, :].T
    Target = y.reshape((1, -1, N_CLASSES))[0, :, :].T
    return Input, Target


def DataGen():
    # LOAD DATA
    data = io.loadmat('DGRdataset.mat')['dataset']
    np.random.seed(SEED)
    np.random.shuffle(data)
    np.random.seed()
    data_test = data[:300, :, :]
    data_train = data[300:, :, :]

    # DATA PREPROCESSING
    X_train = data_train[:, :, :3]/np.max(np.abs(data_train[:, :, :3]))
    X_test = data_test[:, :, :3]/np.max(np.abs(data_test[:, :, :3]))
    Y_train = data_train[:, :, 3]
    Y_test = data_test[:, :, 3]
    print("X train size: ", len(X_train))
    print("X test size: ", len(X_test))
    print("Y train size: ", len(Y_train))
    print("Y test size: ", len(Y_test))
    return X_train, X_test, Y_train, Y_test


##################################################
# SYSTEM RUN
##################################################
def Train(Input, Target):
    Input = Input.reshape((1, -1, 3))[0, :, :].T
    States = DMClassifier(Input)
    States = np.vstack([np.ones((1, len(Input[0, :]))), States])
    Wout = Target.dot(States.T).dot(np.linalg.pinv(np.dot(States, States.T)))
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Train_error: ' + str(NRMSE))
    return Wout, NRMSE


def Test(Wout, Input, Target):
    Input = Input.reshape((1, -1, 3))[0, :, :].T
    States = DMClassifier(Input)
    States = np.vstack([np.ones((1, len(Input[0, :]))), States])
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Test_error: ' + str(NRMSE))
    return Output


##################################################
# MAIN
##################################################
if __name__ == '__main__':

    # LOAD DATA
    X_train, X_test, Y_train, Y_test = DataGen()

    # TRAINING PROCESURE
    Target_train = []
    for i in range(1, N_CLASSES+1):
        Target = Y_train/i
        Target[Target[:, 15] != 1, :] = 0
        Target = Target.reshape((1, -1))
        Target_train.append(Target[0, :])
    Target_train = np.array(Target_train)
    Wout, _ = Train(X_train, Target_train)

    # TESTING PROCESURE
    Target_test = []
    for i in range(1, N_CLASSES+1):
        Target = Y_test/i
        Target[Target[:, 15] != 1, :] = 0
        Target = Target.reshape((1, -1))
        Target_test.append(Target[0, :])
    Target_test = np.array(Target_test)
    Output = Test(Wout, X_test, Target_test)

    # ACC CACULATING
    LEN = 30
    ACC = np.zeros((60, 9, N_CLASSES))
    j = 0
    for TH in np.arange(0.21, 0.8, 0.01):
        k = 0
        for THS in np.arange(1, 10):
            for i in range(N_CLASSES):
                Fout = np.heaviside(Output[i, :].reshape(-1, LEN)-TH, 1)
                Fout = np.heaviside(np.sum(Fout, axis=1)-THS, 1)
                Ftar = np.max(Target_test[i, :].reshape(-1, LEN), axis=1)
                Fbox = Fout-Ftar
                ACC[j, k, i] = len(Fbox[Fbox == 0])/len(Fbox)
            k = k+1
        j = j+1
    for i in range(N_CLASSES):
        print(np.max(ACC[:, :, i]))

    # PLOT
    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(X_test.reshape((-1, 3)))
    plt.axis([0, 2000, -1.1, 1.1])
    plt.ylabel('Input')
    plt.subplot(5, 1, 2)
    plt.plot(Target_test[0, :])
    plt.plot(Output[0, :])
    plt.axis([0, 2000, -0.2, 1.2])
    plt.ylabel('O1')
    plt.subplot(5, 1, 3)
    plt.plot(Target_test[1, :])
    plt.plot(Output[1, :])
    plt.axis([0, 2000, -0.2, 1.2])
    plt.ylabel('O2')
    plt.subplot(5, 1, 4)
    plt.plot(Target_test[2, :])
    plt.plot(Output[2, :])
    plt.axis([0, 2000, -0.2, 1.2])
    plt.ylabel('O3')
    plt.subplot(5, 1, 5)
    plt.plot(Target_test[3, :])
    plt.plot(Output[3, :])
    plt.axis([0, 2000, -0.2, 1.2])
    plt.ylabel('O4')
    plt.xlabel('Time Step')
    plt.show()
