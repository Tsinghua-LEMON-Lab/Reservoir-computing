import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import DMnode as DM
from sklearn.model_selection import train_test_split


##################################################
# GLOBAL VARIABLES
##################################################

# Task
N_CLASSES = 1
STEP = 1000

# Hyperparameters optimized
ML = 5
N = 24
T = 0.08
S = 2
alpha = 0.035

##################################################
# NETWORK MODEL
##################################################
Mask = io.loadmat('ECGpara.mat')['Mask']
u = DM.DM_node(T, S, alpha)


def DMClassifier(Input):
    L = len(Input[0, :])
    Input_ex = np.zeros((N, L*ML))
    for j in range(N):
        Input_ex[j, :] = np.dot(Input[[0], :].T, Mask[:, [j]].T).reshape((1, -1))

    memout = np.zeros((N, L*ML))
    Vm = 0
    for i in range(L*ML):
        memout[:, i], Vm = u.test(Input_ex[:, i], Vm)

    neuout = np.zeros((N*ML, L))
    for i in range(L):
        neuout[:, i] = memout[:, i*ML:(i+1)*ML].reshape((-1, 1))[:, 0]

    return neuout


##################################################
# DATA PROCESSING
##################################################
def DataProcess(x, y):
    Input = x.reshape((1, -1, 1))[0, :, :].T
    Target = y.reshape((1, -1, N_CLASSES))[0, :, :].T
    return Input, Target


def DataGen():
    # LOAD DATA
    data = io.loadmat('ECGdataset.mat')['dataset'][:STEP, :, :]

    # DATA PREPROCESSING
    inputs = data[:, :, 0]/np.max(np.abs(data[:, :, 0]), axis=1).reshape((-1, 1))
    labels = data[:, :, 1:]

    print("Data shape: ", inputs.shape)
    print("Labels shape:", labels.shape)

    # SPLIT INTO TRAINING AND TEST SETS
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, test_size=0.3)
    print("X train size: ", len(X_train))
    print("X test size: ", len(X_test))
    print("Y train size: ", len(Y_train))
    print("Y test size: ", len(Y_test))
    return X_train, X_test, Y_train, Y_test


##################################################
# SYSTEM RUN
##################################################
def Train(Input, Target):
    States = DMClassifier(Input)
    States = np.vstack([np.ones((1, len(Input[0, :]))), States])
    Wout = Target.dot(States.T).dot(np.linalg.pinv(np.dot(States, States.T)))
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Train_error: ' + str(NRMSE))
    return Wout, NRMSE


def Test(Wout, Input, Target):
    States = DMClassifier(Input)
    States = np.vstack([np.ones((1, len(Input[0, :]))), States])
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Test_error: ' + str(NRMSE))
    return Output, States, NRMSE


##################################################
# MAIN
##################################################
if __name__ == '__main__':

    # LOAD DATA
    X_train, X_test, Y_train, Y_test = DataGen()

    # TRAINING PROCESURE
    Input, Target = DataProcess(X_train, Y_train)
    Wout, _ = Train(Input, Target)

    # TESTING PROCESURE
    Input, Target = DataProcess(X_test, Y_test)
    Output, States, _ = Test(Wout, Input, Target)

    # ACC CACULATING
    LEN = 50
    ACC = np.zeros((60, 5))
    TH_list = np.zeros(2)
    TH_box = np.arange(0.21, 0.8, 0.01)
    THS_box = np.arange(1, 6)
    j = 0
    for TH in TH_box:
        k = 0
        for THS in THS_box:
            Fout = np.heaviside(Output[0, :].reshape(-1, LEN)-TH, 1)
            Fout = np.heaviside(np.sum(Fout, axis=1)-THS, 1)
            Ftar = np.max(Target[0, :].reshape(-1, LEN), axis=1)
            Fbox = Fout-Ftar
            ACC[j, k] = len(Fbox[Fbox == 0])/len(Fbox)
            k = k+1
        j = j+1
    print(np.max(ACC))

    # PLOT
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(Input.T)
    plt.axis([0, 5000, -1, 1])
    plt.ylabel('Input')
    plt.subplot(2, 1, 2)
    plt.plot(Target[0, :])
    plt.plot(Output[0, :])
    plt.axis([0, 5000, -0.2, 1.2])
    plt.ylabel('Output')
    plt.show()
