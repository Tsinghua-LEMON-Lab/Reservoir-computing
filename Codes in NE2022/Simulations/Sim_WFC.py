import numpy as np
import matplotlib.pyplot as plt
import DMnode as DM

##################################################
# GLOBAL VARIABLES
##################################################
# Test control
SEED = 20
STEP = 400
WUP = 50

# Hyperparameters optimized
ML = 5
N = 24
T = 0.25
S = 2
alpha = 0.2


##################################################
# NETWORK MODEL
##################################################
# Mask setup
Mask = 2*np.random.randint(2, size=(ML, N)) - 1
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
# DATASET
##################################################
def DataGen(step):
    sample = 8
    np.random.seed(SEED)
    p1 = (np.random.rand(sample)-0.5)*2
    p2 = (np.random.rand(sample)-0.5)*2
    np.random.seed()
    Input = np.zeros((1, step))
    Target = np.zeros((1, step))
    for i in range(int(step/sample)):
        q = np.random.randint(2)
        if q == 1:
            Input[0, sample*i:sample*(i+1)] = p1
            Target[0, sample*i:sample*(i+1)] = 1
        else:
            Input[0, sample*i:sample*(i+1)] = p2
            Target[0, sample*i:sample*(i+1)] = 0
    Input = np.vstack([Input, Input, Input])
    return Input, Target


##################################################
# SYSTEM RUN
##################################################
def Train(Input, Target):
    States = DMClassifier(Input)
    States = np.vstack([np.ones((1, STEP)), States])
    Wout = Target[:, WUP:].dot(States[:, WUP:].T).dot(np.linalg.pinv(np.dot(States[:, WUP:], States[:, WUP:].T)))
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output[:, WUP:]-Target[:, WUP:])**2, axis=1)/np.var(Target[:, WUP:], axis=1)))
    print('Train_error: ' + str(NRMSE))
    return Wout, NRMSE


def Test(Wout, Input, Target):
    States = DMClassifier(Input)
    States = np.vstack([np.ones((1, STEP)), States])
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output[:, WUP:]-Target[:, WUP:])**2, axis=1)/np.var(Target[:, WUP:], axis=1)))
    print('Test_error: ' + str(NRMSE))
    return Output, States


##################################################
# MAIN
##################################################
if __name__ == '__main__':
    # TRAINING PROCESURE
    Input, Target_train = DataGen(STEP)
    Wout, _ = Train(Input, Target_train)

    # TESTING PROCESURE
    Input, Target_test = DataGen(STEP)
    Output, States = Test(Wout, Input, Target_test)

    # PLOT
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(Input.T)
    plt.ylabel('Input')
    plt.xlim(0, STEP)
    plt.subplot(2, 1, 2)
    plt.plot(Target_test.T)
    plt.plot(Output.T)
    plt.axis([0, STEP, -0.2, 1.2])
    plt.xlabel('Time Step')
    plt.ylabel('Output')

    plt.figure()
    plt.plot(States.T)
    plt.xlim(0, 200)

    plt.show()
