import numpy as np
from scipy import io


##################################################
# SYSTEM RUN
##################################################
def Train(States, Target):
    Wout = Target.dot(States.T).dot(np.linalg.pinv(np.dot(States, States.T)))
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Train_error: ' + str(NRMSE))
    return Wout, NRMSE


def Test(Wout, States, Target):
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Test_error: ' + str(NRMSE))
    return Output, States, NRMSE


def main(TASKNAME, GOAL, SAVE):
    # TRAINING PROCESURE
    States = io.loadmat(TASKNAME+'para.mat')['States_train']
    States = States.reshape((-1, len(States[0, 0, :]))).T
    States = States + 25*(np.random.rand(len(States[:, 0]), len(States[0, :]))-0.5)*2
    Target = io.loadmat(TASKNAME+'para.mat')['Target_train']  
    if TASKNAME=='DGR':
        Target_train = []
        for i in range(1, 5):
            Target_ = Target/i
            Target_[Target_[:, 15]!=1, :] = 0
            Target_ = Target_.reshape((1, -1))
            Target_train.append(Target_[0, :])
        Target = np.array(Target_train)

    Wout, NRMSE_train = Train(States, Target)
    Av = np.max(np.abs(Wout), axis=1).reshape((-1, 1))
    Wout_ = Av*np.clip(Wout/Av + 0.04*(2*np.random.rand(len(Wout[:, 0]), len(Wout[0, :]))-1), -1, 1)

    # TESTING PROCESURE
    States = io.loadmat(TASKNAME+'para.mat')['States_test']
    Target = io.loadmat(TASKNAME+'para.mat')['Target_test']
    Output, States, NRMSE_test = Test(Wout_, States, Target)   

    # SAVE
    if SAVE == 1:
        io.savemat(TASKNAME+'_NATpara.mat', {'Wout': Wout, 'Av': Av, 'States': States, 'Target': Target})

    Target_list = list(Target[GOAL-1, :])
    Output_list = list(Output[GOAL-1, :])
    return Target_list, Output_list, NRMSE_train, NRMSE_test


##################################################
# MAIN
##################################################
if __name__ == '__main__':
    main(TASKNAME='DGR', GOAL=3, SAVE=0)
