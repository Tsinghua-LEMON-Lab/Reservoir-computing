import numpy as np
from scipy import io
import torch
import torch.optim as optim
import torch.nn as nn
import Mylayers as myl


##################################################
# NETWORK MODEL
##################################################
class NNClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNClassifier, self).__init__()
        self.out = myl.RRAMsim(input_size, output_size, bias=False)

    def forward(self, x):
        y = self.out(x)
        return y


##################################################
# SYSTEM RUN
##################################################
def Train_LR(States, Target):
    Wout = Target.dot(States.T).dot(np.linalg.pinv(np.dot(States, States.T)))
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Train_error: ' + str(NRMSE))
    return Wout, NRMSE


def Train_GD(States, Target, N_EPOCHS, LEARNING_RATE):
    Input = torch.from_numpy(np.float32(States.T))
    model = NNClassifier(input_size=len(Input[0, :]), output_size=len(Target[:, 0]))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.MSELoss()

    train_error_ = []
    for epoch in range(N_EPOCHS):
        Output = model(Input)
        loss = loss_func(Output, torch.from_numpy(np.float32(Target.T)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Out = Output.data.numpy().T
        NRMSE = np.mean(np.sqrt(np.mean((Out-Target)**2, axis=1)/np.var(Target, axis=1)))
        train_error_.append(NRMSE)

        print('[Epoch: %3d/%3d] Training Error: %.3f'
              % (epoch+1, N_EPOCHS, train_error_[epoch]))

    Wout = (model.out.Av*model.out.weight_).detach().numpy().T
    Av = np.array([model.out.Av.detach().numpy()]).T
    return Wout, Av


def Test(Wout, States, Target):
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Test_error: ' + str(NRMSE))
    return Output, States, NRMSE


def main(TASKNAME, GOAL, SAVE, TrainMethod):
    # TRAINING PROCESURE
    States = io.loadmat(TASKNAME+'para.mat')['States_train']
    States = States.reshape((-1, len(States[0, 0, :]))).T
    States = States + 25*(np.random.rand(len(States[:, 0]), len(States[0, :]))-0.5)*2
    Target = io.loadmat(TASKNAME+'para.mat')['Target_train']
    if TASKNAME == 'DGR':
        Target_list = []
        for i in range(1, 5):
            Target_ = Target/i
            Target_[Target_[:, 15] != 1, :] = 0
            Target_ = Target_.reshape((1, -1))
            Target_list.append(Target_[0, :])
        Target = np.array(Target_list)
    if TrainMethod == 'LR':
        Wout, NRMSE_train = Train_LR(States, Target)
        Av = np.max(np.abs(Wout), axis=1).reshape((-1, 1))
    elif TrainMethod == 'GD':
        Wout, Av = Train_GD(States, Target, 80000, 0.001)

    # TESTING PROCESURE
    States = io.loadmat(TASKNAME+'para.mat')['States_test']
    Target = io.loadmat(TASKNAME+'para.mat')['Target_test']
    Wout_ = Av*np.clip(Wout/Av + 0.04*(2*np.random.rand(len(Wout[:, 0]), len(Wout[0, :]))-1), -1, 1)
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
    main(TASKNAME='DGR', GOAL=3, SAVE=0, TrainMethod='LR')
