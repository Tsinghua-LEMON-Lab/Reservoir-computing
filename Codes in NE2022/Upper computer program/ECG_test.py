import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import serial
import time
import Interface as MI
from datetime import datetime
from sklearn.model_selection import train_test_split


##################################################
# GLOBAL VARIABLES
##################################################

# Task
STEP = 10000
RANDOM_SEED = 18

# Hardware setting
CS = 0
ML = 5
N = 8


##################################################
# OBJECTS
##################################################
class strcture:
    pass


OPMethod = strcture()
s = serial.Serial('com6', 115200, timeout=15)


##################################################
# HARDWARE RC SETUP
##################################################

# DMReservoir setup
def DMClassifier(Input, Delay_s, Delay_m, Delay_w, TestOnly):
    Input = Input.reshape((1, -1))
    Input = np.uint8(255*(Input[0, :]+0.5))
    memout = MI.DMRC_tb(MI.DMRCtest(OPMethod, Input, ML, Delay_s, Delay_m, Delay_w, TestOnly), s)
    memout = np.array(list(memout))
    if TestOnly==1:
        states = []
        out = memout.reshape((4, -1), order='F')[CS, :]
    else:
        states = memout[:8*len(Input)*ML].reshape((3*N*ML, -1), order='F')
        out = memout[8*len(Input)*ML:].reshape((4, -1), order='F')[CS, :]
    return states, out


##################################################
# DATA PROCESSING
##################################################
def DataProcess(x, y):
    Input = np.zeros((len(x[:, 0]), len(x[0, :]), 3))
    Target = y.reshape((1, -1, 1))[0, :, :].T
    Input[:, :, 0] = x
    Input[:, :, 1] = x
    Input[:, :, 2] = x
    return Input, Target


def DataGen():
    # LOAD DATA
    data = io.loadmat('ECGdataset.mat')['dataset'][:STEP, :, :]

    # DATA PREPROCESSING
    inputs = 0.5*data[:, :, 0]/np.max(np.abs(data[:, :, 0]), axis=1).reshape((-1, 1))
    labels = data[:, :, 1]

    print("Data shape: ", inputs.shape)
    print("Labels shape:", labels.shape)

    # SPLIT INTO TRAINING AND TEST SETS
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, test_size=0.9, random_state=RANDOM_SEED)
    print("X train size: ", len(X_train))
    print("X test size: ", len(X_test))
    print("Y train size: ", len(Y_train))
    print("Y test size: ", len(Y_test))
    return X_train, X_test, Y_train, Y_test


##################################################
# SYSTEM RUN
##################################################
def Train(Input, Target, Delay_s, Delay_m, Delay_w, TestOnly):
    L = len(Input[0, :, 0])
    Num = len(Input[:, 0, 0])
    States = np.ones((3*N*ML, L*Num))
    for i in range(Num):
        States[:, L*i:L*(i+1)], _ = DMClassifier(Input[i, :, :], Delay_s, Delay_m, Delay_w, TestOnly)
        print('Train_num: ' + str(i))
    Wout = Target.dot(States.T).dot(np.linalg.pinv(np.dot(States, States.T)))
    Output = np.dot(Wout, States)
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Train_error: ' + str(NRMSE))
    return States, NRMSE


def Test(Input, Target, Delay_s, Delay_m, Delay_w, TestOnly):
    L = len(Input[0, :, 0])
    Num = len(Input[:, 0, 0])
    Output = np.zeros((1, L*Num))
    States = np.ones((3*N*ML, L*Num))
    for i in range(Num):
        if TestOnly==1:
            _, Output[:, L*i:L*(i+1)] = DMClassifier(Input[i, :, :], Delay_s, Delay_m, Delay_w, TestOnly)
        else:
            States[:, L*i:L*(i+1)], Output[:, L*i:L*(i+1)] = DMClassifier(Input[i, :, :], Delay_s, Delay_m, Delay_w, TestOnly)
        print('Test_num: ' + str(i))
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Test_error: ' + str(NRMSE))
    return Output, States, NRMSE

def main(Av_out, Bias_out, Av_in, Bias_in, Delay_s, Delay_m, Delay_w, TestOnly, SAVE, FIX):

    # Mask setup
    if FIX == 1:
        Mask = io.loadmat('ECGpara.mat')['Mask'][0, :]
    else:
        Mask = np.uint8(np.random.randint(0, 255, 3*ML))

    # Addr setup
    Addr = np.uint8(np.zeros(int(4.5*ML*N)))
    if TestOnly==1:
        AddrData = pd.read_csv('Addr.csv')
        Addr0 = np.array([AddrData['wl_addr'].values, AddrData['bl_addr'].values]).T
        Addr[:3*ML*N] = np.uint8(Addr0[:, 0])
        for i in range(3*ML*N):
            if i % 2 == 1:
                Addr[int(3*ML*N+(i-1)/2)] = np.uint8(Addr0[i, 1]*8) | np.uint8(Addr0[i-1, 1])

    # LOAD DATA
    X_train, X_test, Y_train, Y_test = DataGen()

    # GAIN ADJUST
    MI.A_init(0, s)
    time.sleep(0.1)
    MI.Am_adj(Av_in, s)
    time.sleep(0.1)
    MI.A_init(1, s)
    time.sleep(0.1)
    MI.Am_adj(Bias_in, s)
    time.sleep(0.1)
    MI.A_init(2, s)
    time.sleep(0.1)
    MI.Am_adj(Av_out, s)
    time.sleep(0.1)
    MI.A_init(3, s)
    time.sleep(0.1)
    MI.Am_adj(Bias_out, s)
    time.sleep(0.1)

    # MASK LOAD
    MI.LoadMask(Mask, s)
    time.sleep(0.1)

    # ADDR LOAD
    MI.LoadAddr(Addr, s)
    time.sleep(0.1)

    # Device select
    DMid = range(24)
    DMid_ = []
    DMlist = np.zeros((1, 24))
    DMlist[:, DMid] = 1
    DMlist[:, DMid_] = 0
    ind = np.arange(0, 24, 3)
    temp = np.array([1, 2, 4, 8, 16, 32, 64, 128]).reshape(-1, 1)
    S = np.dot(np.vstack([DMlist[:, ind], DMlist[:, ind+1], DMlist[:, ind+2]]), temp)

    # TRAINING PROCESURE
    MI.F_init(list(np.uint(S[:, 0])), s)
    time.sleep(0.1)
    NRMSE_train = 0
    if TestOnly != 1:
        Input, Target_train = DataProcess(X_train, Y_train)
        States_train, NRMSE_train = Train(Input, Target_train, Delay_s, Delay_m, Delay_w, TestOnly)

    # TESTING PROCESURE
    if TestOnly != 1:
        X_test = X_test[:200, :]
        Y_test = Y_test[:200, :]
    Input, Target_test = DataProcess(X_test, Y_test)
    Output, States_test, NRMSE_test = Test(Input, Target_test, Delay_s, Delay_m, Delay_w, TestOnly)
    OMAX = np.max(Output, axis=1).reshape((-1, 1))
    Output = Output/OMAX

    # SAVE
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    if SAVE == 1:
        if TestOnly == 1:
            Filename = 'data/ECGdata/ECG_test_'+curr_time+'.mat'
            io.savemat(Filename, {'Input_test': X_test, 'Target_test': Target_test, 'Output_test': Output})
        else:
            Filename = 'data/ECGdata/ECG_train_'+curr_time+'.mat'
            io.savemat('ECGpara.mat', {'Mask': Mask, 'ML': ML, 'N': N, 'States_train': States_train,
                   'Target_train': Target_train, 'States_test': States_test, 'Target_test': Target_test})
            io.savemat(Filename, {'States_train': States_train, 'Target_train': Target_train, 'Input_train': X_train,
                   'States_test': States_test, 'Target_test': Target_test, 'Input_test': X_test})

    # ACC CACULATING
    ACC = np.zeros((60, 5))
    TH_list = np.zeros(2)
    if TestOnly == 1:
        TH_box = np.arange(0.21, 0.8, 0.01)
        THS_box = np.arange(1, 6)
        j = 0
        for TH in TH_box:
            k = 0
            for THS in THS_box:
                Fout = np.heaviside(Output[0, :].reshape(-1, 50)-TH, 1)
                Fout = np.heaviside(np.sum(Fout, axis=1)-THS, 1)
                Ftar = np.max(Target_test[0, :].reshape(-1, 50), axis=1)
                Fbox = Fout-Ftar
                ACC[j, k] = len(Fbox[Fbox==0])/len(Fbox)
                k = k+1
            j = j+1
        index = np.unravel_index(ACC.argmax(), ACC.shape)
        index = list(index)
        TH_list[0] = TH_box[index[0]]
        TH_list[1] = THS_box[index[1]]
        io.savemat('ECG_THlist.mat', {'TH_list': TH_list})
        print(np.max(ACC))

    Input_list = list(X_test.reshape((1, -1, 1))[0, :, 0])
    Target_list = list(Target_test[0, :])
    Output_list = list(Output[0, :])
    return Input_list, Target_list, Output_list, TH_list.tolist(), NRMSE_train, NRMSE_test, np.max(ACC)


##################################################
# MAIN
##################################################
if __name__ == '__main__':
    main(Av_out=50, Bias_out=160, Av_in=255, Bias_in=230, Delay_s=10, Delay_m=2, Delay_w=1, TestOnly=1, SAVE=0, FIX=0)
