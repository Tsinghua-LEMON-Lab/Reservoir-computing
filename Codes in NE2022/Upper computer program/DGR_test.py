import numpy as np
import pandas as pd
from scipy import io
import serial
import time
import Interface as MI
from datetime import datetime


##################################################
# GLOBAL VARIABLES
##################################################

# Dataset related
N_CLASSES = 4
STEP = 200
SEED = 18

# Hyperparameters optimized
ML = 8
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
        out = memout.reshape((4, -1), order='F')
    else:
        states = memout[:8*len(Input)*ML].reshape((3*N*ML, -1), order='F')
        out = memout[8*len(Input)*ML:].reshape((4, -1), order='F')
    return states, out


##################################################
# DATA LOADING
##################################################
def DataGen():
    # LOAD DATA
    data = io.loadmat('DGRdataset.mat')['dataset']
    np.random.seed(SEED)
    np.random.shuffle(data)
    np.random.seed()
    data_test = data[:300, :, :]
    data_train = data[300:, :, :]

    # DATA PREPROCESSING
    X_train = 0.5*data_train[:, :, :3]/np.max(np.abs(data_train[:, :, :3]))
    X_test = 0.5*data_test[:, :, :3]/np.max(np.abs(data_test[:, :, :3]))
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
def Train(X_train, Target, Delay_s, Delay_m, Delay_w, TestOnly):
    L = len(X_train[0, :, 0])
    Num = len(X_train[:, 0, 0])
    States = np.ones((Num, L, 3*N*ML))
    for i in range(Num):
        a, _ = DMClassifier(X_train[i, :, :], Delay_s, Delay_m, Delay_w, TestOnly)
        States[i, :, :] = a.T
        print('Train_num: ' + str(i))
    Sdata = np.zeros((Num, L, 3*N*ML+1))
    Sdata[:, :, :3*N*ML] = States
    Sdata[:, :, 3*N*ML] = Target
    NRMSE = np.ones(N_CLASSES)
    for i in range(N_CLASSES):
        Goaldata = Sdata[Target[:, 15]==i+1, :, :]
        Compdata = Sdata[Target[:, 15]!=i+1, :, :]
        data = np.vstack([Goaldata[:STEP, :, :], Compdata[:STEP, :, :]])
        np.random.shuffle(data)
        States_ = data[:, :, :-1].reshape((-1, 3*N*ML)).T
        Target_ = data[:, :, -1]/(i+1)
        Target_[Target_[:, 15]!=1, :] = 0
        Target_ = Target_.reshape((1, -1))
        Wout = Target_.dot(States_.T).dot(np.linalg.pinv(np.dot(States_, States_.T)))
        Output = np.dot(Wout, States_)
        NRMSE[i] = np.mean(np.sqrt(np.mean((Output-Target_)**2, axis=1)/np.var(Target_, axis=1)))
    print('Train_error: ' + str(np.mean(NRMSE)))
    return States, np.mean(NRMSE)


def Test(X_test, Target, Delay_s, Delay_m, Delay_w, TestOnly):
    L = len(X_test[0, :, 0])
    Num = len(X_test[:, 0, 0])
    Output = np.zeros((N_CLASSES, L*Num))
    States = np.ones((3*N*ML, L*Num))
    for i in range(Num):
        if TestOnly==1:
            _, Output[:, L*i:L*(i+1)] = DMClassifier(X_test[i, :, :], Delay_s, Delay_m, Delay_w, TestOnly)
            # time.sleep(0.4)
        else:
            States[:, L*i:L*(i+1)], Output[:, L*i:L*(i+1)] = DMClassifier(X_test[i, :, :], Delay_s, Delay_m, Delay_w, TestOnly)
        print('Test_num: ' + str(i))
    NRMSE = np.mean(np.sqrt(np.mean((Output-Target)**2, axis=1)/np.var(Target, axis=1)))
    print('Test_error: ' + str(NRMSE))
    return Output, States, NRMSE


def main(Av_out, Bias_out, Av_in, Bias_in, Delay_s, Delay_m, Delay_w, TestOnly, SAVE, FIX):

    # Mask setup
    if FIX == 1:
        Mask = io.loadmat('DGRpara.mat')['Mask'][0, :]
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
        States_train, NRMSE_train = Train(X_train, Y_train, Delay_s, Delay_m, Delay_w, TestOnly)

    # TESTING PROCESURE
    Target_test = []
    for i in range(1, N_CLASSES+1):
        Target = Y_test/i
        Target[Target[:, 15]!=1, :] = 0
        Target = Target.reshape((1, -1))
        Target_test.append(Target[0, :])
    Target_test = np.array(Target_test)
    Output, States_test, NRMSE_test = Test(X_test, Target_test, Delay_s, Delay_m, Delay_w, TestOnly)
    OMAX = np.max(Output[:, :], axis=1).reshape((-1, 1))
    Output = Output/OMAX

    # SAVE
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    if SAVE == 1:
        if TestOnly == 1:
            Filename = 'data/DGRdata/DGR_test_'+curr_time+'.mat'
            io.savemat(Filename, {'Input_test': X_test, 'Target_test': Target_test, 'Output_test': Output})
        else:
            Filename = 'data/DGRdata/DGR_train_'+curr_time+'.mat'
            io.savemat('DGRpara.mat', {'Mask': Mask, 'ML': ML, 'N': N, 'States_train': States_train,
                   'Target_train': Y_train, 'States_test': States_test, 'Target_test': Target_test})
            io.savemat(Filename, {'States_train': States_train, 'Target_train': Y_train, 'Input_train': X_train,
                   'States_test': States_test, 'Target_test': Target_test, 'Input_test': X_test})

    # ACC CACULATING
    ACC = np.zeros((60, 9, 4))
    TH_list = np.zeros((4, 2))
    ACC_list = np.zeros(4)
    if TestOnly == 1:
        TH_box = np.arange(0.21, 0.8, 0.01)
        THS_box = np.arange(1, 10)
        j = 0
        for TH in TH_box:
            k = 0
            for THS in THS_box:
                for i in range(4):
                    Fout = np.heaviside(Output[i, :].reshape(-1, 30)-TH, 1)
                    Fout = np.heaviside(np.sum(Fout, axis=1)-THS, 1)
                    Ftar = np.max(Target_test[i, :].reshape(-1, 30), axis=1)
                    Fbox = Fout-Ftar
                    ACC[j, k, i] = len(Fbox[Fbox==0])/len(Fbox)
                k = k+1
            j = j+1
        for i in range(4):
            index = np.unravel_index(ACC[:, :, i].argmax(), ACC[:, :, i].shape)
            index = list(index)
            TH_list[i, 0] = TH_box[index[0]]
            TH_list[i, 1] = THS_box[index[1]]
        io.savemat('DGR_THlist.mat', {'TH_list': TH_list})
    
        for i in range(4):
            print(np.max(ACC[:, :, i]))
            ACC_list[i] = np.max(ACC[:, :, i])

    Ta_list = list(TH_list[:, 0])
    Tb_list = list(TH_list[:, 1])
    ACC_list = list(ACC_list)
    Input_list1 = list(X_test.reshape((-1, 3))[:, 0])
    Input_list2 = list(X_test.reshape((-1, 3))[:, 1])
    Input_list3 = list(X_test.reshape((-1, 3))[:, 2])
    Target_list1 = list(Target_test[0, :])
    Output_list1 = list(Output[0, :])
    Target_list2 = list(Target_test[1, :])
    Output_list2 = list(Output[1, :])
    Target_list3 = list(Target_test[2, :])
    Output_list3 = list(Output[2, :])
    Target_list4 = list(Target_test[3, :])
    Output_list4 = list(Output[3, :])
    return (Input_list1, Input_list2, Input_list3, Target_list1, Output_list1, Target_list2, Output_list2, Target_list3,
    Output_list3, Target_list4, Output_list4, Ta_list, Tb_list, NRMSE_train, NRMSE_test, ACC_list)


##################################################
# MAIN
##################################################
if __name__ == '__main__':
   main(Av_out=50, Bias_out=160, Av_in=255, Bias_in=230, Delay_s=1, Delay_m=0, Delay_w=1, TestOnly=1, SAVE=0, FIX=1)