import pandas as pd
import Interface as MI
import numpy as np
import serial
import time
from scipy import io
from datetime import datetime


##################################################
# GLOBAL VARIABLES
##################################################
Vr = 0.2


##################################################
# OBJECTS
##################################################
class strcture:
    pass


OPMethod = strcture()
s = serial.Serial('com6', 115200, timeout=3)


##################################################
# MAPPING
##################################################
def WeightMapping(w, addr):
    out = np.zeros((len(w), 3))
    for i in range(len(w)):
        wl_addr = int(addr[i, 0])
        bl_addr = int(addr[i, 1])
        _, state = MI.Mapping(OPMethod, w[i], Vr, wl_addr, bl_addr, s)
        out[i, :] = np.array([wl_addr, bl_addr, state], dtype=np.float32)
    return out


##################################################
# READ
##################################################
def WeightRead(addr):
    out = np.zeros((len(addr[:, 0]), 3))
    for i in range(len(addr[:, 0])):
        wl_addr = int(addr[i, 0])
        bl_addr = int(addr[i, 1])
        weight = np.array(list(MI.RRAM_tb(MI.Read(OPMethod, 5, Vr, wl_addr, bl_addr), s)))
        out[i, :] = np.array([wl_addr, bl_addr, weight], dtype=np.float32)
        print('wl_addr:' + str(wl_addr) + ',bl_addr:' + str(bl_addr))
    return out


##################################################
# MAIN
##################################################
def main(TASKNAME, CS, isMapping, ForceZero, Save):

    # GAIN ADJUST
    MI.A_init(2, s)
    time.sleep(0.1)
    MI.Am_adj(50, s)
    time.sleep(0.1)
    MI.A_init(3, s)
    time.sleep(0.1)
    MI.Am_adj(145, s)
    time.sleep(0.1)

    BF = 3
    ER = 10
    MX = 250
    if isMapping==1:
        # WEIGHT MAPPING
        if ForceZero==1:
            Av_out = 200
            MI.A_init(2, s)
            time.sleep(0.1)
            MI.Am_adj(Av_out, s)
            time.sleep(0.1)
        MI.R_init(CS=CS, Rmode=0, s=s)
        time.sleep(0.1)

        usenewaddr = False
        bplen = 1
        ind = []
        newaddr = []
        while usenewaddr|(bplen!=0): 
            Wout = io.loadmat(TASKNAME+'_NATpara.mat')['Wout'][int(CS[0]), :]
            A = io.loadmat(TASKNAME+'_NATpara.mat')['Av'][int(CS[0]), :]
            W = np.round(np.clip(MX*(Wout/np.abs(A)), -255, 255))
            if CS[1] == 'A':
                W[W < 0] = 0
                sign = 1
            elif CS[1] == 'B':             
                W[W > 0] = 0
                sign = -1
            addr = np.zeros((len(W), 2))
            for i in range(len(W)):
                for j in range(BF):
                    addr[i, 0] = int(np.floor((BF*i+j)/8))
                    addr[i, 1] = (BF*i+j) % 8
                    break
            if ForceZero==1:
                TP = np.arange(len(W))[W == 0]
                addr = addr[TP, :]
                W = W[TP]
            if usenewaddr&(bplen==0):
                usenewaddr = False
            if usenewaddr:
                W = W[ind]
                addr = newaddr
                state = WeightMapping(np.abs(W), addr)
            out = WeightRead(addr)
            W_ = sign*out[:, 2]
            newaddr = out[np.abs(W-W_)>ER, :2]
            bplen = len(newaddr[:, 0])
            print(bplen)

            ind = np.arange(len(W))[np.abs(W-W_)>ER]
            if bool(1-usenewaddr)&(bplen!=0):
                usenewaddr = True

    # READ
    MI.R_init(CS=CS, Rmode=0, s=s)
    time.sleep(0.1)

    Wout = io.loadmat(TASKNAME+'_NATpara.mat')['Wout'][int(CS[0]), :]
    A = io.loadmat(TASKNAME+'_NATpara.mat')['Av'][int(CS[0]), :]
    W = np.round(MX*(Wout/np.abs(A)))
    if CS[1] == 'A':
        W[W < 0] = 0
        sign = 1
    elif CS[1] == 'B':             
        W[W > 0] = 0
        sign = -1
    else:
        sign = 1
    addr = np.zeros((len(W), 2))
    for i in range(len(W)):
        for j in range(BF):
            addr[i, 0] = int(np.floor((BF*i+j)/8))
            addr[i, 1] = (BF*i+j) % 8
            break
    if ForceZero==1:
        TP = np.arange(len(W))[W == 0]
        addr = addr[TP, :]
        W = W[TP]
    out = WeightRead(addr)
    W_ = sign*out[:, 2]

    # SAVE
    if ForceZero!=1:
        Addr = out[:, :2]
        name = ['wl_addr', 'bl_addr']
        data = pd.DataFrame(columns=name, data=list(Addr))
        data.to_csv('Addr.csv', index=False)
    if Save==1:
        curr_time = datetime.now().strftime("%Y%m%d_%H%M")
        if CS[1]=='A':
            wstr = '_PosW'+CS[0]+'_'
        else:
            wstr = '_NegW'+CS[0]+'_'
        Filename = 'data/'+TASKNAME+'data/'+TASKNAME+wstr+curr_time+'.mat'
        io.savemat(Filename, {'W': W, 'W_': W_})
    return W.tolist(), W_.tolist()

if __name__ == '__main__':
    main(TASKNAME='HAR', CS='0A', isMapping=1, ForceZero=0, Save=0)
