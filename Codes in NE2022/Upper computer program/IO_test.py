import Interface as MI
import numpy as np
import serial
import time


class strcture:
    pass


OPMethod = strcture()
s = serial.Serial('com6', 115200, timeout=15)


def main(Av_out, Bias_out, Av_in, Bias_in, Delay_s, Delay_m, Delay_w):

    # Device select
    DMid = range(24)
    DMid_  = []
    DMlist = np.zeros((1, 24))
    DMlist[:, DMid] = 1
    DMlist[:, DMid_] = 0
    ind = np.arange(0, 24, 3)
    temp = np.array([1, 2, 4, 8, 16, 32, 64, 128]).reshape(-1, 1)
    S = np.dot(np.vstack([DMlist[:, ind], DMlist[:, ind+1], DMlist[:, ind+2]]), temp)

    # Input config
    cycle = 1
    inv = 1
    p =  list(range(0, 255, inv))+ list(range(255, 0, -inv))
    Input = np.uint8(np.array(cycle*p))
    Input_ex = np.vstack((Input, Input, Input)).reshape((1, -1), order='F')[0, :]

    # Start test
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

    Mask = np.uint8(np.zeros(3))
    MI.LoadMask(Mask, s)

    MI.D_init(list(np.uint(S[:, 0])), s)
    time.sleep(0.1)
    out = MI.DM_tb(MI.DMtest(OPMethod, Input_ex, MaskLength=1, Delay_s=Delay_s, Delay_m=Delay_m, Delay_w=Delay_w), s)
    out = np.array(list(out))
    Vout = (3.3*out/255).reshape((24, -1), order='F')

    return Input.tolist(), Vout.tolist()

if __name__ == '__main__':
    main(Av_out=10, Bias_out=160, Av_in=255, Bias_in=200, Delay_s=1, Delay_m=0, Delay_w=1)