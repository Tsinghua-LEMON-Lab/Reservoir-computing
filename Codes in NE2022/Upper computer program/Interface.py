import numpy as np
import time


def Reg_prog(RegData, ch, s):
    Tmode = [ord('I')]
    IN = [RegData]
    ReadFlag = [ch]
    m = 1
    STH = [m >> 8]
    STL = [m & 255]
    Z = [0]
    wstr = STH + STL + ReadFlag + Z + Z + Z + Z + Z + Z + Z + Z + Tmode + Z + IN
    s.write(bytes(wstr))


def D_init(DS, s):
    Reg_prog(2, 4, s)
    time.sleep(0.1)
    Reg_prog(DS[0], 0, s)
    time.sleep(0.1)
    Reg_prog(DS[1], 1, s)
    time.sleep(0.1)
    Reg_prog(DS[2], 2, s)


def R_init(CS, Rmode, s):
    Reg_prog(16 | Rmode, 4, s)
    time.sleep(0.1)
    if CS[1] == 'A':
        Reg_prog(1 << int(CS[0])*2, 3, s)
    elif CS[1] == 'B':
        Reg_prog(2 << int(CS[0])*2, 3, s)
    elif CS[1] == 'C':
        Reg_prog(3 << int(CS[0])*2, 3, s)
    time.sleep(0.1)



def F_init(DS, s):
    Reg_prog(35, 4, s)
    time.sleep(0.1)
    Reg_prog(255, 3, s)
    time.sleep(0.1)
    Reg_prog(DS[0], 0, s)
    time.sleep(0.1)
    Reg_prog(DS[1], 1, s)
    time.sleep(0.1)
    Reg_prog(DS[2], 2, s)


def A_init(CS, s):
    Reg_prog(CS << 2, 4, s)


def Am_adj(ResValue, s):
    Tmode = [ord('A')]
    IN = [ResValue]
    m = 1
    STH = [m >> 8]
    STL = [m & 255]
    Z = [0]
    wstr = STH + STL + Z + Z + Z + Z + Z + Z + Z + Z + Z + Tmode + Z + IN
    s.write(bytes(wstr))


def DM_tb(OPMethods, s):
    Tmode = [ord('D')]
    ReadFlag = [OPMethods.ReadFlag]
    DP = OPMethods.DP
    DR = OPMethods.DR
    WAIT = OPMethods.WAIT

    m = len(OPMethods.IN)
    STH = [m >> 8]
    STL = [m & 255]

    DPH = [DP >> 8]
    DPL = [DP & 255]

    DRH = [DR >> 8]
    DRL = [DR & 255]

    WAITH = [WAIT >> 8]
    WAITL = [WAIT & 255]

    IN = OPMethods.IN.tolist()
    wstr = STH + STL + ReadFlag + DPH + DPL + DRH + DRL + [0] + [0] + WAITH + WAITL + Tmode + [0] + IN
    s.write(bytes(wstr))
    out = s.read(8*m*ReadFlag[0])
    return out


def DMtest(OPMethods, Input, MaskLength, Delay_s, Delay_m, Delay_w):
    OPMethods.DP = Delay_s
    OPMethods.DR = Delay_m
    OPMethods.WAIT = Delay_w
    OPMethods.IN = Input
    OPMethods.ReadFlag = MaskLength
    return OPMethods


def LoadMask(Mask, s):
    Tmode = [ord('M')]
    m = len(Mask)
    STH = [m >> 8]
    STL = [m & 255]
    Z = [0]
    IN = Mask.tolist()
    wstr = STH + STL + Z + Z + Z + Z + Z + Z + Z + Z + Z + Tmode + Z + IN
    s.write(bytes(wstr))


def RRAM_tb(OPMethods, s):
    Tmode = [ord('R')]
    IsRead = OPMethods.IsRead
    WL_Addr = OPMethods.WL_Addr
    BL_Addr = OPMethods.BL_Addr
    IsForm = OPMethods.IsForm
    WLref = OPMethods.WLref
    BLref = OPMethods.BLref
    SL_Mode = OPMethods.SLmode
    WAIT = OPMethods.SRDelay
    RTW = OPMethods.FormDelay

    ReadFlag = [IsRead*128 | IsForm*64 | BL_Addr]

    STH = [1 >> 8]
    STL = [1 & 255]

    DPH = [WL_Addr]
    DPL = [WLref]

    DRH = [BLref]
    DRL = [SL_Mode & 3]

    RTWH = [RTW >> 8]
    RTWL = [RTW & 255]

    WAITH = [WAIT >> 8]
    WAITL = [WAIT & 255]

    wstr = STH + STL + ReadFlag + DPH + DPL + DRH + DRL + RTWH + RTWL + WAITH + WAITL + Tmode + [0] + [0]
    s.write(bytes(wstr))
    if IsRead:
        out = s.read(1)
    else:
        out = 0
    return out


def Read(OPMethods, Vwl, Vbl, wl_addr, bl_addr):
    OPMethods.IsRead = True
    OPMethods.IsForm = False
    OPMethods.WL_Addr = wl_addr
    OPMethods.BL_Addr = bl_addr
    OPMethods.WLref = int(255*Vwl/5)
    OPMethods.BLref = int(255*Vbl/0.5)
    OPMethods.SLmode = 3
    OPMethods.SRDelay = 50
    OPMethods.FormDelay = 500
    return OPMethods


def Set(OPMethods, Vwl, Vbl, wl_addr, bl_addr):
    OPMethods.IsRead = False
    OPMethods.IsForm = False
    OPMethods.WL_Addr = wl_addr
    OPMethods.BL_Addr = bl_addr
    OPMethods.WLref = int(255*Vwl/5)
    OPMethods.BLref = int(255*Vbl/5)
    OPMethods.SLmode = 0
    OPMethods.SRDelay = 50
    OPMethods.FormDelay = 500
    return OPMethods


def Reset(OPMethods, Vwl, Vsl, wl_addr, bl_addr):
    OPMethods.Rmode = 0
    OPMethods.IsRead = False
    OPMethods.IsForm = False
    OPMethods.WL_Addr = wl_addr
    OPMethods.BL_Addr = bl_addr
    OPMethods.WLref = int(255*Vwl/5)
    OPMethods.BLref = int(255*Vsl/5)
    OPMethods.SLmode = 1
    OPMethods.SRDelay = 50
    OPMethods.FormDelay = 500
    return OPMethods


def Mapping(OPMethods, Weight, Vr, wl_addr, bl_addr, s):
    S = 12
    L = np.array([[Weight-S/2, 0], [Weight+S/2, 0]])
    i = 0
    sm = -1
    rm = -1
    Scnt = 0
    Rcnt = 0
    while i < 100:
        out = RRAM_tb(Read(OPMethods, 5, Vr, wl_addr, bl_addr), s)
        out = np.array(list(out))
        n = np.vstack([np.array([[out[0], 1]]), L])
        n = n[n[:, 0].argsort()]
        ind = np.dot(np.array([[1, 2, 3]]), n[:, [1]])
        if ind == 1:
            if (i-sm) == 1:
                Scnt = Scnt+1
            else:
                Scnt = 1
            Svolt = np.minimum(0.5+(Scnt-1)*0.1, 2.8)
            RRAM_tb(Set(OPMethods, Svolt, 5, wl_addr, bl_addr), s)
            sm = i
        elif ind == 2:
            state = 1
            print('wl_addr:' + str(wl_addr) + ',bl_addr:' + str(bl_addr) + ',Successful')
            break
        elif ind == 3:
            if (i-rm) == 1:
                Rcnt = Rcnt+1
            else:
                Rcnt = 1
            Rvolt = np.minimum(1.5+(Rcnt-1)*0.1, 3)
            RRAM_tb(Reset(OPMethods, 5, Rvolt, wl_addr, bl_addr), s)
            rm = i
        i = i+1

    if ind != 2:
        state = 0
        print('wl_addr:' + str(wl_addr) + ',bl_addr:' + str(bl_addr) + ',Failed')
    return out, state


def LoadAddr(Addr, s):
    Tmode = [ord('W')]
    m = len(Addr)
    STH = [m >> 8]
    STL = [m & 255]
    Z = [0]
    IN = Addr.tolist()
    wstr = STH + STL + Z + Z + Z + Z + Z + Z + Z + Z + Z + Tmode + Z + IN
    s.write(bytes(wstr))


def DMRC_tb(OPMethods, s):
    Tmode = [ord('F')]
    ReadFlag = [OPMethods.ReadFlag]
    DP = OPMethods.DP
    DR = OPMethods.DR
    WAIT = OPMethods.WAIT
    Rmode = [OPMethods.Rmode]

    m = len(OPMethods.IN)
    STH = [m >> 8]
    STL = [m & 255]

    DPH = [DP >> 8]
    DPL = [DP & 255]

    DRH = [DR >> 8]
    DRL = [DR & 255]

    WAITH = [WAIT >> 8]
    WAITL = [WAIT & 255]

    IN = OPMethods.IN.tolist()
    wstr = STH + STL + ReadFlag + DPH + DPL + DRH + DRL + [0] + [0] + WAITH + WAITL + Tmode + Rmode + IN
    s.write(bytes(wstr))
    if Rmode[0]==1:
        out = s.read(int(4*m/3))
    else:
        out = s.read(8*m*ReadFlag[0]+int(4*m/3))
    return out


def DMRCtest(OPMethods, Input, MaskLength, Delay_s, Delay_m, Delay_w, TestOnly):
    OPMethods.DP = Delay_s
    OPMethods.DR = Delay_m
    OPMethods.WAIT = Delay_w
    OPMethods.IN = Input
    OPMethods.ReadFlag = MaskLength
    OPMethods.Rmode = TestOnly
    return OPMethods
