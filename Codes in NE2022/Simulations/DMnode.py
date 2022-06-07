import numpy as np


class DM_node:
    def __init__(self, T, S, alpha):
        self.T = T
        self.S = S
        self.alpha = alpha

    def test(self, Vi, Vm):
        Vm = Vm+self.alpha*(Vi-self.T-Vm)
        Vt = self.T-Vm
        Vo = np.clip(self.S*(Vi-Vt), 0, 1)
        return Vo, Vm
