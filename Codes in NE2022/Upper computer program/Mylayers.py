import torch
import torch.nn.functional as F


class RRAMsim(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(RRAMsim, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.Av = torch.nn.Parameter(0.1*torch.ones(out_features))
        if bias:
            self.isbias = True
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.isbias = False

    def forward(self, input):
        self.weight_ = F.hardtanh(self.weight+0.04*(2*torch.rand(self.in_features, self.out_features)-1), min_val=-1, max_val=1)
        if self.isbias:
            y = self.Av*torch.matmul(input, self.weight_) + self.bias
        else:
            y = self.Av*torch.matmul(input, self.weight_)
        return y
