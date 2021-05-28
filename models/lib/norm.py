import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, Size
from typing import Union, List
from torch.nn.parameter import Parameter
from torch.nn import init

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps: float = 1e-5,*args, **kwargs):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(1,self.num_features,1,1))
        self.bias = Parameter(torch.Tensor(torch.Tensor(1,self.num_features,1,1)))

        self.reset_parameters()


    def forward(self, input: Tensor) -> Tensor:
        normalized_shape = input.shape[1:]
        return F.layer_norm(input, normalized_shape, eps=self.eps).view(-1,*normalized_shape)*self.weight+self.bias
        #return out*self.weight+self.bias

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        init.zeros_(self.bias)

class MultiBatchNorm2d(nn.Module):
    def __init__(self, num_features, split=8):
        super(MultiBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.split=split

    def forward(self,x):
        if self.training:
            tmp=[]
            step = x.shape[0]//self.split
            for i in range(self.split):
                tmp.append(self.bn(x[i*step:(i+1)*step]))
            return torch.cat(tmp,dim=0)
        else:
            return self.bn(x)


class MultiBatchNorm1d(nn.Module):
    def __init__(self, num_features, split=8):
        super(MultiBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.split = split

    def forward(self, x):
        if self.training:
            tmp = []
            step = x.shape[0] // self.split
            for i in range(self.split):
                tmp.append(self.bn(x[i * step:(i + 1) * step]))
            return torch.cat(tmp, dim=0)
        else:
            return self.bn(x)