from models import PyramidNet
import torch.nn as nn
import torch.nn.functional as F
from models.lib import LayerNorm,MultiBatchNorm2d,MultiBatchNorm1d

from functools import partial

class PyramidNetSimCLR(nn.Module):
    def __init__(self, dataset, depth, alpha, out_dim, bottleneck=False, norm=False):
        super(PyramidNetSimCLR, self).__init__()
        if norm == 'IN':
            normlayer2d = partial(nn.InstanceNorm2d, affine=True)
            normlayer1d = partial(nn.InstanceNorm1d, affine=True)
        elif norm == 'None':
            normlayer2d = nn.Identity
            normlayer1d = nn.Identity
        elif norm == 'LN':
            normlayer2d = LayerNorm
            normlayer1d = nn.LayerNorm
        elif norm == 'MultiBatchNorm':
            normlayer2d = MultiBatchNorm2d
            normlayer1d = MultiBatchNorm1d
        else:
            normlayer2d = nn.BatchNorm2d
            normlayer1d = nn.BatchNorm1d

        self.backbone = PyramidNet(dataset, depth, alpha, num_classes=out_dim, bottleneck=bottleneck, norm_layer=normlayer2d)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                         normlayer1d(dim_mlp),
                                         nn.ReLU(inplace=True),
                                         self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)