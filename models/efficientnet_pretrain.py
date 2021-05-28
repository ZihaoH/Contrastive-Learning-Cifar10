# from efficientnet_pytorch import EfficientNet
from .efficientnet import EfficientNet
from efficientnet_pytorch.utils import round_filters
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class EffcicentNet_pretrain(nn.Module):
    def __init__(self,model_name = 'efficientnet-b0', num_classes = 10):
        super(EffcicentNet_pretrain, self).__init__()
        self.feature_net = EfficientNet.from_pretrained(model_name)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self.feature_net._global_params.dropout_rate)
        out_channels = round_filters(1280, self.feature_net._global_params)
        self._fc = nn.Linear(out_channels, num_classes)
        self.swish = nn.SiLU(True)

    def forward(self,x):
        x = self.feature_net.extract_features(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x



