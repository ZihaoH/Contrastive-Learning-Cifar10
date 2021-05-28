from models import PyramidNet
import torch.nn as nn
import torch.nn.functional as F
import torch

class PyramidNetBYOL(nn.Module):
    def __init__(self, dataset, depth, alpha, out_dim=128, bottleneck=False, m=0.999):
        super(PyramidNetBYOL, self).__init__()
        self.m = m
        self.encoder_q = PyramidNet(dataset, depth, alpha, num_classes=out_dim, bottleneck=bottleneck)
        self.encoder_k = PyramidNet(dataset, depth, alpha, num_classes=out_dim, bottleneck=bottleneck)
        dim_mlp = self.encoder_q.fc.in_features

        # add mlp projection head
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(inplace=True), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(inplace=True), self.encoder_k.fc)

        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def forward(self, x1, x2):
        """
        Input:
            x1: a batch of query images
            x2: a batch of key images
        Output:
            q1,q2,k1,k2
        """

        # compute query features
        q1, q2 = self.predictor(self.encoder_q(x1)), self.predictor(self.encoder_q(x2))

        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k1, k2 = self.encoder_k(x1), self.encoder_k(x2)  # keys: NxC
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)

        return q1, q2, k1, k2
