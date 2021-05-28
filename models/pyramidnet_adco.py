from models import PyramidNet
import torch.nn as nn
import torch.nn.functional as F
import torch


class PyramidNetAdCo(nn.Module):
    def __init__(self, dataset, depth, alpha, T, out_dim=128, bottleneck=False, m=0.999):
        super(PyramidNetAdCo, self).__init__()
        self.m = m
        self.T = T
        self.encoder_q = PyramidNet(dataset, depth, alpha, num_classes=out_dim, bottleneck=bottleneck)
        self.encoder_k = PyramidNet(dataset, depth, alpha, num_classes=out_dim, bottleneck=bottleneck)
        dim_mlp = self.encoder_q.fc.in_features

        # add mlp projection head
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_k.fc)

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

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        q_pred = q
        k_pred = self.encoder_q(im_k)  # queries: NxC
        k_pred = nn.functional.normalize(k_pred, dim=1)
        with torch.no_grad():  # no gradient to keys
            # if update_key_encoder:
            self._momentum_update_key_encoder()  # update the key encoder

            q = self.encoder_k(im_q)  # keys: NxC
            q = nn.functional.normalize(q, dim=1)
            q = q.detach()

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            k = k.detach()

        return q_pred, k_pred, q, k


class AdversaryNegatives(nn.Module):
    def __init__(self, bank_size, dim):
        super(AdversaryNegatives, self).__init__()
        self.register_buffer("W", torch.randn(dim, bank_size))
        self.register_buffer("v", torch.zeros(dim, bank_size))

    def forward(self, q):
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)

        logit = torch.einsum('nc,ck->nk', [q, memory_bank])
        return memory_bank, self.W, logit

    def update(self, m, lr, weight_decay, g):
        g = g + weight_decay * self.W
        self.v = m * self.v + g
        self.W = self.W - lr * self.v

    def print_weight(self):
        print(torch.sum(self.W).item())
