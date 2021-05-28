import torch
from einops import rearrange
from torch import nn
import math
from models import ViT
from utils import trunc_normal_
from models import PyramidNet

class ViTMoCo(nn.Module):
    def __init__(self, image_size,
                 patch_size,
                 out_dim=128,
                 num_classes=-1,
                 dim=192,
                 depth=12,
                 heads=3,
                 mlp_dim=192*4,
                 T=0.2, m=0.999, memory_bank=False, K=10240):
        super(ViTMoCo, self).__init__()
        self.m = m
        self.K = K
        self.T = T
        self.memory_bank = memory_bank
        self.encoder_q = ViT(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)
        self.encoder_k = ViT(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)

        # self.encoder_q = PyramidNet('cifar10', 50, 200, num_classes=out_dim, bottleneck=True)
        # self.encoder_k = PyramidNet('cifar10', 50, 200, num_classes=out_dim, bottleneck=True)
        # dim_mlp = self.encoder_q.fc.in_features
        # # add mlp projection head
        # self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_q.fc)
        # self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_k.fc)
        #
        # self.predictor = nn.Sequential(
        #     nn.Linear(out_dim, out_dim),
        #     nn.BatchNorm1d(out_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(out_dim, out_dim),
        # )



        # add mlp projection head
        self.projection_q = nn.Sequential(nn.Linear(dim, mlp_dim), nn.ReLU(inplace=True),
                                          nn.Linear(mlp_dim, mlp_dim), nn.ReLU(inplace=True),
                                          nn.Linear(mlp_dim, out_dim))
        self.projection_k = nn.Sequential(nn.Linear(dim, mlp_dim), nn.ReLU(inplace=True),
                                          nn.Linear(mlp_dim, mlp_dim), nn.ReLU(inplace=True),
                                          nn.Linear(mlp_dim, out_dim))

        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim*4, out_dim),
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        if self.memory_bank:
            self.register_buffer("queue", torch.randn(2, out_dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=1)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys2, keys1):
        batch_size = keys2.shape[0]

        ptr = self.queue_ptr
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[0, :, ptr:ptr + batch_size] = keys2.T
        self.queue[0, :, ptr:ptr + batch_size] = keys1.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr = ptr

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        # idx_crops = torch.cumsum(torch.unique_consecutive(
        #     torch.tensor([inp.shape[-1] for inp in x]),
        #     return_counts=True,
        # )[1], 0)
        # start_idx = 0
        # for end_idx in idx_crops:
        #     _out = self.encoder_q(torch.cat(x[start_idx: end_idx]))
        #     if start_idx == 0:
        #         output = _out
        #     else:
        #         output = torch.cat((output, _out))
        #     start_idx = end_idx
        # # Run the head forward on the concatenated features.
        # q = self.predictor(output)
        for ii in range(len(x)):
            _out = self.predictor(self.projection_q(self.encoder_q(x[ii])))
            if ii==0:
                q = _out
            else:
                q = torch.cat((q,_out))
        q = nn.functional.normalize(q, dim=1)
        q = q.chunk(len(x))  #[crops, batch, dim]

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # k = self.encoder_k(torch.cat(x[:2]))
            for ii in range(2):
                _out = self.projection_k(self.encoder_k(x[ii]))
                if ii == 0:
                    k = _out
                else:
                    k = torch.cat((k, _out))

            # k = self.projection_k(output)
            k = nn.functional.normalize(k, dim=1)
            k = k.detach().chunk(2)   #[2, batch, dim]



        logits1, labels1 = self.get_logits_labels(q, k[1], idx=0) #shoundn't use q[1]
        logits2, labels2 = self.get_logits_labels(q, k[0], idx=1) #shoundn't use q[0]

        if self.memory_bank:
            self._dequeue_and_enqueue(k[1], k[0])
        return logits1, labels1, logits2, labels2

    def get_logits_labels(self, q, k, idx=0): #q:[crops, batch, dim]   k:[batch, dim]
        if self.memory_bank:
            for i in range(len(q)):
                if i+idx == 1: #drop the feature in q and k from same crop
                    continue
                # positive logits: Nx1
                l_pos = torch.einsum('nc,nc->n', (q[i], k)).unsqueeze(-1)
                # negative logits: NxK
                l_neg = torch.einsum('nc,ck->nk', (q[i], self.queue[idx].detach()))
                # logits: Nx(1+K)
                _logit = torch.cat([l_pos, l_neg], dim=1)
                if i == idx:
                    logits = _logit
                else:
                    logits = torch.cat((logits, _logit))

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        else:
            for i in range(len(q)):
                if i+idx == 1: #drop the feature in q and k from same crop
                    continue
                _logit = torch.mm(q[i], k.t())
                N = q[i].size(0)
                _label = range(N)
                _label = torch.LongTensor(_label).cuda()
                if i == idx:
                    logits = _logit
                    labels = _label
                else:
                    logits = torch.cat((logits, _logit))
                    labels = torch.cat((labels, _label))

        # apply temperature
        logits /= self.T
        return logits, labels
