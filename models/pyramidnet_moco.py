from models import PyramidNet
import torch.nn as nn
import torch.nn.functional as F
import torch

class PyramidNetMoCo(nn.Module):
    def __init__(self, dataset, depth, alpha, T, out_dim=128, bottleneck=False, m=0.999, memory_bank = False, K=10240):
        super(PyramidNetMoCo, self).__init__()
        self.m = m
        self.K = K
        self.T = T
        self.memory_bank = memory_bank
        self.encoder_q = PyramidNet(dataset, depth, alpha, num_classes=out_dim, bottleneck=bottleneck)
        self.encoder_k = PyramidNet(dataset, depth, alpha, num_classes=out_dim, bottleneck=bottleneck)
        dim_mlp = self.encoder_q.fc.in_features

        # add mlp projection head
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_k.fc)

        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
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

        # return q1, q2, k1, k2

        logits1, labels1 = self.get_logits_labels(q1, k2, idx=0)
        logits2, labels2 = self.get_logits_labels(q2, k1, idx=1)

        if self.memory_bank:
            self._dequeue_and_enqueue(k2, k1)
        return logits1, labels1, logits2, labels2

    def get_logits_labels(self, q1, k2, idx=0):
        if self.memory_bank:
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', (q1, k2)).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', (q1, self.queue[idx].clone().detach()))
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        else:
            logits = torch.mm(q1, k2.t())
            N = q1.size(0)
            labels = range(N)
            labels = torch.LongTensor(labels).cuda()

        # apply temperature
        logits /= self.T
        return logits, labels