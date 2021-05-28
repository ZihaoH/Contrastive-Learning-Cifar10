import torch
import torch.nn.functional as F
from torch import nn


class DropBlock2D(nn.Module):
    r"""Randomly zeroes spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        if not self.training or self.keep_prob == 1 or self.keep_prob == None:
            return input
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        Msum = F.conv2d(M,
                        torch.ones((input.shape[1], 1, self.block_size, self.block_size)).to(device=input.device,
                                                                                             dtype=input.dtype),
                        padding=self.block_size // 2,
                        groups=input.shape[1])
        torch.set_printoptions(threshold=5000)
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        return input * mask * mask.numel() / mask.sum()  # TODO input * mask * self.keep_prob ?


def drop_connect(inputs, is_training, keep_prob=None):
    """Apply drop connect.
    Args:
      inputs: `Tensor` input tensor.
      is_training: `bool` if True, the model is in training mode.
      keep_prob (float, optional): probability of an element to be kept.
    Returns:
      A output tensor, which should have the same shape as input.
    """
    if not is_training or keep_prob is None or keep_prob == 1:
        return inputs

    batch_size = inputs.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.Tensor(batch_size, 1, 1, 1).uniform_(0, 1)
    binary_tensor = torch.floor(random_tensor).to(device=inputs.device, dtype=inputs.dtype)
    output = (inputs / keep_prob) * binary_tensor
    return output


def get_drop_connect_keep_prob(init_keep_prob, block_num, total_blocks):
    """Get drop connect rate for the ith block."""
    if init_keep_prob is not None:
        return init_keep_prob-(1-init_keep_prob) * float(block_num) / total_blocks
    else:
        return None
