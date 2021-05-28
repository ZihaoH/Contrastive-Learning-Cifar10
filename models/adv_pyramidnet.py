import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from functools import partial

from models import PyramidNet
from .lib import PGDAttacker, NoOpAttacker

class MixBatchNorm2d(nn.BatchNorm2d):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].
    If you set batch_type as 'mix', this network will using one batchnorm (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    During training, the batch_type should be set as 'mix'.
    During validation, we only need the results of the features using some specific batchnormalization.
    if you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.
    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            # input0 = self.aux_bn(input[: batch_size // 2])
            # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
            input0 = super(MixBatchNorm2d, self).forward(input[:batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input


def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')


class AdvPyramidNet(PyramidNet):
    def __init__(self, dataset, depth, alpha, num_classes, bottleneck=False, norm_layer=MixBatchNorm2d, attacker=NoOpAttacker(),writer=None):
        super().__init__(dataset=dataset, depth=depth, alpha=alpha, num_classes=num_classes, bottleneck=bottleneck, norm_layer=norm_layer)
        self.attacker = attacker
        self.mixbn = False
        self.writer=writer
        self.count=0

    def set_attacker(self, attacker):
        self.attacker = attacker

    def set_mixbn(self, mixbn):
        self.mixbn = mixbn

    def forward(self, x, labels=None):
        training = self.training
        input_len = len(x)
        # only during training do we need to attack, and cat the clean and auxiliary pics
        if training:
            self.eval()
            self.apply(to_adv_status)
            if isinstance(self.attacker, NoOpAttacker):
                images = x
                targets = labels
            else:
                aux_images, _ = self.attacker.attack(x, labels, self._forward_impl)
                import torchvision
                if self.count%100==0:
                    grid1 = torchvision.utils.make_grid(aux_images, nrow=16)
                    grid2 = torchvision.utils.make_grid(x, nrow=16)
                    self.writer.add_image('aux_images', grid1,self.count)
                    self.writer.add_image('images', grid2,self.count)
                self.count+=1

                images = torch.cat([x, aux_images], dim=0)
                targets = torch.cat([labels, labels], dim=0)
            self.train()
            if self.mixbn:
                # the DataParallel usually cat the outputs along the first dimension simply,
                # so if we don't change the dimensions, the outputs will be something like
                # [clean_batches_gpu1, adv_batches_gpu1, clean_batches_gpu2, adv_batches_gpu2...]
                # Then it will be hard to distinguish clean batches and adversarial batches.
                self.apply(to_mix_status)
                return self._forward_impl(images).view(2, input_len, -1).transpose(1, 0), targets.view(2,input_len).transpose(1, 0)
            else:
                self.apply(to_clean_status)
                return self._forward_impl(images), targets
        else:
            images = x
            targets = labels
            return self._forward_impl(images), targets
