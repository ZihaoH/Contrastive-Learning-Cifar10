import torch
from torch import nn
import functools
from regularization import DropBlock2D, drop_connect, get_drop_connect_keep_prob

MOVING_AVERAGE_DECAY = 0.1
ACTIVATION = nn.ReLU


class NormActivation(nn.Module):
    def __init__(self, filters, bn_momentum, nonlinearity=True, init_zero=False):
        super(NormActivation, self).__init__()
        self.nonlinearity = nonlinearity
        self.BN = nn.BatchNorm2d(filters, momentum=bn_momentum)
        if init_zero:
            tmp = torch.nn.Parameter(torch.zeros(filters))
            self.BN.weight = tmp
        if self.nonlinearity:
            self.act = ACTIVATION(True)

    def forward(self, x):
        if self.nonlinearity:
            return self.act(self.BN(x))
        else:
            return self.BN(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_filters,
                 se_ratio,
                 expand_ratio=1):
        super(SqueezeExcitation, self).__init__()
        num_reduced_filters = max(1, int(in_filters * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_filters, out_channels=num_reduced_filters, kernel_size=1, bias=True),
            ACTIVATION(True),
            nn.Conv2d(in_channels=num_reduced_filters, out_channels=in_filters * expand_ratio, kernel_size=1,
                      bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.se(x) * x


class BottleNeckBlock(nn.Module):
    def __init__(self, inplanes, planes, strides,
                 use_projection=False,
                 dropblock_keep_prob=None, dropblock_size=None,
                 pre_activation=False,
                 resnetd_shortcut=False, se_ratio=None,
                 drop_connect_keep_prob=None, bn_momentum=MOVING_AVERAGE_DECAY):
        """Bottleneck block variant for residual networks with BN after convolutions.
          Args:
            filters: `int` number of filters for the first two convolutions. Note that
                the third and final convolution will use 4 times as many filters.
            strides: `int` block stride. If greater than 1, this block will ultimately
                downsample the input.
            use_projection: `bool` for whether this block should use a projection
                shortcut (versus the default identity shortcut). This is usually `True`
                for the first block of a block group, which may change the number of
                filters and the resolution.
            dropblock_keep_prob: `float` or `Tensor` keep_prob parameter of DropBlock.
                "None" means no DropBlock.
            dropblock_size: `int` size parameter of DropBlock. Will not be used if
                dropblock_keep_prob is "None".
            pre_activation: whether to use pre-activation ResNet (ResNet-v2).
            resnetd_shortcut: `bool` if True, apply the resnetd style modification to
                the shortcut connection.
            se_ratio: `float` or None. Squeeze-and-Excitation ratio for the SE layer.
            drop_connect_keep_prob: `float` or `Tensor` keep_prob parameter of DropConnect.
                    "None" means no DropBlock.
            bn_momentum: `float` momentum for batch norm layer.
          Returns:
            The output `Tensor` of the block.
          """
        super(BottleNeckBlock, self).__init__()
        self.pre_activation = pre_activation
        self.use_projection = use_projection
        self.resnetd_shortcut = resnetd_shortcut
        self.strides = strides
        self.se_ratio = se_ratio
        self.drop_connect_keep_prob = drop_connect_keep_prob
        if self.pre_activation:
            self.pre_act_BN = NormActivation(inplanes, bn_momentum)
        if self.use_projection:
            # Projection shortcut only in first block within a group. Bottleneck blocks
            # end with 4 times the number of filters.
            outplanes = 4 * planes
            if self.resnetd_shortcut and self.strides == 2:
                self.avgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
                self.shortcut_conv = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, bias=False)
            else:
                self.shortcut_conv = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=2,
                                               stride=strides, bias=False)
            if not self.pre_activation:
                self.shortcut_BN_act = NormActivation(outplanes, bn_momentum, nonlinearity=False)

        self.shortcut_dropblock = DropBlock2D(keep_prob=dropblock_keep_prob, block_size=dropblock_size)

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False),
            NormActivation(planes, bn_momentum),
            DropBlock2D(keep_prob=dropblock_keep_prob, block_size=dropblock_size),
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1, stride=strides, bias=False),
            NormActivation(planes, bn_momentum),
            DropBlock2D(keep_prob=dropblock_keep_prob, block_size=dropblock_size),
            nn.Conv2d(in_channels=planes, out_channels=planes * 4, kernel_size=1, bias=False),
        )
        if not self.pre_activation:
            self.norm_act3 = NormActivation(planes * 4, bn_momentum, nonlinearity=False)
            self.inputs_dropblock = DropBlock2D(keep_prob=dropblock_keep_prob, block_size=dropblock_size)

            if self.se_ratio is not None and self.se_ratio > 0 and self.se_ratio <= 1:
                self.se_block = SqueezeExcitation(in_filters=planes * 4, se_ratio=se_ratio)
            self.end_act = ACTIVATION(True)

    def forward(self, inputs):
        shortcut = inputs
        if self.pre_activation:
            inputs = self.pre_act_BN(inputs)
        if self.use_projection:
            if self.resnetd_shortcut and self.strides == 2:
                shortcut = self.avgPool(inputs)
                shortcut = self.shortcut_conv(shortcut)
            else:
                shortcut = self.shortcut_conv(inputs)
            if not self.pre_activation:
                shortcut = self.shortcut_BN_act(shortcut)

        shortcut = self.shortcut_dropblock(shortcut)
        inputs = self.bottle_neck(inputs)

        if self.pre_activation:
            return inputs + shortcut
        else:
            inputs = self.norm_act3(inputs)
            inputs = self.inputs_dropblock(inputs)

            if self.se_ratio is not None and self.se_ratio > 0 and self.se_ratio <= 1:
                inputs = self.se_block(inputs)

            if self.drop_connect_keep_prob is not None:
                inputs = drop_connect(inputs, self.training, self.drop_connect_keep_prob)
            return self.end_act(inputs + shortcut)


class BlockGroup(nn.Module):
    def __init__(self, inplanes, planes, block_fn, blocks, strides,
                 dropblock_keep_prob=None, dropblock_size=None,
                 pre_activation=False, se_ratio=None,
                 resnetd_shortcut=False, drop_connect_keep_prob=None,
                 bn_momentum=MOVING_AVERAGE_DECAY):
        super(BlockGroup, self).__init__()
        self.block_1 = block_fn(inplanes, planes, strides, use_projection=True,
                                dropblock_keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size,
                                pre_activation=pre_activation,
                                resnetd_shortcut=resnetd_shortcut, se_ratio=se_ratio,
                                drop_connect_keep_prob=drop_connect_keep_prob, bn_momentum=bn_momentum)
        block_n = []
        for _ in range(1, blocks):
            block_n.append(
                block_fn(planes * 4, planes, strides=1,
                         dropblock_keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size,
                         pre_activation=pre_activation,
                         resnetd_shortcut=resnetd_shortcut, se_ratio=se_ratio,
                         drop_connect_keep_prob=drop_connect_keep_prob, bn_momentum=bn_momentum)
            )
        self.block_n = nn.Sequential(*block_n)

    def forward(self, inputs):
        inputs = self.block_1(inputs)
        inputs = self.block_n(inputs)
        return inputs


class ResNet(nn.Module):
    def __init__(self, resnet_depth, num_classes,
                 dropblock_keep_probs=None, dropblock_size=None,
                 pre_activation=False,
                 se_ratio=None, drop_connect_keep_prob=None, use_resnetd_stem=False,
                 resnetd_shortcut=False, dropout_rate=None,
                 bn_momentum=MOVING_AVERAGE_DECAY):
        super(ResNet, self).__init__()
        model_params = {
            50: {'block': BottleNeckBlock, 'layers': [3, 4, 6, 3]},
            101: {'block': BottleNeckBlock, 'layers': [3, 4, 23, 3]},
            152: {'block': BottleNeckBlock, 'layers': [3, 8, 36, 3]},
            200: {'block': BottleNeckBlock, 'layers': [3, 24, 36, 3]},
            270: {'block': BottleNeckBlock, 'layers': [4, 29, 53, 4]},
            350: {'block': BottleNeckBlock, 'layers': [4, 36, 72, 4]},
            420: {'block': BottleNeckBlock, 'layers': [4, 44, 87, 4]}
        }
        if resnet_depth not in model_params:
            raise ValueError('Not a valid resnet_depth:', resnet_depth)
        params = model_params[resnet_depth]
        self._build_model(params['block'], params['layers'], num_classes,
                          dropblock_keep_probs=dropblock_keep_probs,
                          dropblock_size=dropblock_size,
                          pre_activation=pre_activation,
                          use_resnetd_stem=use_resnetd_stem,
                          resnetd_shortcut=resnetd_shortcut,
                          se_ratio=se_ratio,
                          drop_connect_keep_prob=drop_connect_keep_prob,
                          dropout_rate=dropout_rate,
                          bn_momentum=bn_momentum)

    def _build_model(self,
                     block_fn,
                     layers,
                     num_classes,
                     use_resnetd_stem=False,
                     resnetd_shortcut=False,
                     drop_connect_keep_prob=None,
                     se_ratio=None,
                     dropout_rate=None,
                     dropblock_keep_probs=None,
                     dropblock_size=None,
                     pre_activation=False,
                     bn_momentum=MOVING_AVERAGE_DECAY):
        if dropblock_keep_probs is None:
            dropblock_keep_probs = [None] * 4
        if not isinstance(dropblock_keep_probs, list) or len(dropblock_keep_probs) != 4:
            raise ValueError('dropblock_keep_probs is not valid:', dropblock_keep_probs)

        if use_resnetd_stem:
            self.stem = [
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
                NormActivation(32, bn_momentum),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
                NormActivation(32, bn_momentum),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            ]
        else:
            self.stem = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False)]

        if not pre_activation:
            self.stem.append(NormActivation(64, bn_momentum))
        self.stem = nn.Sequential(*self.stem)

        num_layers = len(layers) + 1
        stride_c2 = 2
        custom_block_group = functools.partial(
            BlockGroup,
            dropblock_size=dropblock_size,
            block_fn=block_fn,
            pre_activation=pre_activation,
            se_ratio=se_ratio,
            resnetd_shortcut=resnetd_shortcut,
            bn_momentum=bn_momentum)

        self.G1 = custom_block_group(inplanes=64, planes=64, blocks=layers[0], strides=stride_c2,
                                     dropblock_keep_prob=dropblock_keep_probs[0],
                                     drop_connect_keep_prob=get_drop_connect_keep_prob(drop_connect_keep_prob, 2,
                                                                                       num_layers))
        self.G2 = custom_block_group(inplanes=256, planes=128, blocks=layers[1], strides=2,
                                     dropblock_keep_prob=dropblock_keep_probs[1],
                                     drop_connect_keep_prob=get_drop_connect_keep_prob(drop_connect_keep_prob, 3,
                                                                                       num_layers))
        self.G3 = custom_block_group(inplanes=512, planes=256, blocks=layers[2], strides=2,
                                     dropblock_keep_prob=dropblock_keep_probs[2],
                                     drop_connect_keep_prob=get_drop_connect_keep_prob(drop_connect_keep_prob, 4,
                                                                                       num_layers))
        self.G4 = custom_block_group(inplanes=1024, planes=512, blocks=layers[3], strides=2,
                                     dropblock_keep_prob=dropblock_keep_probs[3],
                                     drop_connect_keep_prob=get_drop_connect_keep_prob(drop_connect_keep_prob, 5,
                                                                                       num_layers))
        self.tail1 = []
        if pre_activation:
            self.tail1.append(NormActivation(2048, bn_momentum))
        self.tail1.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.tail1 = nn.Sequential(*self.tail1)

        self.tail2 = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, inputs):
        inputs = self.stem(inputs)
        inputs = self.G1(inputs)
        inputs = self.G2(inputs)
        inputs = self.G3(inputs)
        inputs = self.G4(inputs)
        inputs = self.tail1(inputs)
        return self.tail2(inputs.view(inputs.shape[:2]))



if __name__ == '__main__':
    # bb=BlockGroup(inplanes=64, planes=64, blocks=5, strides=2,
    #            dropblock_keep_prob=0.8,
    #            drop_connect_keep_prob=0.7,
    #            dropblock_size=7,
    #            block_fn=BottleNeckBlock,
    #            pre_activation=False,
    #            se_ratio=0.5,
    #            resnetd_shortcut=True,
    #            bn_momentum=MOVING_AVERAGE_DECAY
    #            )
    model = ResNet(resnet_depth=50, num_classes=2,
                 dropblock_keep_probs=None, dropblock_size=None,
                 pre_activation=False,
                 se_ratio=0.5, drop_connect_keep_prob=0.8, use_resnetd_stem=True,
                 resnetd_shortcut=True, dropout_rate=0.2,
                 bn_momentum=MOVING_AVERAGE_DECAY).cuda()

    x = torch.randn(4, 3, 224, 224).cuda()
    label = torch.tensor([1,0,0,1]).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(1000):
        optimizer.zero_grad()
        y = model(x)
        loss = criterion(y, label)
        print(loss.item())
        loss.backward()
        optimizer.step()
    print(y.shape)
