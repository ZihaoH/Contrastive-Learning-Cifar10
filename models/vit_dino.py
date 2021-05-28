import torch
from einops import rearrange
from torch import nn
import math
from models import ViT
from utils import trunc_normal_


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class ViTDINO(nn.Module):
    def __init__(self, image_size,
                 patch_size,
                 out_dim,
                 num_classes=-1,
                 dim=512,
                 depth=6,
                 heads=8,
                 mlp_dim=512,
                 use_bn=False,
                 norm_last_layer=True,
                 hidden_dim=1024,
                 bottleneck_dim=512):
        super(ViTDINO, self).__init__()
        self.backbone = ViT(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)
        self.dino_head = DINOHead(in_dim=dim, out_dim=out_dim, use_bn=use_bn, norm_last_layer=norm_last_layer, nlayers=3,
                                  hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)

    def forward(self, x):
        return self.dino_head(self.backbone(x))


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, ViT_DINO):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.model = ViT_DINO

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.model.backbone(torch.cat(x[start_idx: end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.model.dino_head(output)

class PackStudentTeacher(nn.Module):
    def __init__(self, student, teacher):
        super(PackStudentTeacher, self).__init__()
        self.student = student
        self.teacher = teacher

if __name__ == '__main__':
    model = ViTDINO(image_size=32,
                    patch_size=4,
                    out_dim=256,
                    num_classes=-1,
                    dim=512,
                    depth=6,
                    heads=8,
                    mlp_dim=512).cuda()
    x = torch.randn(256, 3, 32, 32).cuda()
    y = model(x)
    print(y.shape)
