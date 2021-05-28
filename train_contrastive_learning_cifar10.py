import argparse
import os
from fjcommon import config_parser
import logging
import math

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
from regularization import RandAugment
from train import MoCoV3,SimCLR,AdCo,BYOL,DINO,MoCoV3Multicrop
from models import *
from optimization import *
from dataload import get_dataset,get_multi_crop_dataset

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--conf-path', default='./configs/ContrastiveLearning/MoCo_ViT_2.cf', help='Path to network config')
parser.add_argument('--seed', default=5, type=int, help='seed for initializing training')



def main():
    args = parser.parse_args()

    training_args = config_parser.parse(args.conf_path)[0]

    training_args.name = args.conf_path[args.conf_path.rfind('/') + 1:][:-3]
    training_args.best_top1 = 0.
    training_args.device = torch.device('cuda', 0)
    training_args.save_path = './ckeckpoint/ContrastiveLearning/{}'.format(training_args.name)

    training_args.writer = SummaryWriter(f"results/ContrastiveLearning/{training_args.name}")

    if args.seed is not None:
        set_seed(args.seed)

    if training_args.multicrop == True:
        train_dataset = get_multi_crop_dataset('./data', training_args)
    else:
        train_dataset = get_dataset('./data')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=training_args.batch_size, shuffle=True,
        num_workers=training_args.workers, pin_memory=True, drop_last=True)

    if training_args.model == 'MoCo':
        model = PyramidNetMoCo(dataset='cifar10', depth=training_args.depth, alpha=training_args.alpha,
                                 T=training_args.temperature, out_dim=training_args.out_dim,
                                 bottleneck=True, m=training_args.m,
                                 memory_bank = training_args.memory_bank, K=training_args.K)
    elif training_args.model == 'MoCo_ViT':
        model = ViTMoCo(image_size=32,
                         patch_size=training_args.patch_size, out_dim=training_args.out_dim, num_classes=-1,
                         dim=training_args.dim, depth=training_args.depth, heads=training_args.heads,
                         mlp_dim=training_args.mlp_dim, T=training_args.temperature, m=training_args.m,
                         memory_bank=training_args.memory_bank, K=training_args.K)
    elif training_args.model == 'SimCLR':
        model = PyramidNetSimCLR(dataset='cifar10', depth=training_args.depth, alpha=training_args.alpha,
                                 out_dim=training_args.out_dim,
                                 bottleneck=True, norm=training_args.norm)
    elif training_args.model == 'AdCo':
        model = PyramidNetAdCo(dataset='cifar10', depth=training_args.depth, alpha=training_args.alpha,
                               T=training_args.temperature, out_dim=training_args.out_dim,
                               bottleneck=True, m=training_args.m)
    elif training_args.model == 'BYOL':
        model = PyramidNetBYOL(dataset='cifar10', depth=training_args.depth, alpha=training_args.alpha,
                               out_dim=training_args.out_dim, bottleneck=True, m=training_args.m)
    elif training_args.model == 'DINO':
        student = ViTDINO(image_size=32, patch_size=training_args.patch_size, out_dim=training_args.out_dim,
                         num_classes=-1, dim=training_args.dim, depth=training_args.depth, heads=training_args.heads,
                         mlp_dim=training_args.mlp_dim, use_bn=False,
                         norm_last_layer=training_args.norm_last_layer, hidden_dim=training_args.hidden_dim,
                         bottleneck_dim=training_args.bottleneck_dim)

        teacher = ViTDINO(image_size=32, patch_size=training_args.patch_size, out_dim=training_args.out_dim,
                          num_classes=-1, dim=training_args.dim, depth=training_args.depth, heads=training_args.heads,
                          mlp_dim=training_args.mlp_dim, use_bn=False,
                          norm_last_layer=training_args.norm_last_layer, hidden_dim=training_args.hidden_dim,
                          bottleneck_dim=training_args.bottleneck_dim)
        teacher.load_state_dict(student.state_dict())
        student = MultiCropWrapper(student)
        teacher = MultiCropWrapper(teacher)
        model = PackStudentTeacher(student, teacher)
    else:
        raise ValueError('model name error: {}'.format(training_args.model))
    model = model.cuda()

    if training_args.no_bias_decay:
        params = split_weights(model)
    else:
        params = model.parameters()

    if training_args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=training_args.lr, weight_decay=training_args.weight_decay)
    elif training_args.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=training_args.lr, momentum=0.9, weight_decay=training_args.weight_decay)
    elif training_args.optim == 'adamw' and training_args.model == 'DINO':
        params_groups = utils.get_params_groups(model.student)
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif training_args.optim == 'adamw' and training_args.model == 'MoCo_ViT':
        params_groups = utils.get_params_groups(model.encoder_q)
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    else:
        raise ValueError('optim name error: {}'.format(training_args.optim))

    warmup_steps = training_args.warmup_epochs * math.floor(50000 / training_args.batch_size)
    total_steps = training_args.epochs * math.floor(50000 / training_args.batch_size)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                warmup_steps,
                                                total_steps)
    if training_args.model == 'MoCo':
        moco = MoCoV3(args=training_args, model=model, optimizer=optimizer, scheduler=scheduler)
        moco.train(train_loader)
    elif training_args.model == 'MoCo_ViT':
        moco = MoCoV3Multicrop(args=training_args, model=model, optimizer=optimizer, scheduler=scheduler)
        moco.train(train_loader)
    elif training_args.model == 'SimCLR':
        simclr = SimCLR(args=training_args, model=model, optimizer=optimizer, scheduler=scheduler)
        simclr.train(train_loader)
    elif training_args.model == 'AdCo':
        memory_bank = AdversaryNegatives(bank_size=training_args.bank_size, dim=training_args.out_dim)
        adco = AdCo(args=training_args, model=model, memory_bank=memory_bank, optimizer=optimizer, scheduler=scheduler)
        adco.train(train_loader)
    elif training_args.model == 'BYOL':
        byol = BYOL(args=training_args, model=model, optimizer=optimizer, scheduler=scheduler)
        byol.train(train_loader)
    elif training_args.model == 'DINO':
        # ============ init schedulers ... ============
        lr_schedule = utils.cosine_scheduler(
            training_args.lr * training_args.batch_size / 256.,  # linear scaling rule
            training_args.min_lr,
            training_args.epochs, len(train_loader),
            warmup_epochs=training_args.warmup_epochs,
        )
        wd_schedule = utils.cosine_scheduler(
            training_args.weight_decay,
            training_args.weight_decay_end,
            training_args.epochs, len(train_loader),
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_schedule = utils.cosine_scheduler(training_args.momentum_teacher, 1.,
                                                   training_args.epochs, len(train_loader))
        dino = DINO(training_args, model.student, model.teacher, optimizer,  lr_schedule, wd_schedule, momentum_schedule)
        dino.train(train_loader)

if __name__ == "__main__":
    main()
