import argparse
import os
from fjcommon import config_parser
import logging
from functools import partial

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
from regularization import RandAugment
from train import train_loop, adv_train_loop
from models import *
from optimization import *
from models.lib import PGDAttacker, NoOpAttacker, LayerNorm, MultiBatchNorm2d

from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--conf-path', default='./configs/cifar10/ViT_2.cf',  help='Path to network config')
parser.add_argument('--seed', default=5, type=int, help='seed for initializing training')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')

def main():
    args = parser.parse_args()

    training_args = config_parser.parse(args.conf_path)[0]

    training_args.name = args.conf_path[args.conf_path.rfind('/') + 1:][:-3]
    training_args.best_top1 = 0.
    training_args.device = torch.device('cuda', 0)
    training_args.save_path = './ckeckpoint/cifar10/{}'.format(training_args.name)

    training_args.writer = SummaryWriter(f"results/cifar10/{training_args.name}")

    if args.seed is not None:
        set_seed(args.seed)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.RandomCrop(32, padding=4),
    if training_args.randaug:
        transform_train.transforms.insert(0, RandAugment(training_args.randaug_n, training_args.randaug_m))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=training_args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=training_args.batch_size, shuffle=False)

    # if training_args.norm == 'IN':
    #     normlayer2d = partial(nn.InstanceNorm2d, affine=True)
    # elif training_args.norm == 'None':
    #     normlayer2d = nn.Identity
    # elif training_args.norm == 'LN':
    #     normlayer2d = LayerNorm
    # elif training_args.norm == 'MultiBatchNorm':
    #     normlayer2d = partial(MultiBatchNorm2d, split=4)
    # else:
    #     normlayer2d = nn.BatchNorm2d

    if training_args.model == 'ResNet':
        model = ResNet(resnet_depth=50, num_classes=training_args.num_classes,
                     dropblock_keep_probs=None, dropblock_size=7,
                     pre_activation=False,
                     se_ratio=0.25, drop_connect_keep_prob=None, use_resnetd_stem=True,
                     resnetd_shortcut=True, dropout_rate=training_args.dropout_rate,)
    elif training_args.model == 'SimpleDLA':
        model = SimpleDLA(num_classes=training_args.num_classes)
    elif training_args.model == 'PyramidNet':
        model = PyramidNet(dataset='cifar10', depth=training_args.depth, alpha=training_args.alpha, num_classes=10, bottleneck=True, norm_layer=normlayer2d)
    elif training_args.model == 'RedNet':
        model = RedNet(depth = training_args.depth,num_classes=10)
    elif training_args.model == 'san':
        model = san(sa_type=training_args.sa_type, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7], num_classes=10)
    elif training_args.model == 'eff_pretrain':
        model = EffcicentNet_pretrain(model_name = training_args.model_name, num_classes = 10)
    elif training_args.model == 'PyramidNetINV':
        model = PyramidNetINV(dataset='cifar10', depth=training_args.depth, alpha=training_args.alpha, num_classes=10, bottleneck=True)
    elif training_args.model == 'AdvPyramidNet':
        attacker = PGDAttacker(training_args.attack_iter, training_args.attack_epsilon, training_args.attack_step_size, prob_start_from_clean=0.2 if not training_args.evaluate else 0.0)
        model = AdvPyramidNet(dataset='cifar10', depth=training_args.depth, alpha=training_args.alpha, num_classes=10,bottleneck=True, attacker=attacker, writer=training_args.writer)

        if training_args.load:
            checkpoint = torch.load(training_args.load)
            to_merge = {}
            for key in checkpoint['model_state_dict']:
                if 'bn_final' in key:
                    tmp = key.split("bn_final")
                    aux_key = 'bn_final' + '.aux_bn' +tmp[1][0]+ tmp[1][1:]
                    to_merge[aux_key] = checkpoint['model_state_dict'][key]
                elif 'bn' in key:
                    tmp = key.split("bn")
                    aux_key = tmp[0] + 'bn' + tmp[1][0] + '.aux_bn' + tmp[1][1:]
                    to_merge[aux_key] = checkpoint['model_state_dict'][key]
            checkpoint['model_state_dict'].update(to_merge)

            model.load_state_dict(checkpoint['model_state_dict'])

    elif training_args.model == 'PyramidNet_metaacon':
        model = PyramidNet_metaacon(dataset='cifar10', depth=training_args.depth, alpha=training_args.alpha, num_classes=10, bottleneck=True)
    elif training_args.model == 'SwinTransformer':
        model = SwinTransformer(img_size=32, patch_size=1, in_chans=3, num_classes=10,
                 embed_dim=64, depths=[2, 4, 2], num_heads=[2, 8, 16],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False)
    elif training_args.model == 'ViT':
        model = ViT(image_size = 32,
                    patch_size = 4,
                    num_classes = 10,
                    dim = 512,
                    depth = 6,
                    heads = 8,
                    mlp_dim = 512)
    else:
        raise ValueError('model name error: {}'.format(training_args.model))
    model = model.cuda()

    logger.warning(f"Net Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if training_args.no_bias_decay:
        params = split_weights(model)
    else:
        params = model.parameters()

    if training_args.slow_fc:
        params = lower_fc(model, training_args.lr)
    else:
        params = model.parameters()

    if training_args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=training_args.lr, weight_decay=training_args.weight_decay)
    elif training_args.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=training_args.lr, momentum=0.9, weight_decay=training_args.weight_decay)
    elif training_args.optim == 'sam':
        optimizer = SAM(params, base_optimizer=torch.optim.SGD, rho=0.05, lr=training_args.lr, momentum=0.9, weight_decay=training_args.weight_decay)
    elif training_args.optim == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=training_args.lr, weight_decay=training_args.weight_decay)
    else:
        raise ValueError('optim name error: {}'.format(training_args.optim))

    criterion = create_loss_fn(training_args)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                training_args.warmup_steps,
                                                training_args.total_steps)

    #vis
    # visualize_network(training_args.writer, model, [32, 32])


    if args.resume:
        if os.path.isfile(args.resume):
            logger.warning(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=training_args.device)
            training_args.best_top1 = checkpoint['best_top1']
            training_args.start_step = checkpoint['step']

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            model_load_state_dict(model, checkpoint['model_state_dict'])

            logger.warning(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.warning(f"=> no checkpoint found at '{args.resume}'")

    if training_args.model != 'AdvPyramidNet':
        train_loop(training_args, model, criterion, optimizer, scheduler, train_loader, test_loader)
    else:
        adv_train_loop(training_args, model, criterion, optimizer, scheduler, train_loader, test_loader)




if __name__=='__main__':
    main()
