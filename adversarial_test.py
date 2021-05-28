import argparse
import os
from fjcommon import config_parser
import logging
from tqdm import tqdm

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
from regularization import RandAugment
from train import train_loop, adv_train_loop
from models import *
from optimization import *
from models.lib import PGDAttacker, NoOpAttacker



logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# parser.add_argument('--conf-path', default='./configs/cifar10/AdvPyramidNet_1.cf',  help='Path to network config')
# parser.add_argument('--resume', default='./ckeckpoint/cifar10/AdvPyramidNet_1/AdvPyramidNet_1_last.pth.tar', type=str, help='path to checkpoint')
# parser.add_argument('--conf-path', default='./configs/cifar10/PyramidNet_5_no_norm.cf',  help='Path to network config')
# parser.add_argument('--resume', default='./ckeckpoint/cifar10/PyramidNet_5_no_norm/PyramidNet_5_no_norm_last.pth.tar', type=str, help='path to checkpoint'
# parser.add_argument('--conf-path', default='./configs/SimCLR/cifar10/PyramidNet_linear_1.cf',  help='Path to network config')
# parser.add_argument('--resume', default='./ckeckpoint/cifar10/PyramidNet_linear_1_SimCLR/PyramidNet_linear_1_last.pth.tar', type=str, help='path to checkpoint')
parser.add_argument('--conf-path', default='./configs/cifar10/MoCo_PyramidNet_5.cf',  help='Path to network config')
parser.add_argument('--resume', default='./ckeckpoint/cifar10/MoCo_PyramidNet_5/PyramidNet_5_last.pth.tar', type=str, help='path to checkpoint')


parser.add_argument('--seed', default=5, type=int, help='seed for initializing training')


parser.add_argument('--attack-iter', default=1, type=float, help='seed for initializing training')
parser.add_argument('--attack-epsilon', default=8, type=float, help='seed for initializing training')
parser.add_argument('--attack-step-size', default=1, type=float, help='seed for initializing training')
parser.add_argument('--evaluate', default=True, type=bool, help='seed for initializing training')


def main():
    args = parser.parse_args()

    attacker_test = PGDAttacker(args.attack_iter, args.attack_epsilon, args.attack_step_size,
                           prob_start_from_clean=0.2 if not args.evaluate else 0.0)

    training_args = config_parser.parse(args.conf_path)[0]

    training_args.name = args.conf_path[args.conf_path.rfind('/') + 1:][:-3]
    training_args.best_top1 = 0.
    training_args.device = torch.device('cuda', 0)
    training_args.save_path = './ckeckpoint/cifar10/{}'.format(training_args.name)

    training_args.writer = SummaryWriter(f"results/cifar10/{training_args.name}")

    if args.seed is not None:
        set_seed(args.seed)


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=training_args.batch_size, shuffle=False)


    if training_args.model == 'ResNet':
        model = ResNet(resnet_depth=50, num_classes=training_args.num_classes,
                     dropblock_keep_probs=None, dropblock_size=7,
                     pre_activation=False,
                     se_ratio=0.25, drop_connect_keep_prob=None, use_resnetd_stem=True,
                     resnetd_shortcut=True, dropout_rate=training_args.dropout_rate,)
    elif training_args.model == 'SimpleDLA':
        model = SimpleDLA(num_classes=training_args.num_classes)
    elif training_args.model == 'PyramidNet':
        model = PyramidNet(dataset='cifar10', depth=training_args.depth, alpha=training_args.alpha, num_classes=10, bottleneck=True)
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

    else:
        raise ValueError('model name error: {}'.format(training_args.model))
    model = model.cuda()

    logger.warning(f"Net Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if args.resume:
        if os.path.isfile(args.resume):
            logger.warning(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=training_args.device)
            model_load_state_dict(model, checkpoint['model_state_dict'])
            logger.warning(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.warning(f"=> no checkpoint found at '{args.resume}'")

    model.train(False)

    correct = 0
    correct_adv = 0
    total = 0

    for test_x, test_y in tqdm(test_loader):
        images, labels = test_x.cuda(), test_y.cuda()
        # grid = torchvision.utils.make_grid(images, nrow=16)
        # image1 = transforms.ToPILImage()(grid.cpu().clone())
        # image1.save('./original.png')
        aux_images, _ = attacker_test.attack(images, labels, model._forward_impl)

        # grid1 = torchvision.utils.make_grid(aux_images, nrow=16)
        # image = transforms.ToPILImage()(grid1.cpu().clone())
        # image.save('./adv_20.png')
        # image.show()

        with torch.no_grad():
            output = model(images)
            if isinstance(output,tuple):
                output,_=output
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            output = model(aux_images)
            if isinstance(output, tuple):
                output, _ = output
            _, predicted = torch.max(output.data, 1)
            correct_adv += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    accuracy_adv = 100 * correct_adv / total

    print('ragular: {}, adversarial: {}'.format(accuracy,accuracy_adv))


if __name__=='__main__':
    main()