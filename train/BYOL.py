import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
from utils import *
import time
from models.lib import kNN
from torchvision import datasets
from torch.cuda.amp import autocast,GradScaler


logger = logging.getLogger(__name__)

class BYOL():
    def __init__(self, args, model, optimizer, scheduler):
        self.args = args
        self.model = model.to(self.args.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        if self.args.amp:
            self.scaler = GradScaler()

    @staticmethod
    def regression_loss(x, y):
        return 2 - 2 * (x * y).sum(dim=-1)

    def train(self, train_loader):
        logging.warning(f"Start MoCo training for {self.args.epochs} epochs.")
        transform_all = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset_kNN = datasets.CIFAR10(root="./data", transform=transform_all, train=True)
        test_dataset_kNN = datasets.CIFAR10(root="./data", transform=transform_all, train=False)
        train_loader_kNN = torch.utils.data.DataLoader(
            train_dataset_kNN, batch_size=100, shuffle=False)
        test_loader_kNN = torch.utils.data.DataLoader(
            test_dataset_kNN, batch_size=100, shuffle=False)

        pre_losses = 1e9
        for epoch_counter in range(self.args.epochs):
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            batches = len(train_loader)
            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, losses],
                prefix="Epoch: [{}]".format(epoch_counter))

            # switch to train mode
            self.model.train()

            end = time.time()
            i=1
            for (images, _) in train_loader:
                # measure data loading time
                data_time.update(time.time() - end)

                images[0] = images[0].cuda(non_blocking=True)
                images[1] = images[1].cuda(non_blocking=True)

                if not self.args.amp:
                    # compute output
                    q1, q2, k1, k2 = self.model(x1=images[0], x2=images[1])
                    loss1 = self.regression_loss(q1, k2)
                    loss2 = self.regression_loss(q2, k1)

                    loss = (loss1+loss2).mean()

                    losses.update(loss.item(), images[0].size(0))

                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()
                    # visualize_param_grad_scalar(self.args.writer, self.model.encoder_q, i+epoch_counter*batches)
                    self.optimizer.step()
                else:
                    with autocast():
                        q1, q2, k1, k2 = self.model(x1=images[0], x2=images[1])
                        loss1 = self.regression_loss(q1, k2)
                        loss2 = self.regression_loss(q2, k1)

                        loss = (loss1+loss2).mean()
                    losses.update(loss.item(), images[0].size(0))
                    self.optimizer.zero_grad()
                    # 缩放损失，反向传播不建议放到autocast下，它默认和前向采用相同的计算精度
                    self.scaler.scale(loss).backward()
                    # 先反缩放梯度，若反缩后梯度不是inf或者nan，则用于权重更新
                    self.scaler.step(self.optimizer)
                    # 更新缩放器
                    self.scaler.update()

                self.scheduler.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                progress.display(i)
                i+=1
            self.args.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=epoch_counter)

            self.args.writer.add_scalar('train/loss', losses.avg, global_step=epoch_counter)

            if self.args.kNN:
                acc = kNN(epoch_counter, self.model.encoder_q, train_loader_kNN, test_loader_kNN, feat_dim=864, K=7)
                self.args.writer.add_scalar('kNN_eval/top1', acc, global_step=epoch_counter)

            if pre_losses > losses.avg:
                pre_losses = losses.avg
                save_best = True
            else:
                save_best = False

            save_checkpoint(self.args,
                            {'epoch': self.args.epochs,
                             'model_state_dict': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'scheduler': self.scheduler.state_dict(), },
                            is_best=save_best)

        logging.warning("Training has finished.")
        # save model checkpoints

        logging.warning(f"Model checkpoint and metadata has been saved at {self.args.save_path}.")