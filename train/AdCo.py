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

class AdCo():
    def __init__(self, args, model, memory_bank, optimizer, scheduler):
        self.args = args
        self.model = model.to(self.args.device)
        self.memory_bank = memory_bank.to(self.args.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        if self.args.amp:
            self.scaler = GradScaler()

    def ctr(self, logits, labels):
        loss = self.criterion(logits, labels)
        acc1 = accuracy(logits, labels, topk=(1,))
        return loss, acc1


    def train(self, train_loader):
        logging.warning(f"Start AdCo training for {self.args.epochs} epochs.")
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
            top1 = AverageMeter('Acc@1', ':6.2f')
            mem_losses = AverageMeter('MemLoss', ':.4e')
            batches = len(train_loader)
            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, losses, mem_losses, top1],
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
                    q_pred, k_pred, q, k = self.model(im_q=images[0], im_k=images[1])
                    l_pos1 = torch.einsum('nc,ck->nk', [q_pred, k.T])
                    l_pos2 = torch.einsum('nc,ck->nk', [k_pred, q.T])

                    d_norm1, d1, l_neg1 = self.memory_bank(q_pred)
                    d_norm2, d2, l_neg2 = self.memory_bank(k_pred)
                    # logits: Nx(N+K)

                    logits1 = torch.cat([l_pos1, l_neg1], dim=1)
                    logits1 /= self.args.temperature
                    logits2 = torch.cat([l_pos2, l_neg2], dim=1)
                    logits2 /= self.args.temperature

                    cur_batch_size = logits1.shape[0]
                    labels = torch.arange(cur_batch_size, dtype=torch.long).cuda()
                    loss1, top11 = self.ctr(logits1, labels)
                    loss2, top12 = self.ctr(logits2, labels)

                    loss = (loss1 + loss2) / 2
                    top1_acc = (top11[0] + top12[0]) / 2

                    losses.update(loss.item(), images[0].size(0))
                    top1.update(top1_acc.item(), images[0].size(0))

                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()
                    # visualize_param_grad_scalar(self.args.writer, self.model.encoder_q, i+epoch_counter*batches)
                    self.optimizer.step()

                else:
                    with autocast():
                        q_pred, k_pred, q, k = self.model(im_q=images[0], im_k=images[1])
                        l_pos1 = torch.einsum('nc,ck->nk', [q_pred, k.T])
                        l_pos2 = torch.einsum('nc,ck->nk', [k_pred, q.T])

                        d_norm1, d1, l_neg1 = self.memory_bank(q_pred)
                        d_norm2, d2, l_neg2 = self.memory_bank(k_pred)
                        # logits: Nx(N+K)

                        logits1 = torch.cat([l_pos1, l_neg1], dim=1)
                        logits1 /= self.args.temperature
                        logits2 = torch.cat([l_pos2, l_neg2], dim=1)
                        logits2 /= self.args.temperature

                        cur_batch_size = logits1.shape[0]
                        labels = torch.arange(cur_batch_size, dtype=torch.long).cuda()
                        loss1, top11 = self.ctr(logits1, labels)
                        loss2, top12 = self.ctr(logits2, labels)

                        loss = (loss1 + loss2) / 2
                        top1_acc = (top11[0] + top12[0]) / 2
                    losses.update(loss.item(), images[0].size(0))
                    top1.update(top1_acc.item(), images[0].size(0))
                    self.optimizer.zero_grad()
                    # 缩放损失，反向传播不建议放到autocast下，它默认和前向采用相同的计算精度
                    self.scaler.scale(loss).backward()
                    # 先反缩放梯度，若反缩后梯度不是inf或者nan，则用于权重更新
                    self.scaler.step(self.optimizer)
                    # 更新缩放器
                    self.scaler.update()

                self.scheduler.step()

                # update memory bank
                with torch.no_grad():
                    with autocast():
                    # update memory bank

                        total_bsize = logits1.shape[1]- self.args.bank_size
                        p_qd1 = nn.functional.softmax(logits1, dim=1)[:, total_bsize:]
                        g1 = torch.einsum('cn,nk->ck', [q_pred.T, p_qd1]) / logits1.shape[0] - torch.mul(
                            torch.mean(torch.mul(p_qd1, l_neg1), dim=0), d_norm1)
                        p_qd2 = nn.functional.softmax(logits2, dim=1)[:, total_bsize:]
                        g2 = torch.einsum('cn,nk->ck', [k_pred.T, p_qd2]) / logits2.shape[0] - torch.mul(
                            torch.mean(torch.mul(p_qd2, l_neg2), dim=0), d_norm1)
                        g = -0.5 * torch.div(g1, torch.norm(d1, dim=0)) / self.args.mem_t - 0.5 * torch.div(g2, torch.norm(d1, dim=0)) / self.args.mem_t  # c*k
                        self.memory_bank.v.data = self.args.momentum * self.memory_bank.v.data + g + self.args.mem_wd * self.memory_bank.W.data
                        self.memory_bank.W.data = self.memory_bank.W.data - self.args.memory_lr * self.memory_bank.v.data
                        logits1 = torch.softmax(logits1, dim=1)
                        batch_prob1 = torch.sum(logits1[:, :logits1.size(0)], dim=1)
                        logits2 = torch.softmax(logits2, dim=1)
                        batch_prob2 = torch.sum(logits2[:, :logits2.size(0)], dim=1)
                        batch_prob = 0.5 * torch.mean(batch_prob1) + 0.5 * torch.mean(batch_prob2)
                        mem_losses.update(batch_prob.item(), logits1.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                progress.display(i)
                i+=1
            self.args.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=epoch_counter)

            self.args.writer.add_scalar('train/loss', losses.avg, global_step=epoch_counter)
            self.args.writer.add_scalar('train/mem_losses', mem_losses.avg, global_step=epoch_counter)
            self.args.writer.add_scalar('train/top1', top1.avg, global_step=epoch_counter)

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