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
from torch.cuda.amp import autocast, GradScaler
import sys

logger = logging.getLogger(__name__)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINO():
    def __init__(self, args, student, teacher, optimizer,  lr_schedule, wd_schedule, momentum_schedule):
        self.args = args
        self.student = student.to(self.args.device)
        self.teacher = teacher.to(self.args.device)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule
        self.momentum_schedule = momentum_schedule
        self.dino_loss = DINOLoss(args.out_dim,
                                  args.local_crops_number + 2, # total number of crops = 2 global crops + local_crops_number
                                  args.warmup_teacher_temp,
                                  args.teacher_temp,
                                  args.warmup_teacher_temp_epochs,
                                  args.epochs,
                                  args.student_temp,
                                  ).cuda()
        if self.args.amp:
            self.scaler = GradScaler()


    def train(self, train_loader):
        logging.warning(f"Start DINO training for {self.args.epochs} epochs.")
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
        it = 0
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
            self.student.train()
            self.teacher.train()

            end = time.time()
            i=1
            for (images, _) in train_loader:
                # measure data loading time
                data_time.update(time.time() - end)

                for ii, param_group in enumerate(self.optimizer.param_groups):
                    param_group["lr"] = self.lr_schedule[it]
                    if ii == 0:  # only the first group is regularized
                        param_group["weight_decay"] = self.wd_schedule[it]

                images = [im.cuda(non_blocking=True) for im in images]

                with torch.cuda.amp.autocast(self.args.amp is not None):
                    teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
                    student_output = self.student(images)
                    loss = self.dino_loss(student_output, teacher_output, epoch_counter)
                    losses.update(loss.item(), images[0].size(0))

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()))
                    sys.exit(1)

                # student update
                self.optimizer.zero_grad()
                param_norms = None
                if self.args.amp is None:
                    loss.backward()
                    if self.args.clip_grad:
                        param_norms = utils.clip_gradients(self.student, self.args.clip_grad)
                    utils.cancel_gradients_last_layer(epoch_counter, self.student,
                                                      self.args.freeze_last_layer)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.args.clip_grad:
                        self.scaler.unscale_(
                            self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
                        param_norms = utils.clip_gradients(self.student, self.args.clip_grad)
                    utils.cancel_gradients_last_layer(epoch_counter, self.student,
                                                      self.args.freeze_last_layer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()


                # EMA update for the teacher
                with torch.no_grad():
                    m = self.momentum_schedule[it]  # momentum parameter
                    for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                progress.display(i)
                it += 1
                i += 1

            self.args.writer.add_scalar('learning_rate', self.lr_schedule[it-1], global_step=epoch_counter)

            self.args.writer.add_scalar('train/loss', losses.avg, global_step=epoch_counter)

            if self.args.kNN:
                acc = kNN(epoch_counter, self.teacher.model.backbone, train_loader_kNN, test_loader_kNN, feat_dim=self.args.dim, K=7)
                self.args.writer.add_scalar('kNN_eval/top1', acc, global_step=epoch_counter)

            if pre_losses > losses.avg:
                pre_losses = losses.avg
                save_best = True
            else:
                save_best = False

            save_checkpoint(self.args,
                            {'epoch': self.args.epochs,
                             'model_state_dict': self.student.state_dict(),
                             'optimizer': self.optimizer.state_dict(), },
                            is_best=save_best)

        logging.warning("Training has finished.")
        # save model checkpoints

        logging.warning(f"Model checkpoint and metadata has been saved at {self.args.save_path}.")
