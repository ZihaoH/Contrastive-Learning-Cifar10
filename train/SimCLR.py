import torch
import torch.nn.functional as F
import logging
import time
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import *
from models.lib import kNN
from torch.cuda.amp import autocast,GradScaler

logger = logging.getLogger(__name__)

class SimCLR(object):
    def __init__(self, args, model, optimizer, scheduler):
        self.args = args
        self.model = model.to(self.args.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        if self.args.amp:
            self.scaler = GradScaler()

    def info_nce_loss(self, features):
        batch_size = features.shape[0]
        labels = torch.cat([torch.arange(batch_size//self.args.n_views) for _ in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        logging.warning(f"Start SimCLR training for {self.args.epochs} epochs.")
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
            self.model.train()
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            losses = AverageMeter('Loss', ':.4e')
            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, losses, top1],
                prefix="Epoch: [{}]".format(epoch_counter))
            end = time.time()
            i=1
            for images, _ in train_loader:
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)
                data_time.update(time.time() - end)

                if not self.args.amp:
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                    losses.update(loss.item(), images[0].size(0))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    with autocast():
                        features = self.model(images)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)
                    losses.update(loss.item(), images[0].size(0))
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                self.scheduler.step()

                batch_time.update(time.time() - end)
                end = time.time()

                progress.display(i)
                i+=1

                train_top1 = accuracy(logits, labels, topk=(1,))
                top1.update(train_top1[0].item())

            self.args.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=epoch_counter)

            self.args.writer.add_scalar('train/loss', losses.avg, global_step=epoch_counter)
            self.args.writer.add_scalar('train/top1', top1.avg, global_step=epoch_counter)

            if self.args.kNN:
                acc = kNN(epoch_counter, self.model.backbone, train_loader_kNN, test_loader_kNN, feat_dim=864, K=7)
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
