import os

from firelab import BaseTrainer
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from src.models import RevNet


class iRevNetTrainer(BaseTrainer):
    def __init__(self, config):
        super(iRevNetTrainer, self).__init__(config)

    def init_dataloaders(self):
        data_path = os.path.join(self.config.firelab.project_path, self.config.data.path)

        ds_train = CIFAR10(data_path, train=True, transform=ToTensor(), download=True)
        ds_val = CIFAR10(data_path, train=False, transform=ToTensor(), download=True)

        self.train_dataloader = DataLoader(ds_train, batch_size=self.config.hp.batch_size)
        self.val_dataloader = DataLoader(ds_val, batch_size=self.config.hp.batch_size)

    def init_models(self):
        self.classifier = RevNetClassifier(self.config.hp.layers).to(self.config.device_name)

    def init_criterions(self):
        self.ce_loss = nn.CrossEntropyLoss()

    def init_optimizers(self):
        self.optim = Adam(self.classifier.parameters(), self.config.hp.lr)

    def train_on_batch(self, batch):
        data, labels = self.unpack_batch(batch)

        logits = self.classifier(data)
        loss = self.ce_loss(logits, labels)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        acc = compute_acc_from_logits(logits, labels)

        self.writer.add_scalar('TRAIN/loss', loss.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/acc', acc.item(), self.num_iters_done)

    def validate(self):
        for batch in self.val_dataloader:
            data, labels = self.unpack_batch(batch)
            logits = self.classifier(data)
            loss = self.ce_loss(logits, labels)
            acc = compute_acc_from_logits(logits, labels)

            self.writer.add_scalar('VAL/loss', loss.item(), self.num_iters_done)
            self.writer.add_scalar('VAL/acc', acc.item(), self.num_iters_done)

    def unpack_batch(self, batch):
        data = batch[0].to(self.config.device_name)
        labels = batch[1].to(self.config.device_name)

        return data, labels

class RevNetClassifier(nn.Module):
    def __init__(self, layers):
        super(RevNetClassifier, self).__init__()

        self.revnet = RevNet(32, layers)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(3072, 10)

    def forward(self, x):
        out = self.revnet(x)
        out = out.view(out.size(0), -1)
        out = self.relu(out)
        out = self.head(out)

        return out


def compute_acc_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    acc = (preds == targets.long()).float().mean()

    return acc
