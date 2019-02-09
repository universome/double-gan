import os

from torchvision.datasets import MNIST
from firelab import BaseTrainer
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize, Compose

from src.models import RevNet, ConvClassifier
from src.losses.wgan_losses import compute_w_dist, compute_gp


class DoubleGANTrainer(BaseTrainer):
    def __init__(self, config):
        super(DoubleGANTrainer, self).__init__(config)

    def init_dataloaders(self):
        data_path = os.path.join(self.config.firelab.project_path, self.config.data.path)
        transformation = Compose([Resize((32, 32)), ToTensor()])

        ds_train = MNIST(data_path, train=True, transform=transformation, download=True)
        ds_val = MNIST(data_path, train=False, transform=transformation, download=True)

        train_x = np.array([x.numpy() for x, _ in ds_train])
        train_y = np.array([1 - x for x in train_x])

        val_x = np.array([x.numpy() for x, _ in ds_val])
        val_y = np.array([1 - x for x in val_x])

        self.train_dataloader = DataLoader(
            [(x, y) for x, y in zip(train_x, train_y)], batch_size=self.config.hp.batch_size)

        self.val_dataloader = DataLoader(
            [(x, y) for x, y in zip(val_x, val_y)], batch_size=2)

    def init_models(self):
        self.gen = RevNet(32, self.config.hp.gen_layers).to(self.config.device_name)
        self.discr_x = ConvClassifier(1).to(self.config.device_name)
        self.discr_y = ConvClassifier(1).to(self.config.device_name)

    def init_criterions(self):
        pass

    def init_optimizers(self):
        self.gen_optim = Adam(self.gen.parameters(), self.config.hp.gen_lr)
        self.discr_x_optim = Adam(self.discr_x.parameters(), self.config.hp.discr_lr)
        self.discr_y_optim = Adam(self.discr_y.parameters(), self.config.hp.discr_lr)

    def train_on_batch(self, batch):
        batch_x = batch[0].to(self.config.device_name)
        batch_y = batch[1].to(self.config.device_name)

        fake_data_y = self.gen.forward(batch_x)
        fake_data_x = self.gen.reverse_forward(batch_y)

        fake_data_y_scores = self.discr_y(fake_data_y)
        fake_data_x_scores = self.discr_x(fake_data_x)

        real_data_y_scores = self.discr_y(batch_y)
        real_data_x_scores = self.discr_x(batch_x)

        w_dist_y = compute_w_dist(fake_data_y_scores, real_data_y_scores)
        w_dist_x = compute_w_dist(fake_data_x_scores, real_data_x_scores)

        gp_y = compute_gp(self.discr_y, fake_data_y, batch_y)
        gp_x = compute_gp(self.discr_x, fake_data_x, batch_x)

        self.gen_optim.zero_grad()
        gen_loss_x = fake_data_x_scores.mean()
        gen_loss_y = fake_data_y_scores.mean()
        #gen_loss = gen_loss_x + gen_loss_y
        gen_loss = gen_loss_y
        gen_loss.backward(retain_graph=True)
        self.gen_optim.step()

        # discr_x_optim.zero_grad()
        # discr_x_loss = w_dist_x + gp_lambda * gp_x
        # discr_x_loss.backward()
        # discr_x_optim.step()

        self.discr_y_optim.zero_grad()
        discr_y_loss = w_dist_y + self.config.hp.gp_lambda * gp_y
        discr_y_loss.backward()
        self.discr_y_optim.step()

        self.writer.add_scalar('TRAIN/GEN_LOSS/X', gen_loss_x.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/GEN_LOSS/Y', gen_loss_y.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/W_DIST/X', w_dist_x.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/W_DIST/Y', w_dist_y.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/GRAD_PENALTY/X', gp_x.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/GRAD_PENALTY/Y', gp_y.item(), self.num_iters_done)

    def validate(self):
        batch = next(iter(self.val_dataloader))
        batch_x = batch[0].to(self.config.device_name)
        batch_y = batch[1].to(self.config.device_name)

        img_real_x = batch_x[0].cpu().numpy()
        img_real_y = batch_y[0].cpu().numpy()
        img_fake_y = self.gen(batch_x)[0].detach().cpu().numpy()
        img_fake_x = self.gen.reverse_forward(batch_y)[0].detach().cpu().numpy()

        self.writer.add_image('REAL/X', img_real_x, self.num_iters_done)
        self.writer.add_image('REAL/Y', img_real_y, self.num_iters_done)
        self.writer.add_image('FAKE/X', img_fake_x, self.num_iters_done)
        self.writer.add_image('FAKE/Y', img_fake_y, self.num_iters_done)
