import os

from firelab import BaseTrainer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize, Compose

from src.models import RevNet, ConvDiscriminator
from src.losses.wgan_losses import compute_w_dist, compute_gp


class DoubleGANTrainer(BaseTrainer):
    def __init__(self, config):
        super(DoubleGANTrainer, self).__init__(config)

    def init_dataloaders(self):
        data_path = os.path.join(self.config.firelab.project_path, self.config.data.path)
        transformation = Compose([Resize((32, 32)), ToTensor()])

        ds_train = MNIST(data_path, train=True, transform=transformation, download=True)
        ds_val = MNIST(data_path, train=False, transform=transformation, download=True)

        train_x = np.array([x.numpy() for x, label in ds_train if label == 7])
        train_y = np.array([1 - x for x in train_x])

        val_x = np.array([x.numpy() for x, label in ds_val if label == 7])
        val_y = np.array([1 - x for x in val_x])

        train_x = rescale_img(train_x)
        train_y = rescale_img(train_y)
        val_x = rescale_img(val_x)
        val_y = rescale_img(val_y)

        self.train_dataloader = DataLoader(
            [(x, y) for x, y in zip(train_x, train_y)], batch_size=self.config.hp.batch_size)

        self.val_dataloader = DataLoader(
            [(x, y) for x, y in zip(val_x, val_y)], batch_size=2)

    def init_models(self):
        self.gen = RevNet(32, self.config.hp.gen_layers).to(self.config.device_name)
        # self.gen = SimplestGenerator().to(self.config.device_name)
        # self.gen = Generator().to(self.config.device_name)
        self.discr_x = ConvDiscriminator(1).to(self.config.device_name)
        self.discr_y = ConvDiscriminator(1).to(self.config.device_name)

    def init_criterions(self):
        pass

    def init_optimizers(self):
        self.gen_optim = Adam(self.gen.parameters(), self.config.hp.gen_lr)
        self.discr_x_optim = Adam(self.discr_x.parameters(), self.config.hp.discr_lr)
        self.discr_y_optim = Adam(self.discr_y.parameters(), self.config.hp.discr_lr)

    def train_on_batch(self, batch):
        batch_x = batch[0].to(self.config.device_name)
        batch_y = batch[1].to(self.config.device_name)

        fake_data_y = self.gen.forward(batch_x).tanh()
        fake_data_x = self.gen.reverse_forward(batch_y).tanh()

        # z = torch.randn(batch_x.size(0), 100).to(self.config.device_name).view(-1, 100, 1, 1)
        # fake_data_y = self.gen.forward(z)
        # fake_data_x = self.gen.forward(z)

        fake_data_y_scores = self.discr_y(fake_data_y)
        fake_data_x_scores = self.discr_x(fake_data_x)

        real_data_y_scores = self.discr_y(batch_y)
        real_data_x_scores = self.discr_x(batch_x)

        w_dist_y = compute_w_dist(fake_data_y_scores, real_data_y_scores)
        w_dist_x = compute_w_dist(fake_data_x_scores, real_data_x_scores)
        gp_y = compute_gp(self.discr_y, fake_data_y, batch_y)
        gp_x = compute_gp(self.discr_x, fake_data_x, batch_x)

        gen_loss_x = fake_data_x_scores.mean()
        gen_loss_y = fake_data_y_scores.mean()
        gen_loss = gen_loss_x + gen_loss_y
        # gen_loss = gen_loss_y

        discr_x_loss = w_dist_x + self.config.hp.gp_lambda * gp_x
        discr_y_loss = w_dist_y + self.config.hp.gp_lambda * gp_y

        # Training discriminator on more amount of steps
        if self.num_iters_done % 5 == 0:
            self.gen_optim.zero_grad()
            gen_loss.backward(retain_graph=True)
            self.gen_optim.step()


        self.discr_x_optim.zero_grad()
        discr_x_loss.backward()
        self.discr_x_optim.step()

        self.discr_y_optim.zero_grad()
        discr_y_loss.backward()
        self.discr_y_optim.step()

        self.writer.add_scalar('TRAIN/GEN_LOSS/X', gen_loss_x.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/GEN_LOSS/Y', gen_loss_y.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/W_DIST/X', w_dist_x.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/W_DIST/Y', w_dist_y.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/GRAD_PENALTY/X', gp_x.item(), self.num_iters_done)
        self.writer.add_scalar('TRAIN/GRAD_PENALTY/Y', gp_y.item(), self.num_iters_done)

        # Some stats
        self.writer.add_histogram('TRAIN/FAKE_DATA_HIST/X', fake_data_x.clone().detach().cpu().numpy(), self.num_iters_done)
        self.writer.add_histogram('TRAIN/FAKE_DATA_HIST/Y', fake_data_y.clone().detach().cpu().numpy(), self.num_iters_done)

    def validate(self):
        batch = next(iter(self.val_dataloader))
        batch_x = batch[0].to(self.config.device_name)
        batch_y = batch[1].to(self.config.device_name)

        img_real_x = batch_x[0].cpu().numpy()
        img_real_y = batch_y[0].cpu().numpy()
        # z = torch.randn(batch_x.size(0), 100).to(self.config.device_name).view(-1, 100, 1, 1)
        # img_fake_y = self.gen(z)[0].detach().cpu().numpy()
        img_fake_y = self.gen(batch_x)[0].detach().cpu().tanh().numpy()
        img_fake_x = self.gen.reverse_forward(batch_y)[0].detach().cpu().tanh().numpy()

        img_real_x = rescale_img_back(img_real_x)
        img_real_y = rescale_img_back(img_real_y)
        img_fake_x = rescale_img_back(img_fake_x)
        img_fake_y = rescale_img_back(img_fake_y)

        self.writer.add_image('REAL/X', img_real_x, self.num_iters_done)
        self.writer.add_image('REAL/Y', img_real_y, self.num_iters_done)
        self.writer.add_image('FAKE/X', img_fake_x, self.num_iters_done)
        self.writer.add_image('FAKE/Y', img_fake_y, self.num_iters_done)


class SimplestGenerator(nn.Module):
    def __init__(self):
        super(SimplestGenerator, self).__init__()

        self.mult = nn.Parameter(torch.rand(1, 32, 32) / 100)
        self.bias = nn.Parameter(torch.rand(1, 32, 32) / 100)

    def forward(self, x):
        return self.mult.exp() * x + self.bias

    def reverse_forward(self, y):
        return (y - self.bias) / self.mult.exp()


class Generator(nn.Module):
    def __init__(self, d=100):
        super(Generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        # self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(d)
        # self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(d*2, 1, 4, 2, 1)

    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = self.deconv5(x).tanh()
        x = self.deconv4(x).tanh()

        return x


def arctanh(x):
    return 0.5 * torch.log((1+x)/(1-x))


def rescale_img(x):
    return (x - 0.5) * 2


def rescale_img_back(y):
    return y / 2 + 0.5
