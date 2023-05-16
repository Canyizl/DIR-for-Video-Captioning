import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.transforms as transforms

import random
import itertools
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import itertools
from torch.autograd import Variable

import numpy as np
import os
import copy

class MLP(nn.Module):
    def __init__(self,in_size,hidden_size,out_size):
        super(MLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU()
        )
        self.l2 = nn.Linear(hidden_size,out_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self,x):
        x = torch.mean(x,dim=1)
        x = self.l1(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_size, attention_size, output_size):
        super(SelfAttention, self).__init__()

        self.attention_size = attention_size
        self.K = nn.Linear(in_features=input_size, out_features=self.attention_size, bias=False)
        self.Q = nn.Linear(in_features=input_size, out_features=self.attention_size, bias=False)
        self.V = nn.Linear(in_features=input_size, out_features=self.attention_size, bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.attention_size, out_features=output_size, bias=False),
            # nn.Tanh(),
        )

    def forward(self, x):
        K = self.K(x)
        Q = self.Q(x).transpose(-1, -2)
        V = self.V(x).transpose(-1, -2)
        logits = torch.div(torch.matmul(K, Q), torch.tensor(np.sqrt(self.attention_size)))
        weight = F.softmax(logits, dim=-1)
        weight = weight.transpose(-1, -2)
        mid_step = torch.matmul(V, weight)
        attention = mid_step.transpose(-1, -2)

        attention = self.output_layer(attention)

        return attention

class Reason(nn.Module):
    def __init__(self,feature_size,latent_size,output_size):
        super(Reason, self).__init__()

        self.ae = CVAE(feature_size,latent_size,output_size)
        self.moco = MoCo()

    def forward(self,logits,rels,rel_dict):
        rels_scores, rels_indices = logits.sort(descending=True, dim=1)  # inds_size: B,N 100,122
        _, rels_dis = logits.sort(descending=False, dim=1)
        topk = 2
        topk_rels_indices = rels_indices[:, :topk]
        disk = 15
        disk_rels_indices = rels_dis[:, :disk]
        loss_vae = torch.tensor([0.]).cuda()
        loss_csl = torch.tensor([0.]).cuda()
        z_rel = torch.zeros([1, 512]).cuda()
        for topk_rel_indices, disk_rel_indices in zip(topk_rels_indices, disk_rels_indices):
            pos_dict1 = (rel_dict[rels[topk_rel_indices[0]][1]]).unsqueeze(0)
            pos_dict2 = (rel_dict[rels[topk_rel_indices[1]][1]]).unsqueeze(0)
            pos_dict1 = pos_dict1.detach()
            pos_dict2 = pos_dict2.detach()
            neg_dict1 = rel_dict[rels[disk_rel_indices[0]][1]].unsqueeze(0)
            for i in range(1, disk, 1):
                neg_dict_i = rel_dict[rels[disk_rel_indices[i]][1]].unsqueeze(0)
                neg_dict1 = torch.cat((neg_dict1, neg_dict_i), 0)
            # print(neg_dict1.size())  10 * 300
            batch_neg_rels_vec = neg_dict1
            recon,mu, log_std = self.ae(pos_dict1,pos_dict2)
            z_ = self.ae.reparametrize(mu,log_std)
            loss_vae_batch = self.ae.loss_function(recon,pos_dict1,mu,log_std)

            loss_vae += loss_vae_batch
            pos_dict1 = pos_dict1.unsqueeze(0)
            loss_csl_batch, z = self.moco(z_, pos_dict1, batch_neg_rels_vec)
            loss_csl += loss_csl_batch
            z_rel = torch.cat((z_rel, z), 0).cuda()

        return loss_vae, loss_csl, z_rel

class CVAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""
    def __init__(self, feature_size, class_size, latent_size):
        super(CVAE, self).__init__()

        self.fc1 = nn.Linear(feature_size + class_size, 512)
        self.fc2_mu = nn.Linear(512, latent_size)
        self.fc2_log_std = nn.Linear(512,latent_size)
        self.fc3 = nn.Linear(latent_size + class_size, 512)
        self.fc4 = nn.Linear(512, feature_size)
        self.relu = nn.LeakyReLU(0.2)
        self.Tanh = nn.Tanh()

    def encode(self, x, y):
        h1 = self.relu(self.fc1(torch.cat([x, y], dim=1)))  # concat features and labels
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z, y):
        h3 = self.relu(self.fc3(torch.cat([z, y], dim=1)))  # concat latents and labels
        recon = self.Tanh(self.fc4(h3))  # use sigmoid because the input image's pixel is between 0-1
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, y):
        mu, log_std = self.encode(x, y)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z, y)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss

class AE(nn.Module):
    def __init__(self, feature_size, latent_size, output_size):
        super(AE, self).__init__()

        self.loss = nn.MSELoss()
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, latent_size),
            nn.Tanh(),
            nn.Linear(latent_size, output_size),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, latent_size),
            nn.Tanh(),
            nn.Linear(latent_size, feature_size),
            nn.Sigmoid()
        )

    def forward(self,x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        loss = self.loss(x,x_)

        return loss,z

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, m=0.999, T=0.07):
        super(MoCo, self).__init__()

        self.m = m
        self.T = T

        self.encoder_inp = nn.Sequential(nn.Linear(300, 512),nn.ReLU(),nn.Linear(512,512),nn.ReLU())
        self.encoder_z = nn.Sequential(nn.Linear(512,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU())
        self.encoder_k = nn.Sequential(nn.Linear(300, 512),nn.ReLU(),nn.Linear(512,512),nn.ReLU())
        self.loss = nn.CrossEntropyLoss()

        for param_q, param_k in zip(self.encoder_inp.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_inp.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, z_,pos1,neg_batch):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        z_p = self.encoder_z(z_)
        pos1 = pos1.detach()
        neg_batch = neg_batch.detach()
        batch_pos_vec = self.encoder_inp(pos1)

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_encoder()  # update the key encoder
            neg_p = self.encoder_k(neg_batch)

        pos = torch.bmm(z_p.view(z_p.size(0), 1, -1),
                        batch_pos_vec.view(batch_pos_vec.size(0), -1, 1)).squeeze(-1)  # 1,1
        neg = torch.mm(z_p, neg_p.transpose(1, 0))  # 1,20
        # logits: Nx(1+K)
        logits = torch.cat([pos, neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.loss(logits,labels)

        return loss,z_p