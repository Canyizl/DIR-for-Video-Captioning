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

        self.ae = AE(feature_size,latent_size,output_size)
        self.moco = MoCo()

    def forward(self,logits,rels,rel_dict):
        rels_scores, rels_indices = logits.sort(descending=True, dim=1)  # inds_size: B,N 100,122
        _, rels_dis = logits.sort(descending=False, dim=1)
        topk = 2
        topk_rels_indices = rels_indices[:, :topk]
        disk = 20
        disk_rels_indices = rels_dis[:, :disk]
        loss_vae = torch.tensor([0.]).cuda()
        loss_csl = torch.tensor([0.]).cuda()
        z_rel = torch.zeros([1, 512]).cuda()
        for topk_rel_indices, disk_rel_indices in zip(topk_rels_indices, disk_rels_indices):
            pos_dict1 = (rel_dict[rels[topk_rel_indices[0]][1]]).unsqueeze(0)
            pos_dict2 = (rel_dict[rels[topk_rel_indices[1]][1]]).unsqueeze(0)
            #dot_rel_dict1 = torch.mul(pos_dict1, pos_dict2)
            #dot_rel_dict2 = torch.mul(dot_rel_dict1, pos_dict3)
            dot_rel_dict1 = torch.cat((pos_dict1, pos_dict2),dim=1)
            batch_pos_rels_vec = dot_rel_dict1.detach()

            neg_dict1 = rel_dict[rels[disk_rel_indices[0]][1]].unsqueeze(0)
            for i in range(1, disk, 1):
                neg_dict_i = rel_dict[rels[disk_rel_indices[i]][1]].unsqueeze(0)
                neg_dict1 = torch.cat((neg_dict1, neg_dict_i), 0)
            # print(neg_dict1.size())  10 * 300
            batch_neg_rels_vec = neg_dict1

            loss_vae_batch,z_ = self.ae(batch_pos_rels_vec)
            loss_vae += loss_vae_batch
            pos_dict1 = pos_dict1.unsqueeze(0)
            loss_csl_batch, z = self.moco(z_, pos_dict1, batch_neg_rels_vec)
            loss_csl += loss_csl_batch
            z_rel = torch.cat((z_rel, z), 0).cuda()

        return loss_vae, loss_csl, z_rel

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

        self.encoder_inp = nn.Sequential(nn.Linear(300, 512),nn.ReLU(),nn.Linear(512,512))
        self.encoder_z = nn.Sequential(nn.Linear(512,512),nn.ReLU(),nn.Linear(512,512))
        self.encoder_k = nn.Sequential(nn.Linear(300, 512),nn.ReLU(),nn.Linear(512,512))
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