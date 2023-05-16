import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import numpy as np
import os
import math
from torch.autograd import Variable

class EncoderVisual(nn.Module):
    def __init__(self, args):
        super(EncoderVisual, self).__init__()
        self.a_feature_size = args.a_feature_size
        self.m_feature_size = args.m_feature_size
        self.hidden_size = args.hidden_size
        hidden_size = self.hidden_size

        # frame feature embedding
        self.frame_feature_embed = nn.Linear(args.a_feature_size, args.appearance_projected_size)
        nn.init.xavier_normal_(self.frame_feature_embed.weight)
        self.bi_lstm1 = nn.LSTM(args.appearance_projected_size, args.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_drop1 = nn.Dropout(p=0.3)

        # i3d feature embedding
        self.i3d_feature_embed = nn.Linear(args.m_feature_size, args.motion_projected_size)
        nn.init.xavier_normal_(self.i3d_feature_embed.weight)
        self.bi_lstm2 = nn.LSTM(args.motion_projected_size, args.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_drop2 = nn.Dropout(p=0.3)

        self.self_attention_a = SelfAttention(hidden_size*2, hidden_size*2, args.appearance_projected_size, args.dropout, True)
        self.self_attention_m = SelfAttention(hidden_size*2, hidden_size*2, args.motion_projected_size, args.dropout, True)
        self.layernorm_sa_a = nn.LayerNorm(args.appearance_projected_size)
        self.layernorm_sa_m = nn.LayerNorm(args.motion_projected_size)
        self.drop_sa = nn.Dropout(args.dropout)

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(2, batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(2, batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c

    def forward(self, inputs):
        frame_feats = inputs[:, :, :self.a_feature_size].contiguous()
        i3d_feats = inputs[:, :, -self.m_feature_size:].contiguous()
        # frame feature embedding
        embedded_frame_feats = self.frame_feature_embed(frame_feats)
        lstm_h1, lstm_c1 = self._init_lstm_state(frame_feats)
        # bidirectional lstm encoder
        frame_feats, _ = self.bi_lstm1(embedded_frame_feats, (lstm_h1, lstm_c1))
        frame_feats = self.lstm_drop1(frame_feats)
        # i3d feature embedding

        embedded_i3d_feats = self.i3d_feature_embed(i3d_feats)
        lstm_h2, lstm_c2 = self._init_lstm_state(i3d_feats)
        # bidirectional lstm encoder
        i3d_feats, _ = self.bi_lstm2(embedded_i3d_feats, (lstm_h2, lstm_c2))
        i3d_feats = self.lstm_drop2(i3d_feats)

        # self attention
        appear = self.self_attention_a(frame_feats)
        appear = self.layernorm_sa_a(appear)

        motion = self.self_attention_m(i3d_feats)
        motion = self.layernorm_sa_m(motion)

        return appear,motion


class SelfAttention(nn.Module):
    def __init__(self, input_size, attention_size, output_size, dropout=0.3, get_pe=False):
        super(SelfAttention, self).__init__()

        self.attention_size = attention_size
        self.dropout = dropout
        self.get_pe = get_pe
        self.pe = PositionalEncoding_old(attention_size)
        self.K = nn.Linear(in_features=input_size, out_features=self.attention_size, bias=False)
        self.Q = nn.Linear(in_features=input_size, out_features=self.attention_size, bias=False)
        self.V = nn.Linear(in_features=input_size, out_features=self.attention_size, bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.attention_size, out_features=output_size, bias=False),
            # nn.Tanh(),
            nn.Dropout(self.dropout)
        )

    def forward(self, x, att_mask=None):
        if self.get_pe:
            x = self.pe(x)
        K = self.K(x)
        Q = self.Q(x).transpose(-1, -2)
        V = self.V(x).transpose(-1, -2)
        logits = torch.div(torch.matmul(K, Q), torch.tensor(np.sqrt(self.attention_size)))
        if att_mask is not None:
            zero_vec = -9e15 * torch.ones_like(logits)
            logits = torch.where(att_mask > 0, logits, zero_vec)
            # logits = logits * att_mask
        weight = F.softmax(logits, dim=-1)
        weight = weight.transpose(-1, -2)
        mid_step = torch.matmul(V, weight)
        # mid_step = torch.matmul(V, weight)
        attention = mid_step.transpose(-1, -2)

        attention = self.output_layer(attention)

        return attention


class PositionalEncoding_old(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.2, max_len=72):
        super(PositionalEncoding_old, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
