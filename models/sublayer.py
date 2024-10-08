import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

class AttentionShare(nn.Module):
    def __init__(self, input_value_size, input_key_size, output_size, dropout=0.1):
        super(AttentionShare, self).__init__()
        self.input_value_size = input_value_size
        self.input_key_size = input_key_size
        self.attention_size = output_size
        self.dropout = dropout

        self.K = nn.Linear(in_features=input_value_size, out_features=output_size, bias=False)
        self.Q = nn.Linear(in_features=input_key_size, out_features=output_size, bias=False)
        self.V = nn.Linear(in_features=input_value_size, out_features=output_size, bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.attention_size, out_features=output_size, bias=False),
            nn.Tanh(),
            nn.LayerNorm(output_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, meta_state, hidden_previous):
        K = self.K(meta_state)
        Q = self.Q(hidden_previous).unsqueeze(2)
        V = self.V(meta_state).transpose(-1, -2)

        logits = torch.div(torch.matmul(K, Q), torch.tensor(np.sqrt(self.attention_size)))
        weight = F.softmax(logits, dim=1)
        # weight = F.sigmoid(logits)
        mid_step = torch.matmul(V, weight)
        # mid_step = torch.matmul(V, weight)

        attention = mid_step.squeeze(2)

        attention = self.output_layer(attention)

        return attention, weight

# self.att = SelfAttention(512, 512, 512, 0.3)
class SelfAttention(nn.Module):
    def __init__(self, input_size, attention_size, output_size, dropout=0.2, get_pe=False):
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

# self.pe = PositionalEncoding_old(attention_size) attention_size=512
class PositionalEncoding_old(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.2, max_len=72):
        super(PositionalEncoding_old, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        # position = ([[0.],[1.],...[71.]])   size=[72,1]
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

