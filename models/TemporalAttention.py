import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, feat_size, bottleneck_size, dropout):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.bottleneck_size = bottleneck_size
        self.dropout = dropout

        self.W = nn.Linear(self.hidden_size, self.bottleneck_size, bias=False)
        self.U = nn.Linear(self.feat_size, self.bottleneck_size, bias=False)
        self.b = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.w = nn.Linear(self.bottleneck_size, 1, bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.bottleneck_size, out_features=self.hidden_size, bias=False),
            nn.Dropout(self.dropout)
        )

    def forward(self, hidden, feats, masks=None):

        hidden = hidden.transpose(0,1).repeat(1,13,1)
        Wh = self.W(hidden)
        Uv = self.U(feats)

        energies = self.w(torch.tanh(Wh + Uv + self.b))
        if masks is not None:
            zero_vec = -9e15 * torch.ones_like(energies)
            energies = torch.where(masks>0, energies, zero_vec)

        weight = F.softmax(energies, dim=-1)
        weight = weight.transpose(-1,-2)
        feats = Uv.transpose(-1,-2)
        weighted_feats = torch.matmul(feats, weight)
        attention = weighted_feats.transpose(-1,-2)
        attention = self.output_layer(attention)

        return attention, weight

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(key_dim,requires_grad=False).float()
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)
        self.ln = nn.LayerNorm(num_units)

    def get_device(self):
        # return the device of the tensor, either "cpu"
        # or number specifiing the index of gpu.
        dev = next(self.parameters()).get_device()
        if dev == -1:
            return "cpu"
        return dev

    def forward(self, query, keys, mask=None):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)

        # attention and _key_dim should be in the same device.
        attention = attention / torch.sqrt(self._key_dim).to(self.get_device())

        if mask is not None:
          mask = mask.repeat(self._h,1,1)
          attention.masked_fill_(mask,-float('inf'))
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        # residual connection
        attention += query
        # apply batch normalization
        #attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)
        # apply layer normalization
        #attention = self.ln(attention)

        return attention

class MatrixAttn(nn.Module):
# self.mattn = MatrixAttn(args.hsz * 3, args.hsz)
    def __init__(self, linin, linout):
        super().__init__()
        self.attnlin = nn.Linear(linin, linout)

    def get_device(self):
        # return the device of the tensor, either "cpu"
        # or number specifiing the index of gpu.
        dev = next(self.parameters()).get_device()
        if dev == -1:
            return "cpu"
        return dev
# _, z = self.mattn(decoder_outputs, (object, object_3))
    def forward(self, dec, emb):
        if len(dec.shape) == 2:
            dec = dec.unsqueeze(1)
        emb, elen = emb
        # dev = emb.get_device()
        # emask and emb should be in the same device
        emask = torch.arange(0, emb.size(1)).unsqueeze(0).repeat(emb.size(0), 1).long().to(self.get_device())

        emask = (emask >= elen.unsqueeze(1)).unsqueeze(1)
        decsmall = self.attnlin(dec)
        unnorm = torch.bmm(decsmall, emb.transpose(1, 2))  #  torch.bmm就是矩阵相乘
        unnorm.masked_fill_(emask, -float('inf'))
        attn = F.softmax(unnorm, dim=2)
        out = torch.bmm(attn, emb)
        return out, attn