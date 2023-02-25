# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch, math


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class AttFlat(nn.Module):
    def __init__(self, num_hid, FLAT_MLP_SIZE=512, FLAT_GLIMPSES=1, DROPOUT_R=0.1, FLAT_OUT_SIZE=512):
        super(AttFlat, self).__init__()
        self.FLAT_GLIMPSES = FLAT_GLIMPSES

        self.mlp = MLP(
            in_size=num_hid,
            mid_size=FLAT_MLP_SIZE,
            out_size=FLAT_GLIMPSES,
            dropout_r=DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            num_hid * FLAT_GLIMPSES,
            FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, num_hid, head_num=8, drop_rate=0.1):
        super(MHAtt, self).__init__()
        self.head_num = head_num
        self.num_hid = num_hid

        self.linear_v = nn.Linear(num_hid, num_hid)
        self.linear_k = nn.Linear(num_hid, num_hid)
        self.linear_q = nn.Linear(num_hid, num_hid)
        self.linear_merge = nn.Linear(num_hid, num_hid)

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.head_num,
            int(self.num_hid / self.head_num)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.head_num,
            int(self.num_hid / self.head_num)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.head_num,
            int(self.num_hid / self.head_num)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.num_hid
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, num_hid, FF_SIZE=2048, DROPOUT_R=0.1):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=num_hid,
            mid_size= FF_SIZE,
            out_size=num_hid,
            dropout_r= DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, num_hid, head_num=8, FF_SIZE=2048, DROPOUT_R=0.1):
        super(SA, self).__init__()

        self.mhatt = MHAtt(num_hid, head_num, DROPOUT_R)
        self.ffn = FFN(num_hid, FF_SIZE, DROPOUT_R)

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(num_hid)

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(num_hid)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, num_hid, head_num=8, FF_SIZE=2048, DROPOUT_R=0.1):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(num_hid, head_num, DROPOUT_R)
        self.mhatt2 = MHAtt(num_hid, head_num, DROPOUT_R)
        self.ffn = FFN(num_hid, FF_SIZE, DROPOUT_R)

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(num_hid)

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(num_hid)

        self.dropout3 = nn.Dropout(DROPOUT_R)
        self.norm3 = LayerNorm(num_hid)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, num_hid, head_num=8, FF_SIZE=2048, DROPOUT_R=0.1, layers=6):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(num_hid, head_num, FF_SIZE, DROPOUT_R) for _ in range(layers)])
        self.dec_list = nn.ModuleList([SGA(num_hid, head_num, FF_SIZE, DROPOUT_R) for _ in range(layers)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y
