import torch
import torch.nn as nn
import utils
import torch.nn.functional as F


class ConvE(nn.Module):
    def __init__(self, h_dim, params):
        super().__init__()

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(params.out_channel)
        self.bn2 = torch.nn.BatchNorm1d(h_dim)

        self.conv_drop = torch.nn.Dropout(params.conv_drop)
        self.fc_drop = torch.nn.Dropout(params.fc_drop)
        self.k_h = params.k_h
        self.k_w = params.k_w
        assert self.k_h * self.k_w == h_dim
        self.conv = torch.nn.Conv2d(1, out_channels=params.out_channel, stride=1, padding=0,
                                    kernel_size=params.ker_sz, bias=False)
        flat_sz_h = int(2 * self.k_h) - params.ker_sz + 1
        flat_sz_w = self.k_w - params.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * params.out_channel
        self.fc = torch.nn.Linear(self.flat_sz, h_dim, bias=False)
        self.ent_drop = nn.Dropout(params.ent_drop_pred)

    def forward(self, head, rel, all_ent):
        # head (bs, h_dim), rel (bs, h_dim)
        # concatenate and reshape to 2D
        c_head = head.view(-1, 1, head.shape[-1])
        c_rel = rel.view(-1, 1, rel.shape[-1])
        # print(c_head.shape)
        # print(c_rel.shape)
        c_emb = torch.cat([c_head, c_rel], 1)
        c_emb = torch.transpose(c_emb, 2, 1).reshape((-1, 1, 2 * self.k_h, self.k_w))

        x = self.bn0(c_emb)
        x = self.conv(x)  # (bs, out_channels, out_h, out_w)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
        x = x.view(-1, self.flat_sz)  # (bs, out_channels * out_h * out_w)
        x = self.fc(x)  # (bs, h_dim)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc_drop(x)  # (bs, h_dim)
        # inference
        # all_ent: (n_ent, h_dim)
        all_ent = self.ent_drop(all_ent)
        x = torch.mm(x, all_ent.transpose(1, 0))  # (bs, n_ent)
        x = torch.sigmoid(x)
        return x
