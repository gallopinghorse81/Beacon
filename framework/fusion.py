# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    def __init__(self, opt):
        super(FusionLayer, self).__init__()
        if opt.self_att:
            self.attn = SelfAtt(opt.id_emb_size, opt.num_heads)
        self.opt = opt
        self.linear = nn.Linear(opt.feature_dim, opt.feature_dim)
        self.drop_out = nn.Dropout(0.5)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, u_out, i_out):
        if self.opt.self_att:
            out = self.attn(u_out, i_out)
            s_u_out, s_i_out = torch.split(out, out.size(1)//2, 1)
            u_out = u_out + s_u_out
            i_out = i_out + s_i_out
        if self.opt.r_id_merge == 'cat':
            u_out = u_out.view(u_out.size(0), -1)
            i_out = i_out.view(i_out.size(0), -1)
        else:
            u_out = u_out.sum(1)
            i_out = i_out.sum(1)

        if self.opt.ui_merge == 'cat':
            out = torch.cat([u_out, i_out], 1)
        elif self.opt.ui_merge == 'add':
            out = u_out + i_out
        elif self.opt.ui_merge == 'element dot':
            out = u_out * i_out
        elif self.opt.ui_merge == 'matmul':
            out = torch.matmul(u_out.unsqueeze(1) , i_out.unsqueeze(2)).squeeze(1)
        else:
            raise ValueError("ui_merge输入不符合规范")


        return out


class SelfAtt(nn.Module):
    def __init__(self, dim, num_heads):
        super(SelfAtt, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim, num_heads, 128, 0.4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)

    def forward(self, user_fea, item_fea):
        fea = torch.cat([user_fea, item_fea], 1).permute(1, 0, 2)
        out = self.encoder(fea)
        return out.permute(1, 0, 2)
