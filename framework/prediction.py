# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionLayer(nn.Module):
    def __init__(self, opt):
        super(PredictionLayer, self).__init__()
        self.output = opt.output
        if opt.output == "fm_normal":
            self.model = FM_NORMAL(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "fm_normal_for_mpcn":
            self.model = FM_NORMAL_FOR_MPCN(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "lfm":
            self.model = LFM(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "lfm_standard":
            self.model = LFM_STANDARD(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "lfm_plus":
            self.model = LFM_PLUS(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "lfm_for_DSRLN":
            self.model = LFM_FOR_DSRLN(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == 'mlp':
            self.model = MLP(opt.feature_dim)
        elif opt.output == 'nfm':
            self.model = NFM(opt.feature_dim)
        elif opt.output == 'pbfl':
            self.model = PBFL(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == 'lfm_plus_mlp':
            self.model = Lfm_Plus_Mlp(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == 'sum':
            self.model = torch.sum
        elif opt.output == 'mf':
            self.model = MF(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "fm_own":
            self.model = FM_OWN(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == "fm_global_bias":
            self.model = FM_GLOBAL_BIAS(opt.feature_dim, opt.user_num, opt.item_num)
        else:
            raise ValueError("output 出错")

    def forward(self, feature, uid, iid):
        if self.output == "lfm":
            return self.model(feature, uid, iid)
        elif self.output == "lfm_standard":
            return self.model(feature, uid, iid)
        elif self.output == "lfm_plus":
            return self.model(feature, u_describe_final, uid, iid)
        elif self.output == "fm_normal":
            return self.model(feature, uid, iid)
        elif self.output == "lfm_for_DSRLN":
            return self.model(feature, uid, iid)
        elif self.output == "fm_normal_for_mpcn":
            return self.model(feature, uid, iid)
        elif self.output == "nfm":
            return self.model(feature, uid, iid)
        elif self.output == "pbfl":
            return self.model(feature, uid, iid)
        elif self.output == "lfm_plus_mlp":
            return self.model(feature, uid, iid)
        elif self.output == "sum":
            return self.model(feature, 1, keepdim=True)
        elif self.output == 'mf':
            return self.model(feature, uid, iid)
        elif self.output == 'fm_own':
            return self.model(feature, uid, iid)
        elif self.output == "fm_global_bias":
            return self.model(feature, uid, iid)


class LFM(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(LFM, self).__init__()


        self.fc = nn.Linear(dim, 1)

        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc.bias, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)


    def rescale_sigmoid(self, score, a, b):
        return a + torch.sigmoid(score) * (b - a)

    def forward(self, feature, user_id, item_id):

        return self.fc(feature) + self.b_users[user_id] + self.b_items[item_id]

class LFM_FOR_DSRLN(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(LFM_FOR_DSRLN, self).__init__()


        self.fc = nn.Linear(dim, 1)

        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.b_global = nn.Parameter(torch.randn(1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc.bias, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)
        nn.init.uniform_(self.b_items, a=0.5, b=1.5)


        nn.init.uniform_(self.b_global, a=0.5, b=1.5)

    def forward(self, feature, user_id, item_id):
        return self.fc(feature) + self.b_users[user_id] + self.b_items[item_id] + self.b_global


class LFM_STANDARD(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(LFM_STANDARD, self).__init__()


        self.fc = nn.Linear(dim, 1)

        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc.bias, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)
        nn.init.uniform_(self.b_items, a=0.5, b=1.5)

    def forward(self, feature, user_id, item_id):
        return self.fc(feature) + self.b_users[user_id] + self.b_items[item_id]

class LFM_PLUS(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(LFM_PLUS, self).__init__()


        self.fc = nn.Linear(dim, 1)
        self.fc_describe = nn.Linear(dim, 1)

        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc.bias, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)

    def rescale_sigmoid(self, score, a, b):
        return a + torch.sigmoid(score) * (b - a)

    def forward(self, feature, u_describe_final, user_id, item_id):

        return self.fc(feature) + u_describe_final + self.b_users[user_id] + self.b_items[item_id]

class NFM(nn.Module):
    def __init__(self, dim):
        super(NFM, self).__init__()
        self.dim = dim

        self.fc = nn.Linear(dim, 1)

        self.fm_V = nn.Parameter(torch.randn(16, dim))
        self.mlp = nn.Linear(16, 16)
        self.h = nn.Linear(16, 1, bias=False)
        self.drop_out = nn.Dropout(0.5)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.1)
        nn.init.uniform_(self.fm_V, -0.1, 0.1)
        nn.init.uniform_(self.h.weight, -0.1, 0.1)

    def forward(self, input_vec, *args):
        fm_linear_part = self.fc(input_vec)
        fm_interactions_1 = torch.mm(input_vec, self.fm_V.t())
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2), torch.pow(self.fm_V, 2).t())
        bilinear = 0.5 * (fm_interactions_1 - fm_interactions_2)

        out = F.relu(self.mlp(bilinear))
        out = self.drop_out(out)
        out = self.h(out) + fm_linear_part
        return out


class FM_NORMAL(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(FM_NORMAL, self).__init__()
        self.dim = dim

        self.fc = nn.Linear(dim, 1)

        self.fm_V = nn.Parameter(torch.randn(dim, 10))



        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)


        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def build_fm(self, input_vec):

        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part
        return fm_output

    def forward(self, feature, uids, iids):
        fm_out = self.build_fm(feature)
        return fm_out


class FM_NORMAL_FOR_MPCN(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(FM_NORMAL_FOR_MPCN, self).__init__()
        self.dim = dim

        self.fc = nn.Linear(dim, 1)

        self.fm_V = nn.Parameter(torch.randn(dim, 5))
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.uniform_(self.fc.bias, -0.05, 0.05)

        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def build_fm(self, input_vec):

        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part
        return fm_output

    def forward(self, feature, uids, iids):
        fm_out = self.build_fm(feature)

        return fm_out + self.b_users[uids] + self.b_items[iids]

class MLP(nn.Module):

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.dim = dim

        self.fc = nn.Linear(dim, 1)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, 0.1, 0.1)
        nn.init.uniform_(self.fc.bias, a=0, b=0.2)

    def forward(self, feature, *args, **kwargs):
        return F.relu(self.fc(feature)[0])


class PBFL(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(PBFL, self).__init__()






        self.ui_id_fc_layer_1 = nn.Linear(100 * 4, 32)
        self.ui_id_fc_layer_2 = nn.Linear(32, 32)
        self.ui_id_fc_layer_3 = nn.Linear(3 * 32, 32)
        self.ui_id_fc_layer_4 = nn.Linear(32, 1)
        self.uid_emb_2 = nn.Embedding(user_num, 32)
        self.iid_emb_2 = nn.Embedding(item_num, 32)



        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        self.init_weight()

    def init_weight(self):
        for fc in [self.ui_id_fc_layer_1, self.ui_id_fc_layer_2, self.ui_id_fc_layer_3, self.ui_id_fc_layer_4]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)
        nn.init.uniform_(self.uid_emb_2.weight, -0.1, 0.1)
        nn.init.uniform_(self.iid_emb_2.weight, -0.1, 0.1)





    def forward(self, feature, uids, iids):

        e_0 = feature
        e_1 = self.elu(self.ui_id_fc_layer_1(e_0))
        e_2 = self.elu(self.ui_id_fc_layer_2(e_1))


        r_0 = torch.cat((e_2, self.uid_emb_2(uids), self.iid_emb_2(iids)), 1)
        r_1 = self.elu(self.ui_id_fc_layer_3(r_0))
        r_2 = self.elu(self.ui_id_fc_layer_4(r_1))
        return r_2



class Lfm_Plus_Mlp(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(Lfm_Plus_Mlp, self).__init__()


        self.fc = nn.Linear(32, 1)
        self.fc_1 = nn.Linear(32, 1)


        self.mlp_layers = nn.ModuleList()
        input_dim = 300


        mlp_hidden_dims = [300, 1]
        for hidden_dim in mlp_hidden_dims:
            self.mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.W_lfm = nn.Linear(300, 1, bias=False)
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)
        self.b_lfm = nn.Parameter(torch.zeros(1))

        self.init_weight()

    def init_weight(self):
        pass








    def forward(self, feature, user_id, item_id):
        x=feature
        for layer in self.mlp_layers:
            x = layer(x)

        mlp_feature = x
        user_b = self.user_bias(user_id).squeeze(1)
        item_b = self.item_bias(item_id).squeeze(1)


        lfm_feature = (self.W_lfm(feature).squeeze(1)+user_b+item_b+self.b_lfm).unsqueeze(1)

        return 0.5*mlp_feature + 0.5*lfm_feature

class MF(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(MF, self).__init__()

        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))
        self.global_mean = nn.Parameter(torch.tensor(4.3467))
        self.init_weight()

    def forward(self, feature, user_id, item_id):
        u_feature = feature[:, :32]
        i_feature = feature[:, 32:]
        pred = torch.matmul(u_feature.unsqueeze(1), i_feature.unsqueeze(2)).squeeze(1)
        return pred + self.b_users[user_id] + self.b_items[item_id] + self.global_mean

    def init_weight(self):
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)
        nn.init.uniform_(self.b_items, a=0.5, b=1.5)

class FM_OWN(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(FM_OWN, self).__init__()
        self.dim = dim

        self.fc = nn.Linear(dim, 1)

        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def build_fm(self, input_vec):

        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part
        return fm_output

    def forward(self, feature, uids, iids):
        fm_out = self.build_fm(feature)
        return fm_out + self.b_users[uids] + self.b_items[iids]

class FM_GLOBAL_BIAS(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(FM_GLOBAL_BIAS, self).__init__()
        self.dim = dim

        self.fc = nn.Linear(dim, 1)

        self.fm_V = nn.Parameter(torch.randn(dim, 10))
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))
        self.b_global = nn.Parameter(torch.randn(1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)
        nn.init.uniform_(self.b_global, a=0.5, b=1.5)

    def build_fm(self, input_vec):

        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V)
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2))
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, 1, keepdim=True) + fm_linear_part
        return fm_output

    def forward(self, feature, uids, iids):
        fm_out = self.build_fm(feature)
        return fm_out + self.b_users[uids] + self.b_items[iids]+ self.b_global
