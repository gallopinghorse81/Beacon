# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import time

from .prediction import PredictionLayer
from .fusion import FusionLayer


class Models_for_Context_Coatt(nn.Module):

    def __init__(self, opt, Net):
        super(Models_for_Context_Coatt, self).__init__()
        self.opt = opt
        self.model_name = self.opt.model
        self.model = Net(opt)



        if self.opt.ui_merge == 'cat':
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * self.opt.num_fea * 2
            else:
                feature_dim = self.opt.id_emb_size * 2
        else:
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * self.opt.num_fea
            else:
                feature_dim = self.opt.id_emb_size

        self.opt.feature_dim = feature_dim
        self.fusion_net = FusionLayer(opt)
        self.predict_net = PredictionLayer(opt)
        self.dropout = nn.Dropout(self.opt.drop_out)

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
            user_doc, item_doc, user_summaries, item_summaries, cateids,\
            user_item_cate2id, item_user_cate2id, user_reviews_bert, item_reviews_bert,\
            user_aux_reviews_bert  = datas















        user_feature, item_feature = self.model(datas)
        ui_feature = self.fusion_net(user_feature, item_feature)
        ui_feature = self.dropout(ui_feature)


        if self.opt.output != "lfm_plus":
            output = self.predict_net(ui_feature, uids, iids).squeeze(1)
        else:
            output = self.predict_net(ui_feature, u_describe_final, uids, iids).squeeze(1)


        return output


    def load(self, path):

        self.load_state_dict(torch.load(path))


    def save(self, name=None):






        torch.save(self.state_dict(), name)
        return name
