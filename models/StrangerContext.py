
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class StrangerContext(nn.Module):
    def __init__(self, opt):
        super(StrangerContext, self).__init__()
        self.opt = opt
        self.num_fea = 2
        self.bert_dim = opt.bert_dim

        self.u_max_r = opt.setNum_rev_of_user
        self.i_max_r = opt.setNum_rev_of_item
        self.review_len = opt.setNum_word_of_rev
        self.num_heads = opt.num_heads


        self.projection_dim = 256
        self.input_projection = nn.Linear(self.bert_dim, self.projection_dim)



        aspect_vectors_address = "aspect_vectors_" + str(self.opt.num_aspect)
        initial_aspect_vectors_tensor = getattr(self.opt, aspect_vectors_address).cuda()
        self.aspect_vectors = nn.Parameter(initial_aspect_vectors_tensor, requires_grad=False)
        self.aspect_proj = nn.Linear(opt.bert_dim, opt.aspect_emb_dim)


        self.review_fusion = ReviewFusion(self.bert_dim)


        self.user_attn_scorers  = nn.ModuleList([
            ContextGatedScoringBlock(d_model=self.projection_dim, n_heads=self.num_heads, d_ff=self.projection_dim*2, dropout=opt.drop_out_for_multihead)
            for _ in range(opt.num_key_reviews)
        ])
        self.item_attn_scorers  = nn.ModuleList([
            ContextGatedScoringBlock(d_model=self.projection_dim, n_heads=self.num_heads, d_ff=self.projection_dim*2, dropout=opt.drop_out_for_multihead)
            for _ in range(opt.num_key_reviews)
        ])


        self.aspect_review_attention = AspectReviewAttention(self.projection_dim, opt.aspect_emb_dim)


        self.dropout = nn.Dropout(opt.drop_out)
        self.uid_emb = nn.Embedding(opt.user_num, opt.id_emb_size)
        self.iid_emb = nn.Embedding(opt.item_num, opt.id_emb_size)
        self.u_fc_layer = nn.Linear(self.opt.aspect_emb_dim * self.opt.num_aspect, self.opt.id_emb_size)
        self.i_fc_layer = nn.Linear(self.opt.aspect_emb_dim * self.opt.num_aspect, self.opt.id_emb_size)
        self.reset_para()

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, \
        user_doc, item_doc, user_summaries, item_summaries, cateids, \
        user_item_cate2id, item_user_cate2id, user_reviews_bert, item_reviews_bert, \
        user_aux_reviews_bert = datas


        user_reviews_bert = user_reviews_bert.detach()
        item_reviews_bert = item_reviews_bert.detach()
        user_aux_reviews_bert = user_aux_reviews_bert.detach()


        batch_size = uids.size(0)
        fused_user_revs_high_dim = self.review_fusion(user_reviews_bert, user_aux_reviews_bert)


        fused_user_revs = self.input_projection(fused_user_revs_high_dim)
        item_reviews_proj = self.input_projection(item_reviews_bert)

        user_key_revs_list = []
        item_key_revs_list = []

        user_scores_mask = torch.zeros(batch_size, self.u_max_r, device=uids.device)
        item_scores_mask = torch.zeros(batch_size, self.i_max_r, device=iids.device)

        for i in range(self.opt.num_key_reviews):

            user_scores = self.user_attn_scorers[i](fused_user_revs, item_reviews_proj)
            masked_user_scores = user_scores + user_scores_mask

            key_user_idx = torch.argmax(masked_user_scores, dim=1)
            user_scores_mask.scatter_(1, key_user_idx.unsqueeze(1), -1e9)
            idx_expanded = key_user_idx.view(batch_size, 1, 1).expand(-1, -1, self.projection_dim)
            selected_user_rev = torch.gather(fused_user_revs, 1, idx_expanded)
            user_key_revs_list.append(selected_user_rev)


            item_scores = self.item_attn_scorers[i](item_reviews_proj, fused_user_revs)
            masked_item_scores = item_scores + item_scores_mask

            key_item_idx = torch.argmax(masked_item_scores, dim=1)
            item_scores_mask.scatter_(1, key_item_idx.unsqueeze(1), -1e9)
            idx_expanded = key_item_idx.view(batch_size, 1, 1).expand(-1, -1, self.projection_dim)
            selected_item_rev = torch.gather(item_reviews_proj, 1, idx_expanded)
            item_key_revs_list.append(selected_item_rev)


        user_key_revs_bert = torch.cat(user_key_revs_list, dim=1)
        item_key_revs_bert = torch.cat(item_key_revs_list, dim=1)



        user_aspect_reps, item_aspect_reps = [], []
        projected_aspect_vectors = self.aspect_proj(self.aspect_vectors)
        for i in range(self.opt.num_aspect):
            aspect_vec_i = projected_aspect_vectors[i].unsqueeze(0).expand(batch_size, -1)
            u_rep = self.aspect_review_attention(user_key_revs_bert, aspect_vec_i)
            i_rep = self.aspect_review_attention(item_key_revs_bert, aspect_vec_i)
            user_aspect_reps.append(u_rep)
            item_aspect_reps.append(i_rep)

        u_fea = torch.cat(user_aspect_reps, dim=1)
        i_fea = torch.cat(item_aspect_reps, dim=1)
        u_fea_final = torch.stack([self.uid_emb(uids), self.u_fc_layer(u_fea)], 1)
        i_fea_final = torch.stack([self.iid_emb(iids), self.i_fc_layer(i_fea)], 1)

        return u_fea_final, i_fea_final

    def reset_para(self):

        print("Initializing uid_emb and iid_emb with Xavier Uniform.")
        xavier_uniform_(self.uid_emb.weight)
        xavier_uniform_(self.iid_emb.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_(m.weight, 1)
                constant_(m.bias, 0)





class ReviewFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate_linear = nn.Linear(embed_dim * 2, embed_dim)
        self.transform_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, user_rev_bert, stranger_rev_bert):
        concat_vec = torch.cat([user_rev_bert, stranger_rev_bert], dim=-1)
        gate = torch.sigmoid(self.gate_linear(concat_vec))
        fused_vec = gate * user_rev_bert + (1 - gate) * torch.tanh(self.transform_linear(stranger_rev_bert))
        return fused_vec




class ContextGatedScoringBlock(nn.Module):

    def __init__(self, d_model=768, n_heads=8, d_ff=None, dropout=0.1):
        super().__init__()





        self.query_modulator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )


        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


        self.scorer = nn.Linear(d_model, 1)

    def forward(self, query, key_value):


        kv_summary = torch.mean(key_value, dim=1, keepdim=True)



        kv_summary_expanded = kv_summary.expand_as(query)
        gate_input = torch.cat([query, kv_summary_expanded], dim=-1)


        modulation_weights = self.query_modulator(gate_input)
        gated_query = query * modulation_weights


        attn_output, _ = self.cross_attn(query=gated_query, key=key_value, value=key_value)



        x = self.norm1(query + self.dropout(attn_output))


        ffn_output = self.ffn(x)
        enhanced_query = self.norm2(x + self.dropout(ffn_output))


        scores = self.scorer(enhanced_query)


        return scores.squeeze(-1)


class AspectReviewAttention(nn.Module):
    def __init__(self, projection_dim, aspect_emb_dim):
        super().__init__()
        self.proj_review = nn.Linear(projection_dim, aspect_emb_dim, bias=False)
        self.scorer = nn.Linear(aspect_emb_dim, 1, bias=False)

    def forward(self, key_reviews_bert, aspect_vec):
        review_proj = self.proj_review(key_reviews_bert)
        scores = self.scorer(torch.tanh(review_proj + aspect_vec.unsqueeze(1))).squeeze(-1)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        aspect_specific_rep = torch.sum(attn_weights * review_proj, dim=1)
        return aspect_specific_rep
