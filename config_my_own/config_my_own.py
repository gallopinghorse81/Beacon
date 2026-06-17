# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


class DefaultConfig:
    model = 'DeepCoNN'
    dataset = 'Digital_Music_data_pre'


    use_gpu = True
    gpu_id = 0
    multi_gpu = False
    gpu_ids = []

    seed = 2019
    num_epochs = 20
    num_workers = 0

    optimizer = 'Adam'
    weight_decay = 1e-3
    lr = 2e-3
    loss_method = 'mse'
    drop_out = 0.1
    summary_model_drop_out = 0.2
    drop_out_for_multihead = 0.3

    use_word_embedding = True

    num_aspect = 3
    id_emb_size = 32
    att_id_emb_size = 100
    query_mlp_size = 128
    fc_dim = 32
    co_multi_head = 4

    biaoji = 100
    doc_len = 500
    filters_num = 100
    kernel_size = 3

    num_fea = 1
    use_review = True
    use_doc = True
    self_att = False


    r_filters_num = 100
    kernel_size = 3
    attention_size = 32
    att_method = 'matrix'
    review_weight = 'softmax'
    aspect_num_heads = 1


    r_id_merge = 'cat'
    ui_merge = 'cat'
    output = 'fm_global_bias'

    fine_step = True
    save_path = ""
    print_opt = 'default'


    batch_size = 128
    print_step = 100
    have_saved_model = 0
    best_valid_mse = 2.00
    setNum_rev_of_user = -1
    setNum_rev_of_item = -1
    setNum_word_of_rev = -1
    setNum_word_of_summary = -1
    category_num = -1
    train_data_size = -1
    step_size = 3
    gamma = 0.5


    num_heads = 2


    num_gcn_layers = 1
    gcn_num_heads = 1


    cl_rate_1 = 0.1
    cl_node_drop_rate = 0.1


    cl_rate_2 = 0.1
    cl_edge_drop_rate = 0.2
    sim_threshold = 0.1

    ssl_temp = 0.2

    bert_dim = 768
    num_key_reviews = 3
    aspect_emb_dim = 128


    def set_path(self, name):
        self.data_root = f'./dataset/{name}'
        prefix = f'{self.data_root}/train'

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'


        self.user_summary_list_path = f'{prefix}/userSummary2Index.npy'
        self.item_summary_list_path = f'{prefix}/itemSummary2Index.npy'

        self.user_item_cate2id_path = f'{prefix}/user_item_cate2id.npy'
        self.item_user_cate2id_path = f'{prefix}/item_user_cate2id.npy'





        self.ui_src_path = f'{prefix}/ui_src.npy'
        self.ui_dst_path = f'{prefix}/ui_dst.npy'
        self.ui_node_map_index_path = f'{prefix}/ui_node_map_index.npy'



        self.edge_rating_idx_directed_path = f'{prefix}/edge_rating_idx_directed.npy'
        self.edge_review_embedding_directed_path = f'{prefix}/edge_review_embedding_directed.npy'
        self.edge_aux_review_embedding_directed_path = f'{prefix}/edge_aux_review_embedding_directed.npy'



        self.user_reviews_bert_path = f'{prefix}/user_reviews_bert.npy'
        self.item_reviews_bert_path = f'{prefix}/item_reviews_bert.npy'
        self.user_aux_reviews_bert_path = f'{prefix}/user_aux_reviews_bert.npy'
        self.item_aux_reviews_bert_path = f'{prefix}/item_aux_reviews_bert.npy'

        dataset_str = name.split("_data/data67500")[0]


        self.aspect_bertopic_vectors_1_path = f'{prefix}/{dataset_str}_topic_embeddings_1.npy'
        self.aspect_bertopic_vectors_3_path = f'{prefix}/{dataset_str}_topic_embeddings_3.npy'
        self.aspect_bertopic_vectors_5_path = f'{prefix}/{dataset_str}_topic_embeddings_5.npy'
        self.aspect_bertopic_vectors_7_path = f'{prefix}/{dataset_str}_topic_embeddings_7.npy'


    def parse(self, para):
        print("load npy from dist...")
        self.users_review_list = np.load(self.user_list_path,
                                         encoding='bytes')
        self.items_review_list = np.load(self.item_list_path, encoding='bytes')
        self.user2itemid_list = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_list = np.load(self.item2userid_path, encoding='bytes')
        self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')

        self.users_summary_list = np.load(self.user_summary_list_path,
                                          encoding='bytes')
        self.items_summary_list = np.load(self.item_summary_list_path, encoding='bytes')


        self.user_item_cate2id_list = np.load(self.user_item_cate2id_path, encoding='bytes')
        self.item_user_cate2id_list = np.load(self.item_user_cate2id_path, encoding='bytes')


        self.ui_src = torch.from_numpy(np.load(self.ui_src_path, encoding='bytes', allow_pickle=True))
        self.ui_dst = torch.from_numpy(np.load(self.ui_dst_path, encoding='bytes', allow_pickle=True))
        self.ui_node_map_index = torch.from_numpy(
            np.load(self.ui_node_map_index_path, encoding='bytes', allow_pickle=True))

        self.edge_rating_idx_directed = torch.from_numpy(
            np.load(self.edge_rating_idx_directed_path, encoding='bytes', allow_pickle=True))
        self.edge_review_embedding_directed = torch.from_numpy(
            np.load(self.edge_review_embedding_directed_path, encoding='bytes', allow_pickle=True))
        self.edge_aux_review_embedding_directed = torch.from_numpy(
            np.load(self.edge_aux_review_embedding_directed_path, encoding='bytes', allow_pickle=True))



        self.user_reviews_bert_list = np.load(self.user_reviews_bert_path, encoding='bytes', allow_pickle=True)
        self.item_reviews_bert_list = np.load(self.item_reviews_bert_path, encoding='bytes', allow_pickle=True)
        self.user_aux_reviews_bert_list = np.load(self.user_aux_reviews_bert_path, encoding='bytes', allow_pickle=True)
        self.item_aux_reviews_bert_list = np.load(self.item_aux_reviews_bert_path, encoding='bytes', allow_pickle=True)

        self.aspect_vectors_1 = torch.from_numpy(np.load(self.aspect_bertopic_vectors_1_path, encoding='bytes', allow_pickle=True))
        self.aspect_vectors_3 = torch.from_numpy(np.load(self.aspect_bertopic_vectors_3_path, encoding='bytes', allow_pickle=True))
        self.aspect_vectors_5 = torch.from_numpy(np.load(self.aspect_bertopic_vectors_5_path, encoding='bytes', allow_pickle=True))
        self.aspect_vectors_7 = torch.from_numpy(np.load(self.aspect_bertopic_vectors_7_path, encoding='bytes', allow_pickle=True))


        for k, v in para.items():
            if not hasattr(self, k):
                print(k, v)
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config_my_own:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')

class Amazon_Instant_Video_data_67500_Config(DefaultConfig):
    def __init__(self):
        self.set_path('Amazon_Instant_Video_data/data67500')

    vocab_size = 16128
    word_dim = 300

    setNum_word_of_rev = 195
    setNum_rev_of_user = 9
    setNum_rev_of_item = 37
    setNum_word_of_summary = 8
    category_num = 1 + 2

    user_num = 5130 + 2
    item_num = 1685 + 2

    train_data_size = 29700
