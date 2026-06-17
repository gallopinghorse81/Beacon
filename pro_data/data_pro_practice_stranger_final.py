# -*- coding: utf-8 -*-
import json
import pandas as pd
import re
import sys
import os
import gzip
import numpy as np
import time
from sklearn.model_selection import train_test_split
from operator import itemgetter
import gensim
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import random
import torch.nn.functional as F

P_REVIEW = 0.9







MAX_DF = 1.0
MIN_DF = 5
MAX_VOCAB = 50000
DOC_LEN = 500
PRE_W2V_BIN_PATH = "GoogleNews-vectors-negative300.bin"


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def parse(path):
    g = gzip.open(path, 'r')
    for l in tqdm(g):
        yield eval(l)


def get_count(data, id):

    idList = data[[id, 'ratings']].groupby(id, as_index=False)
    idListCount = idList.size()
    return idListCount


def get_number_of_users_in_data(data):
    userCount, itemCount = get_count(data, 'user_id'), get_count(data, 'item_id')
    userNum_all = userCount.shape[0]
    itemNum_all = itemCount.shape[0]
    print("===============Start:all  rawData size======================")
    print(f"raw data Num: {data.shape[0]}")
    print(f"raw user Num: {userNum_all}")
    print(f"raw item Num: {itemNum_all}")
    print(f"data densiy: {data.shape[0] / float(userNum_all * itemNum_all):.4f}")
    print("===============End: rawData size========================")
    return userNum_all, itemNum_all


def filter_triplets(tp, min_uc=20, min_sc=20):

    for _ in range(10):
        usercount = get_count(tp, 'user_id')
        tp = tp[tp['user_id'].isin(usercount.index[usercount >= min_uc])]

        songcount = get_count(tp, 'item_id')
        tp = tp[tp['item_id'].isin(songcount.index[songcount >= min_sc])]

        usercount, songcount = get_count(tp, 'user_id'), get_count(tp, 'item_id')
        print(tp.shape[0])
        print(usercount.shape[0])
        print(songcount.shape[0])
        print("\n")

    return tp


def numerize(data):
    userCount, itemCount, categoryCount = get_count(data, 'user_id'), get_count(data, 'item_id'), get_count(data, 'category')
    uidList = userCount.index
    iidList = itemCount.index
    cateidList = categoryCount.index

    user2id = dict((uid, i) for (i, uid) in enumerate(uidList))
    item2id = dict((iid, i) for (i, iid) in enumerate(iidList))
    cate2id = dict((cateid, i) for (i, cateid) in enumerate(cateidList))

    uid = list(map(lambda x: user2id[x], data['user_id']))
    iid = list(map(lambda x: item2id[x], data['item_id']))
    cateid = list(map(lambda x: cate2id[x], data['category']))

    data['user_id'] = uid
    data['item_id'] = iid
    data['category'] = cateid


    item_category_dict = {}
    for i in data.values:
        if i[1] in item_category_dict:
            continue
        else:
            item_category_dict[i[1]] = i[5]

    return data, item_category_dict


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()


def construct_reviews_dict_and_iid_dict_from_train_data(data_train, filename, item_category_dict):
    user_reviews_dict = {}
    item_reviews_dict = {}
    user_iid_dict = {}
    item_uid_dict = {}


    user_i_cateid_dict = {}
    item_u_cateid_dict = {}


    user_summaries_dict = {}
    item_summaries_dict = {}
    user_reviews_and_summaries_dict = {}


    user_rate_dict = {}
    item_rate_dict = {}
    user_revidx_dict = {}
    item_revidx_dict = {}


    for rev_idx, i in enumerate(data_train.values):
        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))

        str_summary = clean_str(i[4].encode('ascii', 'ignore').decode('ascii'))
        str_review_and_summary = clean_str(i[3].encode('ascii', 'ignore').decode('ascii') + ' ' \
                                           + i[4].encode('ascii', 'ignore').decode('ascii'))

        if filename == "Yelp2013":
            str_review = clean_str(str_review)

        if len(str_review.strip()) == 0:
            str_review = "<unk>"
        if len(str_summary.strip()) == 0:
            str_summary = "<unk>"
        if len(str_review_and_summary.strip()) == 0:
            str_review_and_summary = "<unk>"


        if i[0] in user_reviews_dict:
            user_reviews_dict[i[0]].append(str_review)
            user_summaries_dict[i[0]].append(str_summary)
            user_iid_dict[i[0]].append(i[1])
            user_reviews_and_summaries_dict[i[0]].append(str_review_and_summary)
            user_i_cateid_dict[i[0]].append(item_category_dict[i[1]])

            user_rate_dict[i[0]].append(float(i[2]))
            user_revidx_dict[i[0]].append(rev_idx)
        else:
            user_reviews_dict[i[0]] = [str_review]
            user_summaries_dict[i[0]] = [str_summary]
            user_iid_dict[i[0]] = [i[1]]
            user_reviews_and_summaries_dict[i[0]] = [str_review_and_summary]
            user_i_cateid_dict[i[0]] = [item_category_dict[i[1]]]

            user_rate_dict[i[0]] = [float(i[2])]
            user_revidx_dict[i[0]] = [rev_idx]

        if i[1] in item_reviews_dict:
            item_reviews_dict[i[1]].append(str_review)
            item_summaries_dict[i[1]].append(str_summary)
            item_uid_dict[i[1]].append(i[0])
            item_u_cateid_dict[i[1]].append(item_category_dict[i[1]])

            item_rate_dict[i[1]].append(float(i[2]))
            item_revidx_dict[i[1]].append(rev_idx)
        else:
            item_reviews_dict[i[1]] = [str_review]
            item_summaries_dict[i[1]] = [str_summary]
            item_uid_dict[i[1]] = [i[0]]
            item_u_cateid_dict[i[1]] = [item_category_dict[i[1]]]

            item_rate_dict[i[1]] = [float(i[2])]
            item_revidx_dict[i[1]] = [rev_idx]




    user_reviews = []
    for ind in range(len(user_reviews_dict)):
        user_reviews.append(' <SEP> '.join(user_reviews_dict[ind]))




    item_reviews = []
    for ind in range(len(item_reviews_dict)):
        item_reviews.append(' <SEP> '.join(item_reviews_dict[ind]))

    user_reviews_and_summaries = []
    for ind in range(len(user_reviews_and_summaries_dict)):
        user_reviews_and_summaries.append(' <SEP> '.join(user_reviews_and_summaries_dict[ind]))

    vectorizer = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_VOCAB)



    vectorizer.fit(user_reviews_and_summaries)
    vocab = vectorizer.vocabulary_

    vocab['<SEP>'] = MAX_VOCAB












    def clean_review(user_reviews_dict):
        new_dict = {}
        for k, text in user_reviews_dict.items():
            new_reviews = []
            for r in text:
                words = ' '.join([w for w in r.split() if w in vocab])
                new_reviews.append(words)
            new_dict[k] = new_reviews
        return new_dict

    def calculate_doc_len(user_reviews):

        len_sum_list = []

        for line in user_reviews:
            review = [word for word in line.split() if word in vocab]
            len_sum_list.append(len(review))

        def percent_len_sum(rlist, percent):
            x = np.sort(rlist)
            xLen = len(x)
            return x[int(percent * xLen) - 1]

        print("从小到大排序后前25%,50%,75%,85%单词数分别为：")
        print(percent_len_sum(len_sum_list, 0.25))
        print(percent_len_sum(len_sum_list, 0.5))
        print(percent_len_sum(len_sum_list, 0.75))
        print(percent_len_sum(len_sum_list, 0.85))
        return percent_len_sum(len_sum_list, P_REVIEW)

    def clean_doc(user_reviews, set_word_num_doc):
        new_raw = []
        for line in user_reviews:
            review = [word for word in line.split() if word in vocab]
            if len(review) > set_word_num_doc:
                review = review[:set_word_num_doc]
            new_raw.append(review)
        return new_raw

    user_reviews_dict = clean_review(user_reviews_dict)
    item_reviews_dict = clean_review(item_reviews_dict)





    user_summaries_dict = clean_review(user_summaries_dict)
    item_summaries_dict = clean_review(item_summaries_dict)


    print("user doc word")
    calculate_doc_len(user_reviews)
    user_review2doc = clean_doc(user_reviews, DOC_LEN)

    print("item doc word")
    calculate_doc_len(item_reviews)
    item_review2doc = clean_doc(item_reviews, DOC_LEN)


    word_index = {}
    word_index['<unk>'] = 0


    index_num_map_word_index = {}
    index_num_map_word_index[0] = '<unk>'


    for i, w in enumerate(vocab.keys(), 1):
        word_index[w] = i
        index_num_map_word_index[i] = w

    print(f"The vocab size: {len(word_index)}")
    print(f"Average user document length: {sum([len(i) for i in user_review2doc]) / len(user_review2doc)}")
    print(f"Average item document length: {sum([len(i) for i in item_review2doc]) / len(item_review2doc)}")






    return word_index, user_review2doc, item_review2doc, \
        user_reviews_dict, item_reviews_dict, user_summaries_dict, item_summaries_dict, \
        user_iid_dict, item_uid_dict, index_num_map_word_index, \
        user_i_cateid_dict, item_u_cateid_dict, user_rate_dict, item_rate_dict, \
        user_revidx_dict, item_revidx_dict, vocab

def construct_revid_dict_from_train_data(data_train, filename, setNum_rev_of_user, setNum_rev_of_item, userNum_all, itemNum_all):
    user_reviews_dict = {}
    item_reviews_dict = {}
    user_iid_dict = {}
    item_uid_dict = {}

    user_revid_dict = {}
    item_revid_dict = {}

    user_rt_dict = {}
    item_rt_dict = {}

    num_idx = 0

    for i in data_train.values:
        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))
        if filename == "Yelp2013":
            str_review = clean_str(str_review)

        if len(str_review.strip()) == 0:
            str_review = "<unk>"


        if i[0] in user_reviews_dict:
            user_reviews_dict[i[0]].append(str_review)
            user_iid_dict[i[0]].append(i[1])

            user_revid_dict[i[0]].append(num_idx)
            user_rt_dict[i[0]].append(float(i[2]))
        else:
            user_reviews_dict[i[0]] = [str_review]
            user_iid_dict[i[0]] = [i[1]]

            user_revid_dict[i[0]] = [num_idx]
            user_rt_dict[i[0]] = [float(i[2])]

        if i[1] in item_reviews_dict:
            item_reviews_dict[i[1]].append(str_review)
            item_uid_dict[i[1]].append(i[0])

            item_revid_dict[i[1]].append(num_idx)
            item_rt_dict[i[1]].append(float(i[2]))
        else:
            item_reviews_dict[i[1]] = [str_review]
            item_uid_dict[i[1]] = [i[0]]

            item_revid_dict[i[1]] = [num_idx]
            item_rt_dict[i[1]] = [float(i[2])]

        num_idx = num_idx + 1

    def padding_rev_or_rt_ids(iids, num, pad_id):
        if len(iids) >= num:
            new_iids = iids[:num]
        else:
            new_iids = iids + [pad_id] * (num - len(iids))
        return new_iids

    user_revid_list = []
    user_rt_list = []
    for i in range(userNum_all):
        user_revid_list.append(padding_rev_or_rt_ids(user_revid_dict[i], setNum_rev_of_user, -1))
        user_rt_list.append(padding_rev_or_rt_ids(user_rt_dict[i], setNum_rev_of_user, -1))

    item_revid_list = []
    item_rt_list = []
    for i in range(itemNum_all):
        item_revid_list.append(padding_rev_or_rt_ids(item_revid_dict[i], setNum_rev_of_item, -1))
        item_rt_list.append(padding_rev_or_rt_ids(item_rt_dict[i], setNum_rev_of_item, -1))


    return user_revid_list, item_revid_list, user_rt_list, item_rt_list



def countNum(user_reviews_dict):


    minNum_rev_of_user = 100
    maxNum_rev_of_user = 0

    minNum_word_of_rev = 3000
    maxNum_word_of_rev = 0

    numList_rev_of_user = []
    numList_word_of_rev = []

    for i, rev_list in user_reviews_dict.items():
        if len(rev_list) < minNum_rev_of_user:
            minNum_rev_of_user = len(rev_list)
        if len(rev_list) > maxNum_rev_of_user:
            maxNum_rev_of_user = len(rev_list)
        numList_rev_of_user.append(len(rev_list))
        for rev in rev_list:
            if rev != "":
                wordTokens = rev.split()
            if len(wordTokens) < minNum_word_of_rev:
                minNum_word_of_rev = len(wordTokens)
            if len(wordTokens) > maxNum_word_of_rev:
                maxNum_word_of_rev = len(wordTokens)
            numList_word_of_rev.append(len(wordTokens))

    def percent_num(rlist):
        x = np.sort(rlist)
        xLen = len(x)

        print(x[int(0.25 * xLen) - 1])
        print(x[int(0.5 * xLen) - 1])
        print(x[int(0.75 * xLen) - 1])
        print(x[int(0.85 * xLen) - 1])

        return x[int(P_REVIEW * xLen) - 1]

    print("user rev num")
    setNum_rev_of_user = percent_num(numList_rev_of_user)


    print("user word num per rev")
    setNum_word_of_rev = percent_num(numList_word_of_rev)


    return setNum_rev_of_user, setNum_word_of_rev


def construct_pandas_data_frame_from_file(filename):
    users_id = []
    items_id = []
    ratings = []
    reviews = []
    summaries = []

    if filename == "Yelp2013":

        train_dir = filename + "/yelp-2013-seg-20-20.train.ss"
        file = open(train_dir, encoding='UTF-8')
        length = len(open(train_dir, encoding='UTF-8').readlines())
        for line in tqdm(file, total=length):
            value = line.split('\t\t')
            users_id.append(value[0])
            items_id.append(value[1])
            ratings.append(value[2])
            reviews.append(value[3])


        dev_dir = filename + "/yelp-2013-seg-20-20.dev.ss"
        file = open(dev_dir, encoding='UTF-8')
        length = len(open(dev_dir, encoding='UTF-8').readlines())
        for line in tqdm(file, total=length):
            value = line.split('\t\t')
            users_id.append(value[0])
            items_id.append(value[1])
            ratings.append(value[2])
            reviews.append(value[3])

        test_dir = filename + "/yelp-2013-seg-20-20.test.ss"
        file = open(test_dir, encoding='UTF-8')
        length = len(open(test_dir, encoding='UTF-8').readlines())
        for line in tqdm(file, total=length):
            value = line.split('\t\t')
            users_id.append(value[0])
            items_id.append(value[1])
            ratings.append(value[2])
            reviews.append(value[3])
    elif filename == "Yelp2017":
        filedir = filename + ".json"
        file = open(filedir, encoding='UTF-8')
        length = len(open(filedir, encoding='UTF-8').readlines())
        for line in tqdm(file, total=length):
            js = json.loads(line)
            if str(js['user_id']) == 'unknown':
                print("unknown user id")
                continue
            if str(js['business_id']) == 'unknown':
                print("unkown item id")
                continue
            reviews.append(js['text'])
            users_id.append(str(js['user_id']))
            items_id.append(str(js['business_id']))
            ratings.append(str(js['stars']))
    else:
        file = open(filename, errors='ignore')
        for line in file:
            js = json.loads(line)
            if str(js['reviewerID']) == 'unknown':
                print("unknown user id")
                continue
            if str(js['asin']) == 'unknown':
                print("unkown item id")
                continue
            reviews.append(js['reviewText'])
            users_id.append(str(js['reviewerID']))
            items_id.append(str(js['asin']))
            ratings.append(str(js['overall']))
            summaries.append(js['summary'])

    data_frame = {'user_id': pd.Series(users_id), 'item_id': pd.Series(items_id),
                  'ratings': pd.Series(ratings), 'reviews': pd.Series(reviews), 'summaries': pd.Series(summaries)}
    data = pd.DataFrame(data_frame)


    del users_id, items_id, ratings, reviews, summaries
    return data


def construct_pandas_data_category_frame_from_file(filename, data_review_frame):
    meta_flie = "meta_" + filename[:-7] + ".json.gz"
    items_id = []
    categories_second_list = []
    categories_set = set()
    item_asins = set(list(data_review_frame['item_id']))

    for info in parse(meta_flie):
        if info['asin'] not in item_asins:
            continue
        if 'categories' not in info:
            print("wrong answer")
            continue
        if str(info['categories']) == '':
            print("really?")
            continue
        items_id.append(str(info['asin']))
        cate_all = info['categories'][0]
        length = len(cate_all)
        if length >= 2:
            cate_second = cate_all[1]
        else:
            cate_second = cate_all[0]

        categories_second_list.append(cate_second)

        if cate_second not in categories_set:
            categories_set.add(cate_second)


    print("!!!")
    print(len(categories_set))
    print(categories_set)


    data_frame_dict = {'item_id': pd.Series(items_id), 'category': pd.Series(categories_second_list)}
    data_category_frame = pd.DataFrame(data_frame_dict)



    del items_id, categories_second_list, item_asins
    return data_category_frame, len(categories_set)


def check_users_in_train_data(data_train, data_test, userNum_all, itemNum_all):


    userCount, itemCount = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    uids_train = userCount.index
    iids_train = itemCount.index
    userNum_in_train = userCount.shape[0]
    itemNum_in_train = itemCount.shape[0]
    print("===============Start: no-preprocess: trainData size======================")
    print("original train data Num: {}".format(data_train.shape[0]))
    print("original train user Num: {}".format(userNum_in_train))
    print("original train item Num: {}".format(itemNum_in_train))
    print("===============End: no-preprocess: trainData size========================")

    uidMiss = []
    iidMiss = []
    if userNum_in_train != userNum_all or itemNum_in_train != itemNum_all:
        for uid in range(userNum_all):
            if uid not in uids_train:
                uidMiss.append(uid)
        for iid in range(itemNum_all):
            if iid not in iids_train:
                iidMiss.append(iid)

    if len(uidMiss):
        for uid in uidMiss:
            df_temp = data_test[data_test['user_id'] == uid]
            data_test = data_test[data_test['user_id'] != uid]
            data_train = pd.concat([data_train, df_temp])

    if len(iidMiss):
        for iid in iidMiss:
            df_temp = data_test[data_test['item_id'] == iid]
            data_test = data_test[data_test['item_id'] != iid]
            data_train = pd.concat([data_train, df_temp])

    userCount, itemCount = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    uids_train = userCount.index
    iids_train = itemCount.index
    userNum_in_train = userCount.shape[0]
    itemNum_in_train = itemCount.shape[0]
    print("===============Start: final trainData size======================")
    print("final train data Num: {}".format(data_train.shape[0]))
    print("final train user Num: {}".format(userNum_in_train))
    print("final train item Num: {}".format(itemNum_in_train))
    print("===============End: final trainData size========================")

    return data_train, data_test

def extract_rating_idx_directed_array(data_dict):
    y = []
    for i in data_dict.values:
        y.append(int(float(i[2])))
    edge_rating_idx_directed = np.array(y)
    return edge_rating_idx_directed

def extract(data_dict):
    x = []
    y = []
    for i in data_dict.values:
        uid = i[0]
        iid = i[1]
        cateid = i[5]
        x.append([uid, iid, cateid])
        y.append(float(i[2]))
    return x, y


def construct_user_item_pair_rating(data_train, data_val, data_test):
    x_train, y_train = extract(data_train)
    x_val, y_val = extract(data_val)
    x_test, y_test = extract(data_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


def construct_ui_graph_from_train_data(data_train, userNum_all, itemNum_all):
    ui_src = []
    ui_dst = []
    for i in data_train.values:
        ui_src.append(i[0])
        ui_dst.append(i[1] + userNum_all)
    ui_node_map_index = []
    for i in range(userNum_all + itemNum_all):
        ui_node_map_index.append(i)
    return ui_src, ui_dst, ui_node_map_index

def save_npy_to_save_folder(save_folder, x_train, y_train, x_val, y_val, x_test, y_test, ui_src, ui_dst,
                            ui_node_map_index):
    np.save(f"{save_folder}/train/Train.npy", x_train)
    np.save(f"{save_folder}/train/Train_Score.npy", y_train)
    np.save(f"{save_folder}/val/Val.npy", x_val)
    np.save(f"{save_folder}/val/Val_Score.npy", y_val)
    np.save(f"{save_folder}/test/Test.npy", x_test)
    np.save(f"{save_folder}/test/Test_Score.npy", y_test)

    np.save(f"{save_folder}/train/ui_src.npy", ui_src)
    np.save(f"{save_folder}/train/ui_dst.npy", ui_dst)
    np.save(f"{save_folder}/train/ui_node_map_index.npy", ui_node_map_index)
    print(f"Train data size: {len(x_train)}")
    print(f"Val data size: {len(x_val)}")
    print(f"Test data size: {len(x_test)}")


def make_save_folder(filename, number_of_fold):
    idx_str = str(number_of_fold)
    if filename == "Yelp2013":
        save_folder = '../dataset/' + "Yelp2013" + "_data" + "/data" + idx_str
    elif filename == "Yelp2017":
        save_folder = '../dataset/' + "Yelp2017" + "_data" + "/data" + idx_str
    else:
        save_folder = '../dataset/' + filename[:-7] + "_data" + "/data" + idx_str
    print(f"数据集名称：{save_folder}")

    if not os.path.exists(save_folder + '/train'):
        os.makedirs(save_folder + '/train')
    if not os.path.exists(save_folder + '/val'):
        os.makedirs(save_folder + '/val')
    if not os.path.exists(save_folder + '/test'):
        os.makedirs(save_folder + '/test')

    return save_folder


def padding_text(textList, num):
    new_textList = []
    if len(textList) >= num:
        new_textList = textList[:num]
    else:
        padding = [[0] * len(textList[0]) for _ in range(num - len(textList))]
        new_textList = textList + padding
    return new_textList


def padding_ids(iids, num, pad_id):
    if len(iids) >= num:
        new_iids = iids[:num]
    else:
        new_iids = iids + [pad_id] * (num - len(iids))
    return new_iids


def padding_doc(doc, word_length):
    new_doc = []
    for d in doc:
        if len(d) < word_length:
            d = d + [0] * (word_length - len(d))
        else:
            d = d[:word_length]
        new_doc.append(d)
    return new_doc





































def npy_to_save_folder(save_folder, userReview2Index, user_iid_list, userDoc2Index,
                       itemReview2Index, item_uid_list, itemDoc2Index, word_index,
                       index_num_map_word_index, userSummary2Index, itemSummary2Index,
                       user_i_cateid_list, item_u_cateid_list, user_StrangerReview2Index,
                       edge_rating_idx_directed, edge_review_embedding_directed,
                       edge_aux_review_embedding_directed,  review_list_for_bertopic,
                       user_reviews_bert, item_reviews_bert,
                       user_aux_reviews_bert, item_aux_reviews_bert):
    print(f"{now()} start writing npy...")
    np.save(f"{save_folder}/train/userReview2Index.npy", userReview2Index)
    np.save(f"{save_folder}/train/user_item2id.npy", user_iid_list)
    np.save(f"{save_folder}/train/userDoc2Index.npy", userDoc2Index)





    np.save(f"{save_folder}/train/itemReview2Index.npy", itemReview2Index)
    np.save(f"{save_folder}/train/item_user2id.npy", item_uid_list)
    np.save(f"{save_folder}/train/itemDoc2Index.npy", itemDoc2Index)



    np.save(f"{save_folder}/train/word_index.npy", word_index)
    np.save(f"{save_folder}/train/index_num_map_word_index.npy", index_num_map_word_index)


    np.save(f"{save_folder}/train/userSummary2Index.npy", userSummary2Index)
    np.save(f"{save_folder}/train/itemSummary2Index.npy", itemSummary2Index)


    np.save(f"{save_folder}/train/user_item_cate2id.npy", user_i_cateid_list)
    np.save(f"{save_folder}/train/item_user_cate2id.npy", item_u_cateid_list)


    np.save(f"{save_folder}/train/review_list_for_bertopic.npy", review_list_for_bertopic)


    np.save(f"{save_folder}/train/user_StrangerReview2Index.npy", user_StrangerReview2Index)
    np.save(f"{save_folder}/train/edge_rating_idx_directed.npy", edge_rating_idx_directed)
    np.save(f"{save_folder}/train/edge_review_embedding_directed.npy", edge_review_embedding_directed)
    np.save(f"{save_folder}/train/edge_aux_review_embedding_directed.npy", edge_aux_review_embedding_directed)


    print(f"{now()} Saving final user and item BERT review embeddings...")
    np.save(f"{save_folder}/train/user_reviews_bert.npy", user_reviews_bert)
    np.save(f"{save_folder}/train/item_reviews_bert.npy", item_reviews_bert)



    print(f"{now()} Saving final user and item AUXILIARY BERT review embeddings...")
    np.save(f"{save_folder}/train/user_aux_reviews_bert.npy", user_aux_reviews_bert)
    np.save(f"{save_folder}/train/item_aux_reviews_bert.npy", item_aux_reviews_bert)


















def construct_word_emb_from_glove_txt(PRE_W2V_BIN_PATH, word_index):
    vocab_item = sorted(word_index.items(), key=itemgetter(1))
    w2v = []
    out = 0
    if PRE_W2V_BIN_PATH:
        pre_word2v = gensim.models.KeyedVectors.load_word2vec_format(PRE_W2V_BIN_PATH, binary=True)
    else:
        pre_word2v = {}

    print(f"{now()} 开始提取embedding")
    for word, key in vocab_item:
        if word in pre_word2v:
            w2v.append(pre_word2v[word])
        else:
            out += 1
            w2v.append(np.random.uniform(-1.0, 1.0, (300,)))
    print(f"out of vocab: {out}")
    print(f"w2v size: {len(w2v)}")
    w2vArray = np.array(w2v)
    print(w2vArray.shape)
    return w2v

def extract_review_idx_directed_array(data_train, word_index, setNum_word_of_rev):
    edge_review_idx_list = []
    for i in tqdm(data_train.values):
        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))
        if filename == "Yelp2013":
            str_review = clean_str(str_review)

        if len(str_review.strip()) == 0:
            str_review = "<unk>"

        review_word_idx = [word_index[w] for w in str_review.split() if w in word_index]
        if len(review_word_idx) < setNum_word_of_rev:
            review_word_idx = review_word_idx + [0] * (setNum_word_of_rev - len(review_word_idx))
        else:
            review_word_idx = review_word_idx[:setNum_word_of_rev]

        edge_review_idx_list.append(review_word_idx)

    edge_review_idx_directed = np.array(edge_review_idx_list)
    return edge_review_idx_directed





















































































def extract_review_embeddings_batched(data_train, vocab, setNum_word_of_rev,
                                      tokenizer, model, device, batch_size=32):
    print("2")
    model.to(device)
    print("3")
    model.eval()
    print("4")
    pad_token = "<unk>"


    print("Calculating embedding for an empty string to be used as padding...")
    with torch.no_grad():

        empty_input = tokenizer([""], return_tensors='pt', padding=True, truncation=True).to(device)

        empty_outputs = model(**empty_input)


        empty_review_embedding = empty_outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    print(f"Empty review embedding calculated with shape: {empty_review_embedding.shape}")



    processed_texts = []
    print("Preprocessing all review texts with word_index filtering...")

    for i in tqdm(data_train.values):

        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))
        if filename == "Yelp2013":
            str_review = clean_str(str_review)

        if len(str_review.strip()) == 0:
            str_review = "<unk>"


        str_review_process = ' '.join([w for w in str_review.split() if w in vocab])
        review_words = str_review_process.strip().split()

        if len(review_words) > setNum_word_of_rev:
            review_words = review_words[:setNum_word_of_rev]


        processed_review_text = " ".join(review_words)
        processed_texts.append(processed_review_text)



    all_embeddings = []
    print(f"Encoding texts in batches of {batch_size}...")
    for i in tqdm(range(0, len(processed_texts), batch_size)):
        batch_texts = processed_texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=model.config.max_position_embeddings
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())


    edge_review_embedding_directed = np.vstack(all_embeddings)
    return edge_review_embedding_directed, processed_texts, empty_review_embedding


def construct_bert_reviews_dict(data_train, edge_review_embedding_directed, edge_aux_review_embedding_directed):
    print(f"{now()} Grouping original and auxiliary BERT embeddings by user and item...")


    user_reviews_bert_dict = defaultdict(list)
    item_reviews_bert_dict = defaultdict(list)


    user_aux_reviews_bert_dict = defaultdict(list)
    item_aux_reviews_bert_dict = defaultdict(list)



    for idx, row in enumerate(tqdm(data_train.values, desc="Grouping all embeddings")):
        uid = row[0]
        iid = row[1]


        original_embedding = edge_review_embedding_directed[idx]
        user_reviews_bert_dict[uid].append(original_embedding)
        item_reviews_bert_dict[iid].append(original_embedding)


        aux_embedding = edge_aux_review_embedding_directed[idx]
        user_aux_reviews_bert_dict[uid].append(aux_embedding)
        item_aux_reviews_bert_dict[iid].append(aux_embedding)

    return user_reviews_bert_dict, item_reviews_bert_dict, user_aux_reviews_bert_dict, item_aux_reviews_bert_dict


def padding_bert_embeddings(reviews_bert_dict, total_entities, max_reviews, pad_embedding):
    print(f"{now()} Padding/truncating BERT embeddings to create final array...")

    final_embeddings_list = []

    for i in tqdm(range(total_entities), desc="Padding entities"):

        embeddings = reviews_bert_dict.get(i, [])

        if len(embeddings) >= max_reviews:

            padded_embeddings = embeddings[:max_reviews]
        else:

            num_to_pad = max_reviews - len(embeddings)
            padding = [pad_embedding] * num_to_pad
            padded_embeddings = embeddings + padding

        final_embeddings_list.append(padded_embeddings)


    return np.array(final_embeddings_list, dtype=np.float32)


if __name__ == '__main__':
    start_time = time.time()





    filename = "Amazon_Instant_Video_5.json"





    number_of_fold = 67500

    save_folder = make_save_folder(filename, number_of_fold)


    print(f"{now()}: Step1: loading raw review datasets...")

    data_review_frame = construct_pandas_data_frame_from_file(filename)


    print("original user nums", len(set(list(data_review_frame['user_id']))))
    print("original item nums", len(set(list(data_review_frame['item_id']))))

    data_category_frame, category_num = construct_pandas_data_category_frame_from_file(filename, data_review_frame)

    item_in_category = set(list(data_category_frame['item_id']))

    data_review_frame = data_review_frame[data_review_frame['item_id'].isin(item_in_category)]

    data = pd.merge(data_review_frame, data_category_frame, how='left',
                    on='item_id')

    print("after merge item nums", len(set(list(data['item_id']))))



    userNum_all, itemNum_all = get_number_of_users_in_data(data)


    data, item_category_dict = numerize(data)


    print(f"{now()} Step2: split datasets into train/val/test, save into npy data")

    data_train, data_test = train_test_split(data, test_size=0.2,random_state=1234)
    data_train, data_test = check_users_in_train_data(data_train, data_test, userNum_all, itemNum_all)
    data_test, data_val = train_test_split(data_test, test_size=0.5, random_state=1234)

    x_train, y_train, x_val, y_val, x_test, y_test = construct_user_item_pair_rating(data_train, data_val,
                                                                                     data_test)

    ui_src, ui_dst, ui_node_map_index = construct_ui_graph_from_train_data(data_train, userNum_all,
                                                                           itemNum_all)

    save_npy_to_save_folder(save_folder, x_train, y_train, x_val, y_val, x_test, y_test, ui_src, ui_dst,
                            ui_node_map_index)

    print(f"{now()} Step3: Construct the vocab and user/item reviews from training set.")
    word_index, user_review2doc, item_review2doc, user_reviews_dict, item_reviews_dict, \
        user_summaries_dict, item_summaries_dict, user_iid_dict, item_uid_dict, \
        index_num_map_word_index, user_i_cateid_dict, item_u_cateid_dict, \
        user_rate_dict, item_rate_dict, user_revidx_dict, item_revidx_dict, vocab\
        = construct_reviews_dict_and_iid_dict_from_train_data(data_train,
                                                              filename,item_category_dict)









    setNum_rev_of_user, setNum_word_of_rev_user = countNum(user_reviews_dict)
    setNum_rev_of_item, setNum_word_of_rev_item = countNum(item_reviews_dict)

    setNum_word_of_rev = max(setNum_word_of_rev_user, setNum_word_of_rev_item)



    setNum_summary_of_user, setNum_word_of_summary_user = countNum(user_summaries_dict)
    setNum_summary_of_item, setNum_word_of_summary_item = countNum(item_summaries_dict)

    setNum_word_of_summary = max(setNum_word_of_summary_user, setNum_word_of_summary_item)

    print("保留用户评论个数为{}, 保留物品评论个数为{}, 保留用户和物品共同的单个评论单词数为{}, 保留用户和物品共同的单个摘要单词数为{}".format(setNum_rev_of_user, setNum_rev_of_item,
                                                                  setNum_word_of_rev, setNum_word_of_summary))













    print(f"{now()} Step5: padding all the text and id lists and save into npy.")











    userReview2Index = []
    userSummary2Index = []
    user_StrangerReview2Index = []

    userDoc2Index = []
    user_iid_list = []

    user_i_cateid_list = []
    rev_related_aux_rev_indices_list = [-1 for _ in range(len(x_train))]









    for i in range(userNum_all):
        count_user = 0
        dataList = []

        textList = user_reviews_dict[i]
        u_iids = user_iid_dict[i]
        u_i_cateids = user_i_cateid_dict[i]
        u_reviewList = []

        user_iid_list.append(padding_ids(u_iids, setNum_rev_of_user, itemNum_all + 1))
        user_i_cateid_list.append(padding_ids(u_i_cateids, setNum_rev_of_user, category_num + 1))

        doc2index = [word_index[w] for w in user_review2doc[i]]


        u_stranger_reviewList = []
        u_ratings = user_rate_dict[i]
        u_revidxList = user_revidx_dict[i]

        for textList_idx, text in enumerate(textList):
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < setNum_word_of_rev:
                text2index = text2index + [0] * (setNum_word_of_rev - len(text2index))
            else:
                text2index = text2index[:setNum_word_of_rev]
            u_reviewList.append(text2index)


            current_iid = u_iids[textList_idx]
            current_rating = u_ratings[textList_idx]
            own_review = textList[textList_idx]
            own_review_id = u_revidxList[textList_idx]
            stranger_uids = item_uid_dict[current_iid]
            stranger_reviews = item_reviews_dict[current_iid]
            stranger_ratings = item_rate_dict[current_iid]
            stranger_reviews_ids = item_revidx_dict[current_iid]


            same_rating_reviews = []
            plus_one_rating_reviews = []
            minus_one_rating_reviews = []


            for stranger_id, stranger_rating, stranger_review, stranger_review_id in \
                    zip(stranger_uids, stranger_ratings, stranger_reviews, stranger_reviews_ids):

                if stranger_id == i:
                    continue

                review_pair = (stranger_review_id, stranger_review)

                if stranger_rating == current_rating:
                    same_rating_reviews.append(review_pair)

                elif stranger_rating == current_rating + 1:
                    plus_one_rating_reviews.append(review_pair)

                elif stranger_rating == current_rating - 1:
                    minus_one_rating_reviews.append(review_pair)


            stranger_review_id = None
            stranger_text = ""

            if same_rating_reviews:
                stranger_review_id, stranger_text = random.choice(same_rating_reviews)
            elif plus_one_rating_reviews:
                stranger_review_id, stranger_text = random.choice(plus_one_rating_reviews)
            elif minus_one_rating_reviews:
                stranger_review_id, stranger_text = random.choice(minus_one_rating_reviews)
            else:
                stranger_review_id = own_review_id
                stranger_text = own_review


            rev_related_aux_rev_indices_list[u_revidxList[textList_idx]] = stranger_review_id





            strangertext2index = []
            wordTokens = stranger_text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            strangertext2index = [word_index[w] for w in wordTokens]
            if len(strangertext2index) < setNum_word_of_rev:
                strangertext2index = strangertext2index + [0] * (setNum_word_of_rev - len(strangertext2index))
            else:
                strangertext2index = strangertext2index[:setNum_word_of_rev]
            u_stranger_reviewList.append(strangertext2index)



        userReview2Index.append(padding_text(u_reviewList, setNum_rev_of_user))
        user_StrangerReview2Index.append(padding_text(u_stranger_reviewList, setNum_rev_of_user))


        summaryList = user_summaries_dict[i]
        u_summaryList = []

        for text in summaryList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < setNum_word_of_summary:
                text2index = text2index + [0] * (setNum_word_of_summary - len(text2index))
            else:
                text2index = text2index[:setNum_word_of_summary]
            u_summaryList.append(text2index)

        userSummary2Index.append(padding_text(u_summaryList, setNum_rev_of_user))



        userDoc2Index.append(doc2index)

    userDoc2Index = padding_doc(userDoc2Index, DOC_LEN)







    itemReview2Index = []
    itemSummary2Index = []
    itemDoc2Index = []
    item_uid_list = []

    item_u_cateid_list = []

    for i in range(itemNum_all):
        count_item = 0
        dataList = []

        textList = item_reviews_dict[i]
        i_uids = item_uid_dict[i]
        i_u_cateids = item_u_cateid_dict[i]
        i_reviewList = []

        item_uid_list.append(padding_ids(i_uids, setNum_rev_of_item, userNum_all + 1))
        item_u_cateid_list.append(padding_ids(i_u_cateids, setNum_rev_of_item, category_num + 1))

        doc2index = [word_index[w] for w in item_review2doc[i]]

        for text in textList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < setNum_word_of_rev:
                text2index = text2index + [0] * (setNum_word_of_rev - len(text2index))
            else:
                text2index = text2index[:setNum_word_of_rev]
            i_reviewList.append(text2index)

        itemReview2Index.append(padding_text(i_reviewList, setNum_rev_of_item))


        summaryList = item_summaries_dict[i]
        i_summaryList = []

        for text in summaryList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < setNum_word_of_summary:
                text2index = text2index + [0] * (setNum_word_of_summary - len(text2index))
            else:
                text2index = text2index[:setNum_word_of_summary]
            i_summaryList.append(text2index)

        itemSummary2Index.append(padding_text(i_summaryList, setNum_rev_of_item))


        itemDoc2Index.append(doc2index)

    itemDoc2Index = padding_doc(itemDoc2Index, DOC_LEN)


    print("153")
    edge_rating_idx_directed = extract_rating_idx_directed_array(data_train)








    device = "cuda"
    model_name = 'E:/zahuo/Beacon/tranger_final_for_zenodo/pro_data/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    print("154")
    edge_review_embedding_directed, review_list_for_bertopic, empty_review_embedding = extract_review_embeddings_batched(
        data_train=data_train,
        vocab=vocab,
        setNum_word_of_rev=setNum_word_of_rev,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=128
    )












    print("155")
    print(edge_review_embedding_directed.shape)
    edge_aux_review_embedding_directed = edge_review_embedding_directed[rev_related_aux_rev_indices_list]


    print(f"{now()} Step6: Generating user/item specific BERT review arrays.")


    user_reviews_bert_dict, item_reviews_bert_dict,\
    user_aux_reviews_bert_dict, item_aux_reviews_bert_dict = construct_bert_reviews_dict(
        data_train,
        edge_review_embedding_directed,
        edge_aux_review_embedding_directed
    )


    embedding_dim = edge_review_embedding_directed.shape[1]



    user_reviews_bert = padding_bert_embeddings(
        reviews_bert_dict=user_reviews_bert_dict,
        total_entities=userNum_all,
        max_reviews=setNum_rev_of_user,
        pad_embedding=empty_review_embedding
    )

    item_reviews_bert = padding_bert_embeddings(
        reviews_bert_dict=item_reviews_bert_dict,
        total_entities=itemNum_all,
        max_reviews=setNum_rev_of_item,
        pad_embedding=empty_review_embedding
    )



    user_aux_reviews_bert = padding_bert_embeddings(
        reviews_bert_dict=user_aux_reviews_bert_dict,
        total_entities=userNum_all,
        max_reviews=setNum_rev_of_user,
        pad_embedding=empty_review_embedding
    )

    item_aux_reviews_bert = padding_bert_embeddings(
        reviews_bert_dict=item_aux_reviews_bert_dict,
        total_entities=itemNum_all,
        max_reviews=setNum_rev_of_item,
        pad_embedding=empty_review_embedding
    )

    npy_to_save_folder(save_folder, userReview2Index, user_iid_list, userDoc2Index,
                      itemReview2Index, item_uid_list, itemDoc2Index, word_index,
                       index_num_map_word_index, userSummary2Index, itemSummary2Index,
                       user_i_cateid_list, item_u_cateid_list, user_StrangerReview2Index,
                       edge_rating_idx_directed, edge_review_embedding_directed,
                       edge_aux_review_embedding_directed,review_list_for_bertopic,
                       user_reviews_bert, item_reviews_bert,
                       user_aux_reviews_bert, item_aux_reviews_bert)


    print(f"{now()} Step6: start word embedding mapping...")
    w2v = construct_word_emb_from_glove_txt(PRE_W2V_BIN_PATH, word_index)

    np.save(f"{save_folder}/train/w2v.npy", w2v)

    end_time = time.time()
    print(f"{now()} all steps finised, cost time: {end_time - start_time:.4f}s")
