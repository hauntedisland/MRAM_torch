import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))

def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)
    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()
    n_nodes = max(n_nodes, max(max(triplets[:, 0]), max(triplets[:, 2])) + 1)   # CKG
    # n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    # n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1   # 从1开始计算，已经算上了ui interact
    return triplets     # np.array


def build_graph(train_data, triplets):
    kg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)
    
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    # for h_id, r_id, t_id in tqdm(ui_triplets, ascii=True):
    #     hd[h_id].append([t_id, r_id])   # kg dict
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        kg_graph.add_edge(h_id, t_id, key=r_id)
        if r_id != 0:
            rd[r_id].append([h_id, t_id])

    return kg_graph, rd

def build_single_adj(relation_dict):
    user_item_pairs = np.array(relation_dict[0])
    cf = user_item_pairs.copy()
    vals = [1.] * len(cf)

    ui_adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_users, n_items))
    iu_adj = sp.coo_matrix((vals, (cf[:, 1], cf[:, 0])), shape=(n_items, n_users))
    return ui_adj, iu_adj


def build_adj_matrix(relation_dict):
    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    print("Begin to build adjacent matrix ...")
    np_mat = np.array(relation_dict[0])     # UI only

    cf = np_mat.copy()
    
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    vals = [1.] * len(cf)
    # adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
    adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_users + n_items, n_users + n_items))
    mean_mat = _si_norm_lap(adj)

    # mean_mat = mean_mat.tocsr()[:n_users, n_users:].tocoo()
    mean_mat = mean_mat.tocsr().tocoo()
    return adj, mean_mat


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            # relation_adj_values = np.zeros(n_nodes, dtype=np.float32)
            # for pair in np_mat:
            #     head_id = pair[0]
            #     tail_id = pair[1]
            #     # 将头实体和尾实体对应的位置填充为1，表示有连接关系
            #     relation_adj_values[head_id] = 1
            #     relation_adj_values[tail_id] = 1
            cf = np_mat.copy()
            cf[:, 0] = cf[:, 0] + n_users
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * (len(cf) * 2)
            row = np.concatenate((cf[:, 0], cf[:, 1]), axis=0)
            col = np.zeros(len(row), dtype=np.int32)
            adj_r = sp.coo_matrix((vals, (row, col)), shape=(n_nodes, 1))
            # adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes)
            adj = sp.hstack([adj, adj_r])

    # build relation adj matrix R
    adj_kg_mat = adj.reshape((n_nodes, n_nodes + n_relations - 1))  # + KG relation (n_relations-1)
    adj_kg_mat_t = adj_kg_mat.transpose()  # (n_node+n_relations-1, n_node)
    # R^T
    # 获取行索引大于等于n_nodes的元素对应的索引和数据值
    relevant_indices = np.where(adj_kg_mat_t.row >= n_nodes)[0]
    new_row_indices = adj_kg_mat_t.row[relevant_indices] - n_nodes  # 调整行索引，使其从0开始（如果需要）
    new_col_indices = adj_kg_mat_t.col[relevant_indices]
    new_data = adj_kg_mat_t.data[relevant_indices]

    # 使用获取到的数据构建新的coo_matrix
    adj_r_t = sp.coo_matrix((new_data, (new_row_indices, new_col_indices)), shape=(n_relations - 1, n_nodes))
    z = sp.coo_matrix((n_relations-1, n_relations-1), dtype=np.float32)     # 边与边的连接关系：0
    down = sp.hstack([adj_r_t, z])
    # 把上下拼起来
    final_mat = sp.vstack([adj_kg_mat, down]).reshape((n_nodes+n_relations-1, n_nodes+n_relations-1))
    mean_mat = _si_norm_lap(final_mat)

    # concat A and R
    # norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    # mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    # norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    # mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    # return adj_mat_list, norm_mat_list, mean_mat_list
    return final_mat, mean_mat


def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('combining train_cf and kg data ...')

    # KG
    # kg_triplets = read_triplets(directory + 'kg.txt')
    # kg_graph, relation_dict = build_graph(train_cf, kg_triplets)
    print('building the graph ...')
    # CKG
    triplets = read_triplets(directory + 'triplets.txt')
    ckg_graph, relation_dict = build_graph(train_cf, triplets)
    # 
    
    print('building the adj mat ...')
    # ckg_mat, ckg_mean_mat = build_sparse_relational_graph(relation_dict)
    adj_mat, adj_mean_mat = build_adj_matrix(relation_dict)     # normalized ui mat
    ui_mat, iu_mat = build_single_adj(relation_dict)

    
    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    # return train_cf, test_cf, user_dict, n_params, kg_graph, [adj_mat, ui_mat, iu_mat], adj_mean_mat
    return train_cf, test_cf, user_dict, n_params, ckg_graph, [adj_mat, ui_mat, iu_mat], adj_mean_mat
