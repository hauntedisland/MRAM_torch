import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
import torch
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

def generate_kg_batch(kg_dict, batch_size, highest_neg_idx):
    exist_heads = kg_dict.keys()
    if batch_size <= len(exist_heads):
        batch_head = random.sample(exist_heads, batch_size)
    else:   # batch_size > exist_heads
        batch_head = [random.choice(list(exist_heads)) for _ in range(batch_size)]

    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for h in batch_head:
        relation, pos_tail = sample_pos_triples_for_h(kg_dict, h, 1)
        batch_relation += relation
        batch_pos_tail += pos_tail

        neg_tail = sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
        batch_neg_tail += neg_tail

    batch_head = torch.LongTensor(batch_head)
    batch_relation = torch.LongTensor(batch_relation)
    batch_pos_tail = torch.LongTensor(batch_pos_tail)
    batch_neg_tail = torch.LongTensor(batch_neg_tail)
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


def sample_pos_triples_for_h(kg_dict, head, n_sample_pos_triples):
    pos_triples = kg_dict[head]
    n_pos_triples = len(pos_triples)

    sample_relations, sample_pos_tails = [], []
    while True:
        if len(sample_relations) == n_sample_pos_triples:
            break

        pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
        tail = pos_triples[pos_triple_idx][0]
        relation = pos_triples[pos_triple_idx][1]

        if relation not in sample_relations and tail not in sample_pos_tails:
            sample_relations.append(relation)
            sample_pos_tails.append(tail)
    return sample_relations, sample_pos_tails


def sample_neg_triples_for_h(kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
    pos_triples = kg_dict[head]

    sample_neg_tails = []
    while True:
        if len(sample_neg_tails) == n_sample_neg_triples:
            break

        tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
        if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
            sample_neg_tails.append(tail)
    return sample_neg_tails


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
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)
    hd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])
        hd[h_id].append([t_id, r_id])

    return ckg_graph, rd, hd


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
    mean_mat = mean_mat.tocsr().tocoo()
    return mean_mat


def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory + 'kg.txt')

    print('building the graph ...')
    graph, relation_dict, kg_dict = build_graph(train_cf, triplets)      # graph只包含KG

    print('building the adj mat ...')
    adj_mean_mat = build_adj_matrix(relation_dict)

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

    return train_cf, test_cf, user_dict, kg_dict, triplets, n_params, graph, adj_mean_mat

