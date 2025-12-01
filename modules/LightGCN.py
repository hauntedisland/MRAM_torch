import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import RGCNConv, GCNConv

init = nn.init.xavier_uniform_

class LightGraphConv(nn.Module):
    def __init__(self, adj_mat, conv_layers, n_users, n_items):
        super(LightGraphConv, self).__init__()
        self.adj_mat = adj_mat
        self.convs = conv_layers  # encode layer
        self.n_users = n_users
        self.n_items = n_items

    def forward(self, user_emb, item_emb):
        # concat user emb and updated item emb
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        temp_emb = all_emb
        for i in range(self.convs):
            temp_emb = torch.sparse.mm(self.adj_mat, temp_emb)
            embs.append(temp_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out[:self.n_users], light_out[self.n_users:]


class LightGCN(nn.Module):
    def __init__(self, data_config, args_config, adj_mean_mat):
        super(LightGCN, self).__init__()
        self.decay = args_config.l2
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_nodes = data_config['n_nodes']  # all nodes
        self.emb_size = args_config.dim
        self.encode_layer = args_config.encode_layer
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_norm = adj_mean_mat

        self.user_embed = torch.nn.Embedding(self.n_users, self.emb_size)
        self.item_embed = torch.nn.Embedding(self.n_items, self.emb_size)

        # coo -> tensor
        self.adj_mat = self._convert_sp_mat_to_sp_tensor(self.adj_norm).to(self.device)
        self.encoder = LightGraphConv(self.adj_mat, self.encode_layer, self.n_users, self.n_items)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
        # return torch.sparse_coo_tensor(i, v, coo.shape)

    def _calculate_embedding(self):
        user_emb, item_emb = self.encoder(self.user_embed.weight, self.item_embed.weight)
        return user_emb, item_emb

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        
        user_emb, item_emb = self._calculate_embedding()
        u_e = user_emb[user]
        pos_e, neg_e = item_emb[pos_item], item_emb[neg_item]

        mf_loss = self.create_bpr_loss(u_e, pos_e, neg_e)
        return mf_loss

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # L2
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        return mf_loss

    def generate(self):
        return self._calculate_embedding()

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())
