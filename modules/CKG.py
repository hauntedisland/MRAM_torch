import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax as scatter_softmax
from torch_scatter import scatter_sum

import math

class CKGConv(nn.Module):
    def __init__(self, dims, n_relations, edge_index, edge_type, layer, relation_emb):
        super(CKGConv, self).__init__()
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.layer = layer
        self.relation_emb = relation_emb
        
        self.W_Q = nn.Parameter(nn.init.normal_(torch.Tensor(dims, dims), std=0.1))
        
        self.n_relation = n_relations
        self.n_heads = 2
        self.d_k = dims // self.n_heads

    def _agg_layer(self, user_emb, entity_emb, inter_edge, inter_edge_w):
        head, tail = self.edge_index
        head_emb = entity_emb[head]
        tail_emb = entity_emb[tail]
        
        # attention from entity to item/entity
        query = (head_emb @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (tail_emb @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = key * self.relation_emb(self.edge_type - 1).view(-1, self.n_heads, self.d_k)
        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_score = scatter_softmax(edge_attn_score, head)
        relation_emb = self.relation_emb(self.edge_type - 1)
        neigh_relation_emb = tail_emb * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)

        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        entity_agg_res = torch.zeros_like(entity_emb)
        entity_agg = entity_agg_res.index_add_(0, head, entity_agg)
        entity_agg = F.normalize(entity_agg)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]    # 交互边权重
        user_agg = torch.zeros_like(user_emb)
        user_agg = user_agg.index_add_(0, inter_edge[0, :], item_agg)

        return entity_agg, user_agg

    def forward(self, user_emb, entity_emb, inter_edge, inter_edge_w):
        user_embs = [user_emb]
        entity_embs = [entity_emb]
        for i in range(self.layer):
            entity_emb, user_emb = self._agg_layer(user_emb, entity_emb, inter_edge, inter_edge_w)
            user_embs.append(user_emb)
            entity_embs.append(entity_emb)
        user_embs = torch.mean(torch.stack(user_embs, dim=1), dim=1)
        entity_embs = torch.mean(torch.stack(entity_embs, dim=1), dim=1)
        return user_embs, entity_embs
    
class CKGGCN(nn.Module):
    def __init__(self, data_config, args_config, kg_graph, adj_mean_mat):
        super(CKGGCN, self).__init__()
        self.decay = args_config.l2
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']
        self.emb_size = args_config.dim
        self.kg_encode_layer = args_config.kg_encode_layer
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_norm = adj_mean_mat
        self.adj_norm = self._convert_sp_mat_to_sp_tensor(self.adj_norm).to(self.device)
        self.Graph = self.adj_norm.coalesce().to(self.device)
        self.inter_edge_w = self.Graph.values()[:self.Graph.values().shape[0] // 2]
        self.inter_edge = [self.Graph.indices()[0, :self.Graph.indices()[0].shape[0] // 2],
                           self.Graph.indices()[1, :self.Graph.indices()[0].shape[0] // 2] - self.n_users]
        self.inter_edge = torch.stack(self.inter_edge, dim=0)

        self.kg_edge_index, self.kg_edge_type = self._get_edges(kg_graph)

        self.user_embed = torch.nn.Embedding(self.n_users, self.emb_size)
        self.item_embed = torch.nn.Embedding(self.n_items, self.emb_size)
        self.entity_embed = torch.nn.Embedding(self.n_entities, self.emb_size)
        # self.relation_embed = nn.Parameter(nn.init.normal_(torch.empty(self.n_relations, self.emb_size), std=0.1))
        self.relation_embed = torch.nn.Embedding(self.n_relations, self.emb_size)
        
        self.ckgencoder = CKGConv(self.emb_size, self.n_relations, self.kg_edge_index, self.kg_edge_type, self.kg_encode_layer, self.relation_embed)

    def _get_edges(self, graph):  # graph:[num_nodes, [h, t, r_id]]
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]. [h, t]
        type = graph_tensor[:, -1]  # [-1, 1]. r_id
        return index.t().long().to(self.device), type.long().to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
        # return torch.sparse_coo_tensor(i, v, coo.shape)

    def _calculate_embedding(self):
        user_emb, item_emb = self.ckgencoder(self.user_embed.weight, torch.cat([self.item_embed.weight, self.entity_embed.weight]),
                                                       self.inter_edge, self.inter_edge_w)
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
        return self._calculate_embedding()[:2]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())