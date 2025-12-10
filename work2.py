import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv, GCNConv, MessagePassing
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops, degree
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_sum, scatter, scatter_softmax

import math
from einops import rearrange, repeat, einsum

# import dgl
# import dgl.function as fn

init = nn.init.xavier_uniform_

class CKGGCN(nn.Module):
    def __init__(self, dims, n_relations, edge_index, edge_type, layer, relation_emb):
        super(CKGGCN, self).__init__()
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.layer = layer
        self.relation_emb = nn.Parameter(relation_emb.data.clone())
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
        key = key * self.relation_emb[self.edge_type - 1].view(-1, self.n_heads, self.d_k)
        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_score = pyg_softmax(edge_attn_score, head)
        relation_emb = self.relation_emb[self.edge_type - 1]
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

        relation_update = torch.zeros_like(self.relation_emb)  # [n_relation, latdim]
        for r in range(self.n_relation):
            mask = (self.edge_type - 1) == r  # 筛选出当前关系的边
            if mask.sum() > 0:
            # edge_attn_score: [n_edge, n_head]
                head_r = head[mask]
                tail_r = tail[mask]
                rel_attn = (entity_agg[head_r] * self.relation_emb[r]).sum(dim=1)   # [n_edges_r]
                rel_attn = F.leaky_relu(rel_attn, 0.2)
                rel_attn = pyg_softmax(rel_attn, head_r)
                # relation_update[r] = (entity_agg[head_r] * rel_attn.view(-1, 1)).sum(dim=0)
                weighted_emb = entity_emb[tail_r] * rel_attn.unsqueeze(-1)  # [n_edges_r, dim]
                relation_update[r] = scatter_sum(weighted_emb, head_r, dim=0).mean(dim=0)
        relation_update = F.normalize(relation_update, p=2, dim=1)
        self.relation_emb.data = self.relation_emb.data + 0.1 * relation_update
        self.relation_emb.data = F.normalize(self.relation_emb.data, p=2, dim=1)
        return entity_agg, user_agg

    def forward(self, user_emb, entity_emb, inter_edge, inter_edge_w):
        user_embs = [user_emb]
        entity_embs = [entity_emb]
        for i in range(self.layer):
            entity_emb, user_emb= self._agg_layer(user_emb, entity_emb, inter_edge, inter_edge_w)
            user_embs.append(user_emb)
            entity_embs.append(entity_emb)
        user_embs = torch.mean(torch.stack(user_embs, dim=1), dim=1)
        entity_embs = torch.mean(torch.stack(entity_embs, dim=1), dim=1)
        return user_embs, entity_embs, self.relation_emb


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


"""
KG attention聚合与采样
"""
class AGGLayer(MessagePassing):
    def __init__(self, channel, topk=15, comp_op='add', bn=True):
        super(AGGLayer, self).__init__()
        self.topk = topk
        self.comp_op = comp_op
        self.dim = channel

        self.neigh_w = nn.Linear(channel, channel, bias=False)
        self.act = nn.Tanh()
        self.bn = nn.BatchNorm1d(channel) if bn else None

    def forward(self, x, edge_index, edge_type, relation_emb):
        return self.propagate(edge_index, x=x, edge_type=edge_type, rel_emb=relation_emb)

    def message(self, x_j, x_i, edge_type, rel_emb, index):
        """
        x_j: 邻居 [E, D], x_i: 中心 [E, D]
        index: target node id for each edge [E]
        """
        r_emb = rel_emb[edge_type]  # [E, D]

        # composition
        if self.comp_op == 'add':
            comp = x_j + r_emb
        elif self.comp_op == 'mul':
            comp = x_j * r_emb
        else:
            raise ValueError

        # attention score
        score = (comp * x_i).sum(dim=-1)  # [E]

        # softmax per target node
        alpha = scatter_softmax(score, index)  # 注意：index 在 propagate 时自动传入

        return comp, alpha, score

    def aggregate(self, inputs, index, dim_size=None):
        comp_msg, alpha, score = inputs
        if self.topk > 0:
            # TODO: 对每个中心节点采样
            topk_score, topk_idx = torch.topk(score, k=min(self.topk, score.size(0)), dim=0, sorted=False)
            # 构建 mask
            mask = torch.zeros(score.size(0), device=score.device)
            mask[topk_idx] = 1.0
            alpha = alpha * mask
            alpha = alpha / (alpha.sum(dim=0, keepdim=True) + 1e-12)  # renormalize

        # 加权消息
        weighted = comp_msg * alpha.unsqueeze(-1)
        out = scatter(weighted, index, dim=0, dim_size=dim_size, reduce='sum')
        return out

    def update(self, aggr_out):
        out = self.neigh_w(aggr_out)
        if self.bn:
            out = self.bn(out)
        out = self.act(out)
        return out


class Disentangle2(nn.Module):
    def __init__(self, adj_mat, edge_index, edge_type, channel, n_users, n_items, n_intent, n_relation, layer):
        super(Disentangle2, self).__init__()
        self.n_users = n_users
        self.n_items = n_items

        self.adj = adj_mat
        self.edge_index = edge_index
        self.edge_type = edge_type

        self.n_intent = n_intent
        self.n_relation = n_relation
        self.emb_size = channel
        self.convs = layer

        self.rel_embs = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_relation, self.emb_size) * 0.01)
            for _ in range(self.convs)
        ])
        # 意图映射矩阵
        self.L = nn.Linear(channel, channel, bias=False)
        self.S = nn.Linear(channel, channel, bias=False)

        self.agg_layers = nn.ModuleList([
            AGGLayer(channel, topk=15, comp_op='add', bn=True)
                for _ in range(layer)
            ])

        self.ent_drop = nn.Dropout(0.2)
        self.rel_drop = nn.Dropout(0.2)

    def compute_corr(self, x1, x2):
        # Subtract the mean
        x1_mean = torch.mean(x1, 0, True)
        x1 = x1 - x1_mean
        x2_mean = torch.mean(x2, 0, True)
        x2 = x2 - x2_mean

        # Compute the cross correlation
        sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
        sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
        corr = torch.abs(torch.mean(x1 * x2)) / (sigma1 * sigma2)

        return corr

    def forward(self, entity_emb):
        common = self.S(entity_emb)
        private = self.L(entity_emb)

        corr_total = 0.0

        for i, (layer, rel_emb) in enumerate(zip(self.agg_layers, self.rel_embs)):
            # dropout
            common = self.ent_drop(common)
            private = self.ent_drop(private)
            rel_emb = self.rel_drop(rel_emb)

            common_agg = layer(common, self.edge_index, self.edge_type, rel_emb)
            private_agg = layer(private, self.edge_index, self.edge_type, rel_emb)

            # 残差链接
            common = common + common_agg
            private = private + private_agg

            corr_total += self.compute_corr(common_agg, private_agg)

        corr = corr_total / self.convs
        return common, private, corr

class GatedEncoder(nn.Module):
    def __init__(self, emb_size):
        super(GatedEncoder, self).__init__()
        self.gate_net = nn.Sequential(
                # nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                # nn.Linear(emb_size, emb_size),
                nn.Sigmoid()
            )

    def forward(self, item_emb, kg_emb):
        gate_weights = self.gate_net(kg_emb)
        return item_emb * gate_weights

class WORK2(nn.Module):
    def __init__(self, data_config, args_config, kg_graph, adj_mat, adj_mean_mat):
        super(WORK2, self).__init__()
        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.ssm = args_config.ssm

        # self.mess_drop_rate = args_config.mess_dropout_rate
        # self.res = args_config.res_lambda

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']
        self.n_nodes = data_config['n_nodes']  # all nodes

        self.n_intent = args_config.n_intent
        self.emb_size = args_config.dim
        self.kg_emb_size = args_config.kg_dim
        self.kg_encode_layer = args_config.kg_encode_layer
        self.encode_layer = args_config.encode_layer
        self.decode_layer = args_config.decode_layer
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.ind = "cosine"
        self.k = args_config.topk
        self.adj_mat = adj_mat[0]  # coo matrix
        self.ui_mat = adj_mat[1]
        self.iu_mat = adj_mat[2]
        self.adj_norm = adj_mean_mat

        # self.ckg_edge_index, self.ckg_edge_type = self._get_edges(ckg_graph)
        self.kg_edge_index, self.kg_edge_type = self._get_edges(kg_graph)

        self.all_embed = torch.nn.Embedding(self.n_nodes, self.emb_size)
        self.user_embed = torch.nn.Embedding(self.n_users, self.emb_size)
        self.item_embed = torch.nn.Embedding(self.n_items, self.emb_size)
        self.entity_embed = torch.nn.Embedding(self.n_entities, self.emb_size)
        self.relation_embed = torch.nn.Embedding(self.n_relations, self.emb_size)

        # 用ui index
        edge_index, edge_weight = from_scipy_sparse_matrix(self.adj_norm)
        self.edge_index = edge_index.to(self.device)
        self.edge_weight = edge_weight.to(self.device)
        self.edge_weight = self.edge_weight.to(torch.float32)

        # self.adj_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.adj_norm_tsr = self._convert_sp_mat_to_sp_tensor(self.adj_norm).to(self.device)

        self.ui_mat = self._convert_sp_mat_to_sp_tensor(self.ui_mat).to(self.device)
        self.iu_mat = self._convert_sp_mat_to_sp_tensor(self.iu_mat).to(self.device)

        self.cl_temp = args_config.cl_temp
        self.cl_rate = args_config.cl_rate

        # self.kg_encoder = R_GraphConv(self.emb_size, self.kg_edge_index, self.kg_edge_type, self.kg_encode_layer,
        #                               self.n_users,
        #                               self.n_items, self.n_relations)

        # 1. LightGCN
        self.encoder = LightGraphConv(self.adj_norm_tsr, self.encode_layer, self.n_users, self.n_items)

        # 2. GraphConv
        # self.encoder = GraphConv(self.edge_index, self.edge_weight, self.encode_layer, self.n_users, self.n_items,
        #                         channel=self.emb_size)

        # 3. GraphConv2
        # self.encoder = GraphConv2(self.edge_index, self.edge_weight, self.encode_layer, self.n_users, self.n_items,
        #                 channel=self.emb_size)

        self.decoder = Disentangle2(self.adj_norm_tsr, self.kg_edge_index, self.kg_edge_type, self.emb_size, self.n_users, self.n_items,
                                   self.n_intent, self.n_relations, self.decode_layer)

        self.gated_encoder = GatedEncoder(self.emb_size)

    def _get_edges(self, graph):  # graph:[num_nodes, [h, t, r_id]]
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]. [h, t]
        type = graph_tensor[:, -1]  # [-1, 1]. r_id
        return index.t().long().to(self.device), type.long().to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)

    def _convert_sp_mat_to_edge_index(self, X):
        coo = X.tocoo()
        row = torch.from_numpy(coo.row).long()  # 行索引
        col = torch.from_numpy(coo.col).long()  # 列索引
        edge_index = torch.stack([row, col], dim=0)

        return edge_index

    def _calculate_embedding(self):
        """
        1. 对KG进行解耦，KGAT聚合、去噪，输出聚合后的解耦entity emb与cor loss
        2. 使用LightGCN聚合用户、物品emb
        3. gate方式对用户、物品解耦
        """
        # import pdb;pdb.set_trace()
        common, private, cor = self.decoder(self.entity_embed.weight)
        user_emb, item_emb = self.encoder(self.user_embed.weight, self.item_embed.weight)

        item_intent1 = self.gated_encoder(item_emb, common)
        item_intent2 = self.gated_encoder(item_emb, private)
        enhanced_item_emb = item_intent1 + item_intent2     # TODO:融合策略；用户解耦
        return user_emb, enhanced_item_emb, cor

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        # 2. with corr
        user_int_emb, item_int_emb, cor = self._calculate_embedding()
        u_e = user_int_emb[user]
        pos_e, neg_e = item_int_emb[pos_item], item_int_emb[neg_item]
        mf_loss = self.create_bpr_loss(u_e, pos_e, neg_e, cor)
        total_loss = mf_loss
        return total_loss, cor

        """使用正交loss"""
        # loss_orth_u = self.calculate_orthogonal_loss(user_int_emb)
        # loss_orth_i = self.calculate_orthogonal_loss(item_int_emb)
        # total_loss = mf_loss + 0.01 * (loss_orth_u + loss_orth_i)

        """使用CL损失"""
        # batch_user_intents = torch.stack([emb[user] for emb in user_int_list], dim=1)   # [B, K, D]
        # batch_pos_intents  = torch.stack([emb[pos_item] for emb in item_int_list], dim=1)  # [B, K, D]
        # cl_loss = self.calculate_cl_loss(batch_user_intents, batch_pos_intents)

        # total_loss = mf_loss + self.cl_rate * cl_loss
        # return total_loss


    def ssm_loss(self, users, pos_items):
        pos_user_norm = F.normalize(users)
        pos_item_norm = F.normalize(pos_items)

        pos_score = torch.sum(pos_user_norm * pos_item_norm, dim=1)
        neg_score = torch.matmul(pos_user_norm, pos_item_norm.t())

        pos_score = torch.exp(pos_score / 0.2)
        neg_score = torch.sum(torch.exp(neg_score / 0.2), dim=1)

        ssm_loss = (-1) * torch.log(pos_score / neg_score)
        ssm_loss = torch.mean(ssm_loss)

        return self.ssm * ssm_loss

    def create_bpr_loss_wo_cor(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # L2
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # return mf_loss
        return mf_loss + emb_loss

    def create_bpr_loss(self, users, pos_items, neg_items, cor):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # L2
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor
        return mf_loss + emb_loss + cor_loss

    def calculate_orthogonal_loss(self, embeddings):
        """
        计算嵌入矩阵的正交正则化损失
        embeddings: [N, K*D] 形状的嵌入矩阵，其中N是用户/物品数量，K是意图数量，D是每个意图的维度
        """
        K = self.n_intent
        reshaped_emb = rearrange(embeddings, 'n (k d) -> n k d', k=K)

        # 计算意图之间的相关性矩阵 [N, K, K]
        corr_matrix = torch.bmm(reshaped_emb, reshaped_emb.transpose(1, 2))  # [N, K, K]

        # 计算每个样本的意图间相关性
        # 不同意图之间的相关性接近0，相同意图的相关性接近1
        I = torch.eye(K, device=embeddings.device)
        orth_loss = ((corr_matrix - I) ** 2).sum(dim=(1,2)).mean()
        return orth_loss

    def calculate_cl_loss(self, user_embeddings, item_embeddings):
        """
        user_embeddings: [B, K, D]  -> 每个用户在 K 个 intent 下的表征
        item_embeddings: [B, K, D]  -> 每个正样本物品在 K 个 intent 下的表征
        """
        B, K, D = user_embeddings.shape
        # 1. 归一化
        user_embeddings = F.normalize(user_embeddings, dim=-1)
        item_embeddings = F.normalize(item_embeddings, dim=-1)

        sim_ui = torch.matmul(user_embeddings, rearrange(item_embeddings, 'b k d -> b d k')) / self.cl_temp
        sim_iu = torch.matmul(item_embeddings, rearrange(user_embeddings, 'b k d -> b d k')) / self.cl_temp

        # labels = torch.arange(self.n_intent, device=user_embeddings.device)
        # labels = repeat(labels, 'k -> b k', b=user_embeddings.size(0))  # [B, K]
        # labels = rearrange(labels, 'b k -> (b k)')                       # [B*K]
        labels = torch.arange(K, device=user_embeddings.device)  # [0,1,2,...,K-1]
        labels = labels.unsqueeze(0).expand(B, K)                # [B, K]
        labels = labels.contiguous().view(-1)                    # [B*K]

        # 每一行 (User_k) 需要从 K 个 Item Intent 中找到属于自己的那个 Item_k
        # logits: [B*K, K]
        logits_ui = rearrange(sim_ui, 'b k1 k2 -> (b k1) k2')
        logits_iu = rearrange(sim_iu, 'b k1 k2 -> (b k1) k2')

        loss_ui = F.cross_entropy(logits_ui, labels)
        loss_iu = F.cross_entropy(logits_iu, labels)

        return (loss_ui + loss_iu) / 2

    def generate(self):
        return self._calculate_embedding()[:2]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())


