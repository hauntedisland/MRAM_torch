import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv, GCNConv, MessagePassing
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops, degree
from torch_geometric.utils import softmax as scatter_softmax
from torch_scatter import scatter_sum

import math
from einops import rearrange, repeat

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
        edge_attn_score = scatter_softmax(edge_attn_score, head)
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
                rel_attn = scatter_softmax(rel_attn, head_r)
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

class R_GraphConv(nn.Module):
    def __init__(self, emb_size, edge_index, edge_type, conv_layers, n_users, n_items, n_relations):
        super(R_GraphConv, self).__init__()
        self.in_channel = emb_size
        self.out_channel = emb_size
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.layer = conv_layers  # encode layer
        self.n_users = n_users
        self.n_items = n_items
        self.n_relation = n_relations
        self.hidden_channel = 128

        self.convs_layers = nn.ModuleList()
        # 只做一层
        self.convs_layers.append(RGCNConv(self.in_channel, self.out_channel, self.n_relation))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, all_emb):
        x = all_emb
        for i in range(self.layer - 1):
            x = self.convs_layers[i](x, self.edge_index, self.edge_type)  # node emb include entity
            x = self.activation(x)
            x = self.dropout(x)
        #最后一层不激活，dropout
        x = self.convs_layers[-1](x, self.edge_index, self.edge_type)
        out = x

        # user_emb = out[:self.n_users, :]
        item_emb = out[:self.n_items, :]

        # weights = [conv.weight for conv in self.convs_layers]  # 列表，每个元素形状为 [num_relations, input_dim, output_dim]
        # weights = torch.stack(weights, dim=0)  # [num_layers, num_relations, input_dim, output_dim]
        # # 对多层 weight 取平均
        # avg_weight = weights.mean(dim=0)  # [num_relations, input_dim, output_dim]
        # r_emb = avg_weight.mean(dim=1)  # [num_relations, output_dim]

        # TODO: relation emb: 选取最后一层
        weights = [conv.weight for conv in self.convs_layers]
        final_weight = weights[-1]  # [num_relations, in_dim, out_dim]

        relation_emb = final_weight.mean(dim=1)  # [num_relations, out_dim]

        return item_emb, relation_emb


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

class GraphConv(nn.Module):
    def __init__(self, edge_index, edge_weight, layer, n_users, n_items, channel):
        super(GraphConv, self).__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        self.num_conv_layers = layer
        self.n_users = n_users
        self.n_items = n_items
        self.in_channel = channel

        # 图卷积层，输入输出维度相同以支持残差连接
        self.convs_layers = nn.ModuleList([
            GCNConv(channel, channel) for _ in range(layer)
        ])

        # 激活函数和Dropout
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.5)

        # 批量归一化层
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(channel) for _ in range(layer)
        ])

        self.initial_bn = nn.BatchNorm1d(channel)

        # 可学习的层权重，用于加权平均
        self.weights = nn.Parameter(torch.ones(layer + 1))

        # 参数初始化
        for conv in self.convs_layers:
            nn.init.xavier_uniform_(conv.lin.weight.data)
            if conv.lin.bias is not None:
                conv.lin.bias.data.fill_(0.0)

    def forward(self, user_emb, item_emb):
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = []
        embs = [all_emb]
        embs = [self.initial_bn(all_emb)]

        for i in range(self.num_conv_layers):
            residual = all_emb  # 残差连接
            # 图卷积
            all_emb = self.convs_layers[i](all_emb, self.edge_index, self.edge_weight)
            # 残差相加
            all_emb += residual
            # 批量归一化
            all_emb = self.bn_layers[i](all_emb)
            # 激活函数和Dropout（最后一层不应用）
            if i != self.num_conv_layers - 1:
                all_emb = self.activation(all_emb)
                all_emb = self.dropout(all_emb)
            embs.append(all_emb)

        # 可学习的加权平均
        embs = torch.stack(embs, dim=1)  # [num_nodes, num_layers, channel]
        weights = F.softmax(self.weights, dim=0)
        light_out = torch.einsum('nlc,l->nc', embs, weights)
        # light_out = torch.mean(embs, dim=1)

        return light_out[:self.n_users], light_out[self.n_users:]

class GraphConv2(nn.Module):
    def __init__(self, edge_index, edge_weight, layer, n_users, n_items, channel):
        super(GraphConv2, self).__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.layer = layer
        self.n_users = n_users
        self.n_items = n_items
        self.in_channel = channel
        self.out_channel = channel
        self.hidden_channel = 64
        # self.W = nn.Parameter(init(torch.empty(size=(64, 64))))

        self.convs_layers = nn.ModuleList([
            GCNConv(in_channels=channel, out_channels=channel) for _ in range(self.layer)
        ])

    def forward(self, user_emb, item_emb):
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        #1. 不使用第0层emb
        embs = []
        #2. 使用简单的[all_emb] 0.1101
        embs = [all_emb]
        #3. 添加归一化的初始嵌入 0.1292
        embs = [F.normalize(all_emb, p=2, dim=1)]

        for i in range(self.layer):
            all_emb = self.convs_layers[i](all_emb, self.edge_index, self.edge_weight)
            norm_emb = F.normalize(all_emb, p=2, dim=1)
            embs.append(norm_emb)

        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        # out = self.convs_layers[-1](all_emb, self.edge_index, self.edge_weight)
        # out = self.down_proj(out)
        return out[:self.n_users], out[self.n_users:]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Disentangle(nn.Module):
    def __init__(self, adj_mat, ui_mat, iu_mat, channel, n_users, n_items, n_intent, n_relation, layer, ind, tmp=0.2,
                 k=4, save_path=None):
        super(Disentangle, self).__init__()
        self.adj = adj_mat  # 没有归一化的. coo_tensor
        self.ui_mat = ui_mat  # coo_tensor. [n_user, n_item]
        self.iu_mat = iu_mat  # coo_tensor. [n_item, n_user]
        self.n_users = n_users
        self.n_items = n_items

        self.n_intent = n_intent
        self.n_relation = n_relation
        self.emb_size = channel
        self.convs = layer
        self.ind = ind  # cosine
        self.temperature = tmp
        # disentangle_weight_att = init(torch.empty(self.n_intent, self.n_relation))
        # self.disentangle_weight_att = nn.Parameter(disentangle_weight_att)

        Q = init(torch.empty(self.n_intent, channel))
        self.Q = nn.Parameter(Q)
        # concat intent emb, mlp
        # self.mlp = nn.ModuleList()
        # for _ in range(self.n_intent):
        #     self.mlp.append(nn.Linear(channel*2, channel))
        # self.weight_nets = nn.ModuleList(
        #     [nn.Sequential(
        #         nn.Linear(channel * 2, channel),
        #         nn.ReLU(),
        #         nn.Linear(channel, channel)
        #     ) for _ in range(self.n_intent)]
        # )
        # EGLN
        self.topk = k
        self.W1 = nn.Parameter(init(torch.empty(channel, 16)))
        self.W2 = nn.Parameter(init(torch.empty(channel, 16)))
        self.save_path = save_path

    def forward(self, user_emb, item_emb, r_emb):
        """relation-intent attention"""
        # KGIN
        # intent_emb = torch.mm(nn.Softmax(dim=-1)(self.disentangle_weight_att), r_emb)  # [intent, emb]
        attention_scores = torch.matmul(self.Q, r_emb.T)  # [intent, relation]
        # attention_weights = F.softmax(attention_scores, dim=1) # 行归一化
        col_sum = torch.sum(attention_scores, dim=0, keepdim=True)  # 线性归一化
        attention_weights = attention_scores / col_sum
        intent_emb = torch.matmul(attention_weights, r_emb)

        """KGIN方式计算intent-aware embeddings"""
        # score_ = torch.mm(user_emb, intent_emb.t())  # [n_user, intent]
        # # 线性归一化
        # col = torch.sum(score_, dim=1, keepdim=True)
        # score = (score_ / col).unsqueeze(-1)
        # score = nn.Softmax(dim=1)(score_).unsqueeze(-1) # [n_user, intent, 1]
        # disen_weight = intent_emb.expand(self.n_users, self.n_intent, self.emb_size)    # [n_user, intent, 64]
        # user_int_emb = user_emb.unsqueeze(1).expand(-1, self.n_intent, -1) * (disen_weight * score)
        #
        # score1_ = torch.mm(item_emb, intent_emb.t())  # [n_item, intent]
        # score1 = nn.Softmax(dim=1)(score1_).unsqueeze(-1)   # [n_item, intent, 1]
        # disen_weight1 = intent_emb.expand(self.n_items, self.n_intent, self.emb_size)
        # item_int_emb = item_emb.unsqueeze(1).expand(-1, self.n_intent, -1) * (disen_weight1 * score1)

        """intent+node,扰动"""
        user_int_list, item_int_list = [], []
        for x in range(self.n_intent):
            int_emb = intent_emb[x, :]
            user_int = user_emb * int_emb
            item_int = item_emb * int_emb
            user, item = self.EGLN(user_int, item_int)
            user_int_list.append(user)
            item_int_list.append(item)

        user_int_emb = torch.cat(user_int_list, dim=1)
        item_int_emb = torch.cat(item_int_list, dim=1)

        assert user_int_emb.shape == (self.n_users, self.emb_size * self.n_intent)
        assert item_int_emb.shape == (self.n_items, self.emb_size * self.n_intent)

        return user_int_emb, item_int_emb, user_int_list, item_int_list

    def nor_sparse_matrix(self, sparse_matrix):
        sparse_matrix = sparse_matrix.coalesce()
        # 计算每一行的和
        row_sum = torch.sparse.sum(sparse_matrix, dim=1).to_dense()
        # 处理除零问题
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        # 归一化
        normalized_values = sparse_matrix.values() / row_sum[sparse_matrix.indices()[0]]
        return torch.sparse_coo_tensor(sparse_matrix.indices(), normalized_values, sparse_matrix.size())

    def sparse_add(self, sparse_a, sparse_b):
        sparse_a = sparse_a.coalesce()
        sparse_b = sparse_b.coalesce()
        # 合并两个稀疏矩阵的索引和值
        indices = torch.cat([sparse_a.indices(), sparse_b.indices()], dim=1)
        values = torch.cat([sparse_a.values(), sparse_b.values()])

        # 对相同索引的值求和
        unique_indices, inverse_indices = torch.unique(indices, dim=1, return_inverse=True)
        summed_values = torch.zeros_like(unique_indices[0], dtype=values.dtype)
        summed_values.scatter_add_(0, inverse_indices, values)

        # 构建新的稀疏矩阵
        return torch.sparse_coo_tensor(unique_indices, summed_values, sparse_a.size())

    def EGLN(self, user_emb, item_emb):
        # projection
        user_emb1 = torch.mm(user_emb, self.W1)
        item_emb1 = torch.mm(item_emb, self.W2)
        # normalize embedding
        user_rep = F.normalize(user_emb1, p=2, dim=1)
        item_rep = F.normalize(item_emb1, p=2, dim=1)
        # u-i
        sim_matrix = torch.sigmoid(torch.mm(user_rep, item_rep.T))  # [n_user, n_item]
        # save sim matrix
        # if self.save_path:
        #     sim_matrix_df = pd.DataFrame(sim_matrix.detach().cpu().numpy())
        #     sim_matrix_df.to_csv(self.save_path + f'sim_matrix {self.n_intent}.csv', index=False)
        # loss in EGLN
        # loss_simi_adj = 0.1 * torch.mean((sim_matrix - self.ui_mat) ** 2)  # eq11
        # select topk for subgraph construction
        user_topk_values, user_topk_indices = torch.topk(sim_matrix, self.topk, dim=1)
        user_topk_values = user_topk_values.view(-1)  # [n_user*topk]

        # new: rank topk, select some of them
        combined = list(
            zip(user_topk_values.tolist(), user_topk_indices[:, 0].tolist(), user_topk_indices[:, 1].tolist()))
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
        filtered_combined = sorted_combined[:1500]  # TODO: change
        filtered_values = torch.tensor([x[0] for x in filtered_combined])  # 相似度值
        filtered_indices = torch.tensor([[x[1], x[2]] for x in filtered_combined])  # 用户-物品对索引
        user_filtered_sparse_simi = torch.sparse_coo_tensor(filtered_indices.t(), filtered_values,
                                                       (self.n_users, self.n_items))
        # construct sparse simi
        # user_topk_columns = user_topk_indices.view(-1, 1).to(torch.int64)  # [n_user*topk, 1]
        # user_all_rows = torch.arange(self.n_users).view(-1, 1)  # [n_user, 1]
        # user_topk_rows = torch.tile(user_all_rows, (1, self.topk)).view(-1, 1)  # [n_user * topk, 1]
        # user_topk_indices = torch.cat([user_topk_rows, user_topk_columns], dim=1)
        # user_item_sparse_simi = torch.sparse_coo_tensor(user_topk_indices.t(), user_topk_values,
        #                                                 (self.n_users, self.n_items))


        # i-u
        item_topk_values, item_topk_indices = torch.topk(sim_matrix.t(), self.topk, dim=1)
        item_topk_values = item_topk_values.view(-1)
        combined = list(
            zip(item_topk_values.tolist(), item_topk_indices[:, 0].tolist(), item_topk_indices[:, 1].tolist()))
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
        filtered_combined = sorted_combined[:1500]  # TODO: change
        filtered_values = torch.tensor([x[0] for x in filtered_combined])  # 相似度值
        filtered_indices = torch.tensor([[x[1], x[2]] for x in filtered_combined])  # 用户-物品对索引
        item_filtered_sparse_simi = torch.sparse_coo_tensor(filtered_indices.t(), filtered_values,
                                                            (self.n_items, self.n_users))
        # item_topk_columns = item_topk_indices.view(-1, 1).to(torch.int64)
        # item_all_rows = torch.arange(self.n_items).view(-1, 1)
        # item_topk_rows = torch.tile(item_all_rows, (1, self.topk)).view(-1, 1)
        # item_topk_indices = torch.cat([item_topk_rows, item_topk_columns], dim=1)
        # item_user_sparse_simi = torch.sparse_coo_tensor(item_topk_indices.t(), item_topk_values,
        #                                                 (self.n_items, self.n_users))

        # A(E) = A(R) + A. add edges
        # add_sparse_user_matrix = self.sparse_add(self.ui_mat, user_item_sparse_simi)
        # add_sparse_item_matrix = self.sparse_add(self.iu_mat, item_user_sparse_simi)
        add_sparse_user_matrix = self.sparse_add(self.ui_mat, user_filtered_sparse_simi)
        add_sparse_item_matrix = self.sparse_add(self.iu_mat, item_filtered_sparse_simi)

        # 归一化
        # user_item_final_matrix = self.nor_sparse_matrix(add_sparse_user_matrix)
        # item_user_final_matrix = self.nor_sparse_matrix(add_sparse_item_matrix)
        user_item_final_matrix = self.nor_sparse_matrix(user_filtered_sparse_simi)
        item_user_final_matrix = self.nor_sparse_matrix(item_filtered_sparse_simi)

        # GNN
        final_user, final_item = self.EGLN_GNN(user_item_final_matrix, item_user_final_matrix, user_emb, item_emb,
                                               layer=self.convs)
        return final_user, final_item

    def EGLN_GNN(self, ui_mtx, iu_mtx, user_emb, item_emb, layer):
        user_emb_layer1 = torch.sparse.mm(ui_mtx, item_emb) + user_emb
        item_emb_layer1 = torch.sparse.mm(iu_mtx, user_emb) + item_emb
        user_emb_layer2 = torch.sparse.mm(ui_mtx, item_emb_layer1) + user_emb_layer1
        item_emb_layer2 = torch.sparse.mm(iu_mtx, user_emb_layer1) + item_emb_layer1
        user_emb_layer3 = torch.sparse.mm(ui_mtx, item_emb_layer2) + user_emb_layer2
        item_emb_layer3 = torch.sparse.mm(iu_mtx, user_emb_layer2) + item_emb_layer2
        user_emb_layer4 = torch.sparse.mm(ui_mtx, item_emb_layer3) + user_emb_layer3
        item_emb_layer4 = torch.sparse.mm(iu_mtx, user_emb_layer3) + item_emb_layer3
        if layer == 1:
            final_user_emb, final_item_emb = user_emb_layer1, item_emb_layer1
        elif layer == 2:
            final_user_emb, final_item_emb = user_emb_layer2, item_emb_layer2
        elif layer == 3:
            final_user_emb, final_item_emb = user_emb_layer3, item_emb_layer3
        elif layer == 4:
            final_user_emb, final_item_emb = user_emb_layer4, item_emb_layer4
        return final_user_emb, final_item_emb

    def GNN(self, all_emb, adj):
        embs = []  # remove layer0
        temp_emb = all_emb
        for i in range(self.convs):
            temp_emb = torch.sparse.mm(adj, temp_emb)
            embs.append(temp_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        user_emb, item_emb = torch.split(light_out, [self.n_users, self.n_items])
        return user_emb, item_emb


class MRAM(nn.Module):
    def __init__(self, data_config, args_config, kg_graph, adj_mat, adj_mean_mat):
        super(MRAM, self).__init__()
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
        self.relation_embed = nn.Parameter(nn.init.normal_(torch.empty(self.n_relations, self.emb_size), std=0.1))

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

        self.kg_encoder = R_GraphConv(self.emb_size, self.kg_edge_index, self.kg_edge_type, self.kg_encode_layer,
                                      self.n_users,
                                      self.n_items, self.n_relations)

        # 1. LightGCN
        self.encoder = LightGraphConv(self.adj_norm_tsr, self.encode_layer, self.n_users, self.n_items)

        # 2. GraphConv
        # self.encoder = GraphConv(self.edge_index, self.edge_weight, self.encode_layer, self.n_users, self.n_items,
        #                         channel=self.emb_size)

        # 3. GraphConv2
        # self.encoder = GraphConv2(self.edge_index, self.edge_weight, self.encode_layer, self.n_users, self.n_items,
        #                 channel=self.emb_size)

        self.decoder = Disentangle(self.adj_norm_tsr, self.ui_mat, self.iu_mat, self.emb_size, self.n_users, self.n_items,
                                   self.n_intent, self.n_relations,
                                   self.decode_layer, self.ind, k=self.k, save_path=args_config.data_path + args_config.dataset)


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
        编码：
        随机初始化u；
        RGCN: I, Wr选取最后一层的
        GCN：u,i"""
        kg_item_emb , r_emb = self.kg_encoder(self.entity_embed.weight)
        user_emb, item_emb = self.encoder(self.user_embed.weight, self.item_embed.weight)
        attention = KGAttention(self.emb_size)
        enhanced_item_emb = attention(item_emb, kg_item_emb)
        # user_int_emb, item_int_emb, user_int_list, item_int_list = self.decoder(user_emb, item_emb, r_emb)

        # return user_int_emb, item_int_emb, user_int_list, item_int_list
        # user_emb, item_emb = self.encoder(self.user_embed.weight, self.item_embed.weight)
        return user_emb, enhanced_item_emb


    def _get_kg_embedding(self, h, r, pos_t, neg_t):  # rectorch
        h_e = self.all_embed(h).unsqueeze(1)  # (kg_batch_size, 1, relation_dim)
        pos_t_e = self.all_embed(pos_t).unsqueeze(1)
        neg_t_e = self.all_embed(neg_t).unsqueeze(1)
        r_e = self.relation_emb(r)
        r_trans_w = self.trans_w(r).view(r.size(0), self.emb_size,
                                         self.kg_emb_size)  # (kg_batch_size, embed_dim, kg_dim)

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        """tempo: gcn only"""
        user_emb, item_emb = self._calculate_embedding()

        u_e = user_emb[user]
        pos_e, neg_e = item_emb[pos_item], item_emb[neg_item]

        mf_loss = self.create_bpr_loss_wo_cor(u_e, pos_e, neg_e)
        total_loss = mf_loss

        # user_int_emb, item_int_emb, user_int_list, item_int_list = self._calculate_embedding()

        # u_e = user_int_emb[user]
        # pos_e, neg_e = item_int_emb[pos_item], item_int_emb[neg_item]
        # mf_loss = self.create_bpr_loss_wo_cor(u_e, pos_e, neg_e)

        # batch_user_intents = torch.stack([emb[user] for emb in user_int_list], dim=1)   # [B, K, D]
        # batch_pos_intents  = torch.stack([emb[pos_item] for emb in item_int_list], dim=1)  # [B, K, D]

        # if not getattr(self, 'use_cl', True):
        #     cl_loss = 0.0
        # else:
        #     batch_user_intents = torch.zeros(1, device=self.device)  # 默认值
        #     batch_pos_intents = torch.zeros(1, device=self.device)   # 默认值
        #     cl_loss = self.calculate_cl_loss(batch_user_intents, batch_pos_intents)

        # total_loss = mf_loss

        return total_loss


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
        # return mf_loss
        return mf_loss + emb_loss + cor_loss

    def calculate_loss_transE(self, h, r, pos_t, neg_t):
        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = torch.nn.functional.softplus(pos_tail_score - neg_tail_score).mean()
        # kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        loss = kg_loss
        return loss

    def calculate_cl_loss(self, user_embeddings, item_embeddings):
        """
        user_embeddings: [B, K, D]  -> 每个用户在 K 个 intent 下的表征
        item_embeddings: [B, K, D]  -> 每个正样本物品在 K 个 intent 下的表征
        """
        # 1. 归一化
        user_embeddings = F.normalize(user_embeddings, dim=-1)
        item_embeddings = F.normalize(item_embeddings, dim=-1)

        sim_ui = torch.matmul(user_embeddings, rearrange(item_embeddings, 'b k d -> b d k')) / self.cl_temp
        sim_iu = torch.matmul(item_embeddings, rearrange(user_embeddings, 'b k d -> b d k')) / self.cl_temp

        labels = torch.arange(self.n_intent, device=user_embeddings.device)
        labels = repeat(labels, 'k -> b k', b=user_embeddings.size(0))  # [B, K]
        labels = rearrange(labels, 'b k -> (b k)')                       # [B*K]

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


class KGAttention(nn.Module):
    def __init__(self, emb_size):
        super(KGAttention, self).__init__()
        self.gate_net = nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.Sigmoid()
            )

    def forward(self, item_emb, kg_emb):
        gate_weights = self.gate_net(kg_emb)
        return item_emb * gate_weights