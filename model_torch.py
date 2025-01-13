import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import RGCNConv, GCNConv

# import util.losses

init = nn.init.xavier_uniform_

# class RGAT(nn.Module):
#     def __init__(self, latdim, n_hops, mess_dropout_rate, n_item, n_relation):
#         super(RGAT, self).__init__()
#         self.mess_dropout_rate = mess_dropout_rate
#         self.W = nn.Parameter(init(torch.empty(size=(2 * latdim, latdim)), gain=nn.init.calculate_gain('relu')))
#
#         self.leakyrelu = nn.LeakyReLU(0.2)
#         self.n_hops = n_hops
#         self.dropout = nn.Dropout(p=mess_dropout_rate)
#         self.n_item = n_item
#
#         # projection from item space to relation space(?)
#         self.W2 = nn.Parameter(init(torch.empty(size=(64, 64)), gain=nn.init.calculate_gain('relu')))
#
#     def agg(self, entity_emb, relation_emb, kg):
#         edge_index, edge_type = kg
#         head, tail = edge_index
#         a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
#         # 将entity通过W矩阵映射到relation emb上. [head+tail, d] * [relation, d]
#         e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type - 1]).sum(
#             -1)
#         e = self.leakyrelu(e_input)
#         # 将head位置的向量(item)都转换为概率分布
#         e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
#         # 按概率聚合连接的entity，更新item. e本身就是一种概率分数，使用view(-1,1)
#         agg_emb = entity_emb[tail] * e.view(-1, 1)
#         # 将与每个head索引相关的所有tail索引对应的加权实体嵌入向量求和。
#         agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
#         # 1.计算relation -> item attention (学习一个映射矩阵)
#         # transformed_agg = torch.mm(agg_emb[:self.n_item, :], self.W2)   # 线性变换
#         # ir_att = F.softmax(torch.matmul(transformed_agg, relation_emb.t()), dim=1)     # attention on relation_dim. [item, relation]
#         # KGIN
#         score = F.softmax(torch.mm(relation_emb, agg_emb.t()), dim=1)  # (relation, item)
#         r_emb = torch.matmul(score, agg_emb)
#         # 2.用注意力更新relation
#         # r_emb = torch.matmul(ir_att, relation_emb)  # shape error.ir_att shape, how to multiply?
#         # agg_emb = agg_emb + entity_emb
#         return agg_emb, r_emb
#

#     def forward(self, entity_emb, relation_emb, kg, res_lambda, mess_dropout=True):
#         entity_res_emb = entity_emb
#         for _ in range(self.n_hops):
#             entity_emb, r_emb = self.agg(entity_emb, relation_emb, kg)
#             if mess_dropout:
#                 entity_emb = self.dropout(entity_emb)
#             entity_emb = F.normalize(entity_emb)
#
#             entity_res_emb = res_lambda * entity_res_emb + entity_emb
#         return entity_res_emb, r_emb


"""
    pass transE updated item embedding
    LightGCN[user || item embedding]
    return user, item embedding
"""


class R_GraphConv(nn.Module):
    def __init__(self, channel, edge_index, edge_type, conv_layers, n_users, n_items, n_relations):
        super(R_GraphConv, self).__init__()
        self.emb_size = channel
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.layer = conv_layers  # encode layer
        self.n_users = n_users
        self.n_items = n_items
        self.n_relation = n_relations
        self.hidden_channel = 128
        # self.fc = nn.Linear(channel * channel, channel)
        # self.W_r = nn.Parameter(torch.randn(self.n_relation, channel))

        self.convs_layers = nn.ModuleList()
        # layer1
        self.convs_layers.append(RGCNConv(channel, self.hidden_channel, self.n_relation))
        for _ in range(self.layer - 2):
            self.convs_layers.append(RGCNConv(self.hidden_channel, self.hidden_channel, self.n_relation))
        # last layer
        self.convs_layers.append(RGCNConv(self.hidden_channel, channel, self.n_relation))
        # for layer in range(conv_layers):
        #     self.convs_layers.append(RGCNConv(channel, channel, self.n_relation))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, all_emb):
        # embs = [all_emb]
        # for i in range(self.layer):
        #     all_emb = self.convs_layers[i](all_emb, self.edge_index, self.edge_type)  # node emb include entity
        #     all_emb = self.activation(all_emb)
        #     all_emb = self.dropout(all_emb)
        #     norm_emb = F.normalize(all_emb, p=2, dim=1)
        #     embs.append(norm_emb)
        #
        # embs = torch.stack(embs, dim=1)
        # out = torch.mean(embs, dim=1)
        x = all_emb
        for i in range(self.layer):
            x = self.convs_layers[i](x, self.edge_index, self.edge_type)  # node emb include entity
            x = self.activation(x)
            x = self.dropout(x)
        out = x
        user_emb = out[:self.n_users, :]
        item_emb = out[self.n_users:self.n_users + self.n_items, :]
        # TODO: relation emb: 平均每层weight
        # weights = [conv.weight for conv in self.convs_layers]  # 列表，每个元素形状为 [num_relations, input_dim, output_dim]
        # weights = torch.stack(weights, dim=0)  # [num_layers, num_relations, input_dim, output_dim]
        # # 对多层 weight 取平均
        # avg_weight = weights.mean(dim=0)  # [num_relations, input_dim, output_dim]
        # r_emb = avg_weight.mean(dim=1)  # [num_relations, output_dim]

        # TODO: relation emb: 选取最后一层
        weights = [conv.weight for conv in self.convs_layers]
        final_weight = weights[-1]  # [num_relations, in_dim, out_dim]
        r_emb = final_weight.mean(dim=1)  # [num_relations, out_dim]
        # 全连接降维
        # weight = avg_weight.view(self.n_relation, -1)
        # r_emb = self.fc(weight)
        # r_emb = r_emb.mean(dim=(1, 2))  # 降维: [num_relations, embedding_dim]
        return item_emb, r_emb
        # return user_emb, item_emb


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
    def __init__(self, adj_mat, layer, n_users, n_items, n_relations, channel):
        super(GraphConv, self).__init__()
        self.adj_mat = adj_mat  # edge index
        self.convs = layer  # encode layer
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_channel = 128
        self.W = nn.Parameter(init(torch.empty(size=(64, 64))))
        self.convs_layers = nn.ModuleList([
            pyg_nn.GCNConv(in_channels=channel, out_channels=channel) for _ in range(layer)
        ])
        self.conv = nn.Linear(64, 64)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # self.convs_layers = nn.ModuleList()
        # # layer1
        # self.convs_layers.append(GCNConv(channel, self.hidden_channel))
        # for _ in range(self.convs-2):
        #     self.convs_layers.append(GCNConv(self.hidden_channel, self.hidden_channel))
        # # last layer
        # self.convs_layers.append(GCNConv(self.hidden_channel, channel))
        # self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, user_emb, item_emb):  # entity kg embedding from TransE
        # concat user emb and updated item emb

        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]

        for i in range(self.convs):
            all_emb = self.convs_layers[i](all_emb, self.adj_mat)
            # all_emb = self.conv(all_emb)
            # all_emb = torch.matmul(all_emb, self.W)
            # norm_emb = torch.sparse.mm(self.adj_mat, all_emb)
            # all_emb = self.leaky_relu(all_emb)
            norm_emb = F.normalize(all_emb, p=2, dim=1)
            embs.append(norm_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out[:self.n_users], light_out[self.n_users:]


class Disentangle(nn.Module):
    def __init__(self, adj_mat, ui_mat, iu_mat, channel, n_users, n_items, n_intent, n_relation, layer, ind, tmp=0.2,
                 k=4, save_path=None):
        super(Disentangle, self).__init__()
        self.adj = adj_mat  # 没有归一化的. coo_tensor
        self.ui_mat = ui_mat  # coo_tensor. [n_user, n_item]
        self.iu_mat = iu_mat  # coo_tensor. [n_item, n_user]
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = self.n_items + self.n_users
        self.n_intent = n_intent
        self.n_relation = n_relation
        self.emb_size = channel
        self.convs = layer
        self.ind = ind  # cosine
        self.temperature = tmp
        # weight = init(torch.empty(self.n_intent, self.n_relation))
        Q = init(torch.empty(self.n_intent, channel))
        self.Q = nn.Parameter(Q)
        # concat intent emb, mlp

        # EGLN
        self.topk = k
        self.W1 = nn.Parameter(init(torch.empty(channel, 16)))
        self.W2 = nn.Parameter(init(torch.empty(channel, 16)))
        self.save_path = save_path

    def cal_edge(self, tensor):  # from DISENE
        S = torch.mm(tensor, tensor.T)
        S = S / torch.norm(S, dim=1, keepdim=True)
        upper_triangular_indices = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)
        upper_triangular_values = S[upper_triangular_indices[0], upper_triangular_indices[1]]
        average_inner_product = torch.mean(upper_triangular_values)
        # equation (1) in DISENE
        # mask = (S - average_inner_product > 0).float()
        logits = torch.stack([- (S - average_inner_product), S - average_inner_product], dim=-1)

        def gumbel_softmax(logits, temperature=1.0):
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))  # 生成 Gumbel 噪声
            y = logits + gumbel_noise
            return torch.softmax(y / temperature, dim=-1)

        prob_distribution = gumbel_softmax(logits, temperature=0.01)[:, 0]
        mask = prob_distribution[:, :, 1]
        return mask * self.adj
        # equation (2)
        # masked_adj = mask * self.cf_adj
        # return masked_adj  # shape:[n_user+n_item, n_user+n_item]

    def subgraph(self, all_emb):  # 合起来乘
        sim_matrix = torch.mm(all_emb, all_emb.T)  # [n_node, n_node]
        # select topk for subgraph construction
        topk_value, topk_idx = torch.topk(sim_matrix, self.topk)  # [n_node, topk]
        topk_value = topk_value.view(-1)
        row = topk_idx.view(-1, 1).to(torch.int64)
        col = torch.arange(self.n_nodes).reshape(-1, 1).repeat(1, self.topk)
        indice1 = torch.stack([row.squeeze(), col.squeeze()], dim=0)
        sparse_simi = torch.sparse_coo_tensor(indice1, topk_value, torch.Size([self.n_nodes, self.n_nodes]))
        # A(E) = A + A(R). to dense add:
        adj = self.adj.to_dense() + sparse_simi.to_dense()
        row_sum = adj.sum(dim=1, keepdim=True)
        # sparse add:
        # indice2 = self.adj.indices()
        # value2 = self.adj.values()
        # indices = torch.cat([indice1, indice2], dim=1)
        # value = torch.cat([topk_value, value2])
        # unique_indice = torch.unique(indices, dim=1)
        # summed_value = torch.zeros_like()
        # summed_adj = torch.sparse_coo_tensor()
        # 归一化
        norm_adj = adj / row_sum
        loss_simi_adj = 0.1 * torch.mean((sim_matrix - self.adj) ** 2)  # eq11
        # 子图学习
        user_emb, item_emb = self.GNN(all_emb, norm_adj)
        return user_emb, item_emb, loss_simi_adj

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
        filtered_combined = sorted_combined[:1000]  # TODO: change
        filtered_values = torch.tensor([x[0] for x in filtered_combined])  # 相似度值
        filtered_indices = torch.tensor([[x[1], x[2]] for x in filtered_combined])  # 用户-物品对索引
        item_filtered_sparse_simi = torch.sparse_coo_tensor(filtered_indices.t(), filtered_values,
                                                            (self.n_users, self.n_items))
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
        user_item_final_matrix = self.nor_sparse_matrix(add_sparse_user_matrix)
        item_user_final_matrix = self.nor_sparse_matrix(add_sparse_item_matrix)

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
            # TODO:如何将adj设置为根据loss更新的矩阵？
            temp_emb = torch.sparse.mm(adj, temp_emb)
            embs.append(temp_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        user_emb, item_emb = torch.split(light_out, [self.n_users, self.n_items])
        return user_emb, item_emb

    """relation emb解耦"""
    def forward(self, user_emb, item_emb, r_emb):
        """dimension: [16,64] -> [64]"""
        # K = torch.sum(r_emb, dim=0)     # [dim]
        # QK = self.Q * K                 # [intent, dim]
        # col_sum = torch.sum(QK, dim=0, keepdim=True)
        # QK = QK / col_sum
        # intent_emb = QK * K     # [intent, dim]
        # relation-intent attention
        attention_scores = torch.matmul(self.Q, r_emb.T)  # [intent, relation]
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


        """按位乘"""
        disen_weight = intent_emb.unsqueeze(0).expand(self.n_users, -1, -1)
        disen_weight1 = intent_emb.unsqueeze(0).expand(self.n_items, -1, -1)
        user_emb1 = user_emb.unsqueeze(1).expand(-1, self.n_intent, -1)
        item_emb1 = item_emb.unsqueeze(1).expand(-1, self.n_intent, -1)
        # user_int_emb = (user_emb1 * disen_weight).reshape(self.n_users, self.emb_size * self.n_intent)
        user_int_emb = user_emb1 * disen_weight
        # item_int_emb = (item_emb1 * disen_weight1).reshape(self.n_items, self.emb_size * self.n_intent)
        item_int_emb = item_emb1 * disen_weight1
        user_int_list, item_int_list = [], []
        for i in range(self.n_intent):
            # int_emb = all_emb[:, i, :]
            user_int = user_int_emb[:, i, :]
            item_int = item_int_emb[:, i, :]
            user, item = self.EGLN(user_int, item_int)
            # user_int, item_int, sim_loss = self.subgraph(int_emb)
            user_int_list.append(user)
            item_int_list.append(item)

        user_int_emb = torch.cat(user_int_list, dim=1)
        item_int_emb = torch.cat(item_int_list, dim=1)
        assert user_int_emb.shape == (self.n_users, self.emb_size * self.n_intent)
        assert item_int_emb.shape == (self.n_items, self.emb_size * self.n_intent)
        # return user_int_emb, item_int_emb, self.calculate_cor_loss(intent_emb)
        return user_int_emb, item_int_emb, self.calculate_cor_loss(intent_emb)

    """dimension level concat intent"""

    # def forward(self, user_emb, entity_emb, r_kg_emb):  # relation embedding from transE
    #     item_emb1 = entity_emb[:self.n_items, :]
    #     # [16,64] -> [64]
    #     r_kg_emb1 = torch.sum(r_kg_emb, dim=0)
    #     # intent_emb = torch.mm(nn.Softmax(dim=-1)(self.weight), r_kg_emb)
    #     """dimension level QKV"""
    #     # [4,64] * [64]
    #     # softmax归一化
    #     att_score = nn.Softmax(dim=0)(self.weight * r_kg_emb1)  # [intent, dim]
    #     # 线性归一化
    #     # col_sum = torch.sum(att_score, dim=0, keepdim=True)
    #     # att_score = att_score / col_sum
    #     intent_emb = att_score * r_kg_emb1  # [intent, dim]
    #
    #     # intent_weight = att_score.unsqueeze(1) * r_kg_emb.unsqueeze(0)  # r_kg_emb:[relation, 1, dim]
    #     # dis_intent_weight = torch.sum(intent_weight, dim=1)     # [intent, dim]
    #
    #     # try2
    #     # intent_emb = torch.matmul(att_score, r_kg_emb.T)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         # [intent, relation]
    #     # dis_intent_weight = torch.matmul(intent_emb, r_kg_emb)
    #     """全连接"""
    #     all_emb = torch.cat([user_emb, item_emb1, intent_emb], dim=0)
    #     all_emb = self.W(all_emb)
    #     user_int_emb = all_emb[:self.n_users, :]
    #     item_int_emb = all_emb[self.n_users:self.n_users + self.n_items, :]
    #     """按位乘intent和node emb"""
    #
    #     # disen_weight = dis_intent_weight.unsqueeze(0).expand(self.n_users, -1, -1)
    #     # disen_weight1 = dis_intent_weight.unsqueeze(0).expand(self.n_items, -1, -1)
    #     # user_emb1 = user_emb.unsqueeze(1).expand(-1, self.n_intent, -1)
    #     # item_emb = item_emb1.unsqueeze(1).expand(-1, self.n_intent, -1)
    #     # # concat user, item embedding to n_intent*dim
    #     # user_int_emb = (user_emb1 * disen_weight).reshape(self.n_users, self.n_intent * self.emb_size)
    #     # item_int_emb = (item_emb * disen_weight1).reshape(self.n_items, self.n_intent * self.emb_size)
    #     """use subgraph adjacent matrix"""
    #     # all_emb = torch.cat((user_emb, item_emb1), dim=0).unsqueeze(1).expand(-1, self.n_intent,
    #     #                                                                       -1)  # shape: [n_node, n_intent, dim]
    #     # disen_weight = torch.mm(nn.Softmax(dim=-1)(self.weight), relation_emb).unsqueeze(0).expand(
    #     #         self.n_users+self.n_items, -1, -1)   # shape: [n_node, n_intent, dim]
    #     # # disentangle
    #     # all_int_emb = all_emb * disen_weight
    #     # user_int_list, item_int_list = [], []
    #     # # for each intent, build subgraph
    #     # for i in range(self.n_intent):
    #     #     all_e = all_int_emb[:, i, :].squeeze(1)  # shape:[n_node, dim]
    #     #     # calculate node similarity to decide build edge or not, create mask matrix
    #     #     int_adj = self.cal_edge(all_e)
    #     #     # subgraph GNN
    #     #     user_int, item_int = self.GNN(all_e, int_adj)
    #     #     user_int_list.append(user_int)
    #     #     item_int_list.append(item_int)
    #     # # concat intent embeddings, return final result
    #     # user_int_emb = torch.cat(user_int_list, dim=1)
    #     # item_int_emb = torch.cat(item_int_list, dim=1)
    #
    #     # user_int_emb = ui_int_emb[:self.n_users, :]
    #     # item_int_emb = ui_int_emb[self.n_users:, :]
    #     # assert user_int_emb.shape == (self.n_users, self.emb_size * self.n_intent)
    #     # assert item_int_emb.shape == (self.n_items, self.emb_size * self.n_intent)
    #     return user_int_emb, item_int_emb, self.calculate_cor_loss(intent_emb)
    #     # return user_int_emb, item_int_emb

    def calculate_cor_loss(self, tensors):
        def orthogonal_loss(intent_emb):
            intent_emb_norm = intent_emb / intent_emb.norm(dim=1, keepdim=True)
            identity = torch.eye(self.n_intent, device=intent_emb.device)
            return torch.norm(torch.matmul(intent_emb_norm, intent_emb_norm.T) - identity)

        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = torch.nn.functional.normalize(tensor_1, dim=0)
            normalized_tensor_2 = torch.nn.functional.normalize(tensor_2, dim=0)
            return (normalized_tensor_1 * normalized_tensor_2).sum(
                dim=0
            ) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = (
                torch.matmul(tensor_1, tensor_1.t()) * 2,
                torch.matmul(tensor_2, tensor_2.t()) * 2,
            )  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(
                torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8
            ), torch.sqrt(
                torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8
            )  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation(tensors):
            # tensors: [n_factors, dimension]
            # normalized_tensors: [n_factors, dimension]
            normalized_tensors = torch.nn.functional.normalize(tensors, dim=1)
            scores = torch.mm(normalized_tensors, normalized_tensors.t())
            scores = torch.exp(scores / self.temperature)
            cor_loss = -torch.sum(torch.log(scores.diag() / scores.sum(1)))
            return cor_loss

        """cul similarity for each latent factor weight pairs"""
        if self.ind == "mi":
            return MutualInformation(tensors)
        elif self.ind == "distance":
            cor_loss = 0.0
            for i in range(self.n_intent):
                for j in range(i + 1, self.n_intent):
                    cor_loss += DistanceCorrelation(tensors[i], tensors[j])
        elif self.ind == "cosine":
            cor_loss = 0.0
            for i in range(self.n_intent):
                for j in range(i + 1, self.n_intent):
                    cor_loss += CosineSimilarity(tensors[i], tensors[j])
            # cor_loss += orthogonal_loss(tensors)
        else:
            raise NotImplementedError(
                f"The independence loss type [{self.ind}] has not been supported."
            )
        return cor_loss


class MRAM(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat, adj_mean_mat, pretrain_emb=None):
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
        self.graph = graph  # CKG
        self.ckg_edge_index, self.ckg_edge_type = self._get_edges(graph)

        self.all_embed = torch.nn.Embedding(self.n_nodes, self.emb_size)
        # self.relation_emb = torch.nn.Embedding(self.n_relations, self.emb_size)
        self.user_embed = torch.nn.Embedding(self.n_users, self.emb_size)
        self.item_embed = torch.nn.Embedding(self.n_items, self.emb_size)
        self.intent_embed = torch.nn.Embedding(self.n_intent, self.emb_size)
        if pretrain_emb is not None:
            self.user_embed.weight = nn.Parameter(pretrain_emb['embedding_user.weight'])
            self.item_embed.weight = nn.Parameter(pretrain_emb['embedding_item.weight'])
            self.intent_embed.weight = nn.Parameter(pretrain_emb['embedding_intent.weight'])

        # self.trans_w = torch.nn.Embedding(self.n_relations, self.emb_size * self.kg_emb_size)
        # 用ui index
        edge_index, _ = from_scipy_sparse_matrix(self.adj_mat)
        self.edge_index = edge_index.to(self.device)
        self.adj_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.ui_mat = self._convert_sp_mat_to_sp_tensor(self.ui_mat).to(self.device)
        self.iu_mat = self._convert_sp_mat_to_sp_tensor(self.iu_mat).to(self.device)
        self.kg_encoder = R_GraphConv(self.emb_size, self.ckg_edge_index, self.ckg_edge_type, self.encode_layer,
                                      self.n_users,
                                      self.n_items, self.n_relations)
        # self.encoder = LightGraphConv(self.adj_mat, self.encode_layer, self.n_users, self.n_items)
        self.encoder = GraphConv(self.edge_index, self.encode_layer, self.n_users,
                                 self.n_items, self.n_relations, self.emb_size)
        # self.encoder = R_GraphConv(self.emb_size, self.ckg_edge_index, self.ckg_edge_type, self.encode_layer, self.n_users,
        #                                                   self.n_items, self.n_relations)
        self.decoder = Disentangle(self.adj_mat, self.ui_mat, self.iu_mat, self.emb_size, self.n_users, self.n_items,
                                   self.n_intent, self.n_relations,
                                   self.decode_layer, self.ind, k=self.k, save_path=args_config.data_path + args_config.dataset)
        # self.decoder = Disentangle(self.cf_mat, self.emb_size, self.decode_layer, self.n_users, self.n_items,
        #                            self.n_intent, self.n_relations)

        # self.reg_loss = EmbLoss()

    def _get_edges(self, graph):  # graph:[num_nodes, [h, t, r_id]]
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]. [h, t]
        type = graph_tensor[:, -1]  # [-1, 1]. r_id
        return index.t().long().to(self.device), type.long().to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        # return torch.sparse.FloatTensor(i, v, coo.shape)
        return torch.sparse_coo_tensor(i, v, coo.shape)

    def _convert_sp_mat_to_edge_index(self, X):
        coo = X.tocoo()
        row = torch.from_numpy(coo.row).long()  # 行索引
        col = torch.from_numpy(coo.col).long()  # 列索引
        edge_index = torch.stack([row, col], dim=0)

        return edge_index

    def _calculate_embedding(self):
        # RGCN
        item_emb, r_emb = self.kg_encoder(self.all_embed.weight)
        user_emb, item_emb = self.encoder(self.user_embed.weight, item_emb)
        user_int_emb, item_int_emb, cor_loss = self.decoder(user_emb, item_emb, r_emb)
        return user_int_emb, item_int_emb, cor_loss
        #
        # transR: relation embedding. RGCN: item embedding
        # item_emb = self.kg_encoder(self.all_embed.weight)
        # user_emb, item_emb = self.encoder(self.user_embed.weight, item_emb)
        # user_int_emb, item_int_emb, cor_loss = self.decoder(user_emb, item_emb, self.relation_emb.weight)

        """pretrain from KGIN"""
        # user_int_emb, item_int_emb, cor_loss = self.decoder(self.user_embed.weight, self.item_embed.weight, self.intent_embed.weight)
        # return self.user_embed.weight, self.item_embed.weight

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
        # 更新all_emb
        # user_int_emb, item_int_emb = self._calculate_embedding()
        user_int_emb, item_int_emb, cor_loss = self._calculate_embedding()
        u_e = user_int_emb[user]
        pos_e, neg_e = item_int_emb[pos_item], item_int_emb[neg_item]
        # ssm_loss = self.ssm_loss(u_e, pos_e)
        # mf_loss = self.create_bpr_loss(u_e, pos_e, neg_e, cor_loss)
        mf_loss = self.create_bpr_loss_wo_cor(u_e, pos_e, neg_e)

        return mf_loss, cor_loss
        # return ssm_loss + mf_loss

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

    def generate(self):
        return self._calculate_embedding()

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())
