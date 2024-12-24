import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
#     # TODO: 实现更新relation emb
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


class GraphConv(nn.Module):
    def __init__(self, user_emb, adj_mat, conv_layers, n_users, n_items, n_relations):
        super(GraphConv, self).__init__()
        self.user_emb = user_emb  # [entity, channel]
        self.adj_mat = adj_mat  # [entity+n_relation-1, entity+n_relation-1]
        self.convs = conv_layers  # encode layer
        self.n_users = n_users
        self.n_items = n_items
        self.n_relation = n_relations - 1

    def forward(self, entity_kg_emb):  # entity kg embedding from TransE
        user_emb = self.user_emb
        item_emb = entity_kg_emb[:self.n_items, :]
        # concat user emb and updated item emb
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        temp_emb = all_emb
        # GCN
        for i in range(self.convs):
            temp_emb = torch.sparse.mm(self.adj_mat, temp_emb)
            embs.append(temp_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        return light_out[:self.n_users], light_out[self.n_users:]
        # return light_out[:self.n_users]


class Disentangle(nn.Module):
    def __init__(self, channel, n_users, n_items, n_intent, n_relation, layer):
        super(Disentangle, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = self.n_items + self.n_users
        self.n_intent = n_intent
        self.n_relation = n_relation
        self.emb_size = channel
        self.convs = layer
        self.net = nn.Sequential(nn.Linear())
        # 将relation解耦和intent解耦矩阵设置为可学习的
        weight = init(torch.empty(self.n_intent, self.n_relation))      # xavier initialization
        self.weight = nn.Parameter(weight)

    def cal_edge(self, tensor):
        S = torch.mm(tensor, tensor.T)
        upper_triangular_indices = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)
        upper_triangular_values = S[upper_triangular_indices[0], upper_triangular_indices[1]]
        average_inner_product = torch.mean(upper_triangular_values)
        # equation (1) in DISENE
        mask = (S - average_inner_product > 0).float()
        return mask
        # equation (2)
        # masked_adj = mask * self.cf_adj
        # return masked_adj  # shape:[n_user+n_item, n_user+n_item]

    def GNN(self, all_emb, adj):
        embs = [all_emb]
        temp_emb = all_emb
        for i in range(self.convs):
            # TODO:如何将adj设置为根据loss更新的矩阵？
            temp_emb = torch.sparse.mm(adj, temp_emb)
            embs.append(temp_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        user_emb, item_emb = torch.split(light_out, [self.n_users, self.n_items])
        return user_emb, item_emb

    # def forward(self, user_emb, item_emb):
    #     all_emb = torch.cat((user_emb, item_emb), dim=0).unsqueeze(1).expand(-1, self.n_intent,
    #                                                                          -1)  # shape: [n_node, n_intent, dim]
    #     disen_weight = self.disen_weight.unsqueeze(0).expand(self.n_users + self.n_items, -1,
    #                                                          -1)  # shape: [n_node, n_intent, dim]
    #     # disentangle
    #     all_int_emb = all_emb * disen_weight
    #     user_int_list, item_int_list = [], []
    #     # for each intent, build subgraph
    #     for int in range(self.n_intent):
    #         all_e = all_int_emb[:, int, :].squeeze(1)  # shape:[n_node, dim]
    #         # calculate node similarity to decide build edge or not, create mask matrix
    #         int_adj = self.cal_edge(all_e)
    #         # subgraph GNN
    #         user_int, item_int = self.GNN(all_e, int_adj)
    #         user_int_list.append(user_int)
    #         item_int_list.append(item_int)
    #     # concat intent embeddings, return final result
    #     user_int_emb = torch.cat(user_int_list, dim=1)
    #     item_int_emb = torch.cat(item_int_list, dim=1)
    #     # check
    #     assert user_int_emb.shape == (self.n_users, self.emb_size * self.n_intent)
    #     assert item_int_emb.shape == (self.n_items, self.emb_size * self.n_intent)
    #     return user_int_emb, item_int_emb

    def forward(self, user_emb, item_emb, r_kg_emb):  # relation embedding from transE
        # ui_emb = all_emb[:self.n_users + self.n_items, :]
        # disen_weight = torch.mm(nn.Softmax(dim=-1)(self.weight), relation_emb).unsqueeze(0).expand(
        #     self.n_users+self.n_items, -1, -1)
        # ui_emb = ui_emb.unsqueeze(1).expand(-1, self.n_intent, -1)
        # ui_int_emb = (ui_emb * disen_weight).reshape(-1, self.n_intent * self.emb_size)

        disen_weight = torch.mm(nn.Softmax(dim=-1)(self.weight), r_kg_emb).unsqueeze(0).expand(
            self.n_users, -1, -1)
        disen_weight1 = torch.mm(nn.Softmax(dim=-1)(self.weight), r_kg_emb).unsqueeze(0).expand(
            self.n_items, -1, -1)
        user_emb1 = user_emb.unsqueeze(1).expand(-1, self.n_intent, -1)
        item_emb = item_emb.unsqueeze(1).expand(-1, self.n_intent, -1)
        # concat user, item embedding to n_intent*dim
        user_int_emb = (user_emb1 * disen_weight).reshape(self.n_users, self.n_intent * self.emb_size)
        item_int_emb = (item_emb * disen_weight1).reshape(self.n_items, self.n_intent * self.emb_size)

        # method2: use subgraph adjacent matrix
        # all_emb = torch.cat((user_emb, item_emb1), dim=0).unsqueeze(1).expand(-1, self.n_intent,
        #                                                                       -1)  # shape: [n_node, n_intent, dim]
        # disen_weight = torch.mm(nn.Softmax(dim=-1)(self.weight), relation_emb).unsqueeze(0).expand(
        #         self.n_users+self.n_items, -1, -1)   # shape: [n_node, n_intent, dim]
        # # disentangle
        # all_int_emb = all_emb * disen_weight
        # user_int_list, item_int_list = [], []
        # # for each intent, build subgraph
        # for i in range(self.n_intent):
        #     all_e = all_int_emb[:, i, :].squeeze(1)  # shape:[n_node, dim]
        #     # calculate node similarity to decide build edge or not, create mask matrix
        #     int_adj = self.cal_edge(all_e)
        #     # subgraph GNN
        #     user_int, item_int = self.GNN(all_e, int_adj)
        #     user_int_list.append(user_int)
        #     item_int_list.append(item_int)
        # # concat intent embeddings, return final result
        # user_int_emb = torch.cat(user_int_list, dim=1)
        # item_int_emb = torch.cat(item_int_list, dim=1)

        # user_int_emb = ui_int_emb[:self.n_users, :]
        # item_int_emb = ui_int_emb[self.n_users:, :]
        assert user_int_emb.shape == (self.n_users, self.emb_size * self.n_intent)
        assert item_int_emb.shape == (self.n_items, self.emb_size * self.n_intent)
        return user_int_emb, item_int_emb


class MRAM(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(MRAM, self).__init__()
        self.decay = args_config.l2
        # self.ssm = args_config.ssm

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

        self.adj_mat = adj_mat
        self.graph = graph  # KG
        self.edge_index, self.edge_type = self._get_edges(graph)
        # self.kg_hop = args_config.layer_num_kg
        # self.all_embed = torch.nn.Embedding(self.n_nodes, self.emb_size)
        self.user_embed = torch.nn.Embedding(self.n_users, self.emb_size)
        self.entity_embed = torch.nn.Embedding(self.n_entities, self.emb_size)
        self.relation_emb = torch.nn.Embedding(self.n_relations, self.emb_size)
        self.trans_w = torch.nn.Embedding(self.n_relations, self.emb_size * self.kg_emb_size)

        self.ckg_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

        self.encoder = GraphConv(self.user_embed.weight, self.ckg_mat, self.encode_layer, self.n_users,
                                 self.n_items, self.n_relations)
        self.decoder = Disentangle(self.emb_size, self.n_users, self.n_items, self.n_intent, self.n_relations, self.decode_layer)
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
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _calculate_embedding(self):
        user_emb, item_emb = self.encoder(self.entity_embed.weight)  # 传入训练来的trans embedding
        user_int_emb, item_int_emb = self.decoder(user_emb, item_emb, self.relation_emb.weight)
        # user_int_emb, item_int_emb = self.decoder(self.all_embed.weight, self.relation_emb.weight)
        return user_int_emb, item_int_emb

    def _get_kg_embedding(self, h, r, pos_t, neg_t):  # rectorch
        h_e = self.entity_embed(h).unsqueeze(1)  # (kg_batch_size, 1, relation_dim)
        pos_t_e = self.entity_embed(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embed(neg_t).unsqueeze(1)
        # h_e = self.all_embed(h).unsqueeze(1) # (kg_batch_size, 1, relation_dim)
        # pos_t_e = self.all_embed(pos_t).unsqueeze(1)
        # neg_t_e = self.all_embed(neg_t).unsqueeze(1)
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
        user_int_emb, item_int_emb = self._calculate_embedding()
        u_e = user_int_emb[user]
        pos_e, neg_e = item_int_emb[pos_item], item_int_emb[neg_item]
        # ssm_loss = self.ssm_loss(u_e, pos_e)
        mf_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        return mf_loss
        # return ssm_loss + mf_loss

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
        # return mf_loss
        return mf_loss + emb_loss

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
