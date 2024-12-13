import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax

init = nn.init.xavier_uniform_


class RGAT(nn.Module):
    def __init__(self, latdim, n_hops, mess_dropout_rate, n_item, n_relation):
        super(RGAT, self).__init__()
        self.mess_dropout_rate = mess_dropout_rate
        self.W = nn.Parameter(init(torch.empty(size=(2 * latdim, latdim)), gain=nn.init.calculate_gain('relu')))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.n_hops = n_hops
        self.dropout = nn.Dropout(p=mess_dropout_rate)
        self.n_item = n_item

        # projection from item space to relation space(?)
        self.W2 = nn.Parameter(init(torch.empty(size=(64, 64)), gain=nn.init.calculate_gain('relu')))

    def agg(self, entity_emb, relation_emb, kg):
        edge_index, edge_type = kg
        head, tail = edge_index
        a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
        # 将entity通过W矩阵映射到relation emb上. [head+tail, d] * [relation, d]
        e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type - 1]).sum(
            -1)
        e = self.leakyrelu(e_input)
        # 将head位置的向量(item)都转换为概率分布
        e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
        # 按概率聚合连接的entity，更新item. e本身就是一种概率分数，使用view(-1,1)
        agg_emb = entity_emb[tail] * e.view(-1, 1)
        # 将与每个head索引相关的所有tail索引对应的加权实体嵌入向量求和。
        agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
        # 1.计算relation -> item attention (学习一个映射矩阵)
        # transformed_agg = torch.mm(agg_emb[:self.n_item, :], self.W2)   # 线性变换
        # ir_att = F.softmax(torch.matmul(transformed_agg, relation_emb.t()), dim=1)     # attention on relation_dim. [item, relation]
        # 偷KGIN
        score = F.softmax(torch.mm(relation_emb, agg_emb.t()), dim=1)   # (relation, item)
        r_emb = torch.matmul(score, agg_emb)
        # 2.用注意力更新relation
        # r_emb = torch.matmul(ir_att, relation_emb)  # shape error.ir_att shape, how to multiply?
        # agg_emb = agg_emb + entity_emb
        return agg_emb, r_emb

    # TODO: 实现更新relation emb
    def forward(self, entity_emb, relation_emb, kg, res_lambda, mess_dropout=True):
        entity_res_emb = entity_emb
        for _ in range(self.n_hops):
            entity_emb, r_emb = self.agg(entity_emb, relation_emb, kg)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            entity_res_emb = res_lambda * entity_res_emb + entity_emb
        return entity_res_emb, r_emb


# RGAT更新item embedding, relation embedding
# 将user, item embedding concat, 使用LightGCN更新
# 返回：user, item, relation embedding
class GraphConv(nn.Module):
    def __init__(self, all_emb, relation_emb, adj_mat, conv_layers, n_users, n_items, n_relations, kg_hop, dim,
                 mess_dropout_rate):
        super(GraphConv, self).__init__()
        self.ckg_emb = all_emb  # [entity, channel]
        self.relation_emb = relation_emb
        self.adj_mat = adj_mat  # [entity+n_relation-1, entity+n_relation-1]
        self.convs = conv_layers  # encode layer
        self.n_users = n_users
        self.n_items = n_items
        self.n_relation = n_relations - 1
        self.mess_dropout_rate = mess_dropout_rate
        # self.rgat = RGAT(dim, kg_hop, self.mess_dropout_rate, self.n_items, self.n_relation)
        self.trans =
    def forward(self, kg, res, mess_dropout=True):
        user_emb = self.ckg_emb[:self.n_users, :]
        entity_emb = self.ckg_emb[self.n_users:, :]
        # change to Trans
        entity_kg_emb, r_emb = self.rgat.forward(entity_emb, self.relation_emb, kg, res, mess_dropout)
        # concat user emb and updated item emb
        all_emb = torch.cat([user_emb, entity_kg_emb[:self.n_items, :]], dim=0)
        embs = [all_emb]
        temp_emb = all_emb
        # GCN
        for i in range(self.convs):
            temp_emb = torch.sparse.mm(self.adj_mat, temp_emb)
            embs.append(temp_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        # user_emb = light_out[:self.n_users, :]
        # item_emb = light_out[self.n_users:self.n_users + self.n_items, :]
        #
        # relation_emb = light_out[self.n_users + self.n_items:self.n_users + self.n_items + self.n_relation, :]
        return light_out[:self.n_users], light_out[self.n_users:], r_emb


class Disentangle(nn.Module):
    def __init__(self, channel, n_users, n_items, n_intent, n_relation):
        super(Disentangle, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_intent = n_intent
        self.n_relation = n_relation - 1
        self.emb_size = channel
        # 将relation解耦和intent解耦矩阵设置为可学习的
        weight = init(torch.empty(self.n_intent, self.n_relation))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

    def forward(self, user_emb, item_emb, relation_emb):
        # [n_intent, n_relation] * [n_relation, dim] = [n_intent, dim]
        # TODO: 目前给所有user的weight都是一样的，没有personalized
        disen_weight = torch.mm(nn.Softmax(dim=-1)(self.weight), relation_emb).unsqueeze(0).expand(
            self.n_users, -1, -1)
        disen_weight1 = torch.mm(nn.Softmax(dim=-1)(self.weight), relation_emb).unsqueeze(0).expand(
            self.n_items, -1, -1)
        user_emb1 = user_emb.unsqueeze(1).expand(-1, self.n_intent, -1)
        item_emb1 = item_emb.unsqueeze(1).expand(-1, self.n_intent, -1)
        # concat user, item embedding to n_intent*dim
        user_int_emb = (user_emb1 * disen_weight).reshape(self.n_users, self.n_intent * self.emb_size)
        # item_int_emb = torch.cat([item_emb for _ in range(self.n_intent)], dim=1)   # TODO
        item_int_emb = (item_emb1 * disen_weight1).reshape(self.n_items, self.n_intent * self.emb_size)
        # mean:
        # user_int_emb = torch.mean(user_int_emb, dim=1)
        # item_int_emb = item_emb
        assert user_int_emb.shape == (self.n_users, self.emb_size * self.n_intent)
        assert item_int_emb.shape == (self.n_items, self.emb_size * self.n_intent)
        return user_int_emb, item_int_emb


class MRAM(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(MRAM, self).__init__()
        self.decay = args_config.l2
        # self.ssm = args_config.ssm

        self.mess_drop_rate = args_config.mess_dropout_rate
        self.res = args_config.res_lambda

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items!
        self.n_nodes = data_config['n_nodes']  # entity + user

        self.n_intent = args_config.n_intent
        self.emb_size = args_config.dim
        self.encode_layer = args_config.encode_layer  # encoder layer
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = adj_mat
        self.graph = graph  # KG
        self.edge_index, self.edge_type = self._get_edges(graph)
        self.kg_hop = args_config.layer_num_kg

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.relation_emb = nn.Parameter(self.relation_emb)
        # self.intent_emb = nn.Parameter(self.intent_emb)
        # CKG encoder
        self.encoder = GraphConv(self.all_embed, self.relation_emb, self.ckg_mat, self.encode_layer, self.n_users,
                                 self.n_items, self.n_relations, self.kg_hop, self.emb_size, self.mess_drop_rate)
        self.decoder = Disentangle(self.emb_size, self.n_users, self.n_items, self.n_intent, self.n_relations)
        # self.decoder = Disentangle(self.cf_mat, self.emb_size, self.decode_layer, self.n_users, self.n_items,
        #                            self.n_intent, self.n_relations)

    def _init_weight(self):
        self.all_embed = init(torch.empty(self.n_nodes, self.emb_size))
        # self.all_embed = initializer(torch.empty(self.n_users + self.n_items, self.emb_size))
        self.relation_emb = init(torch.empty(self.n_relations - 1, self.emb_size))
        # self.intent_emb = initializer(torch.empty(self.n_intent, self.emb_size))  # intent embedding

        # [n_users+n_entities, n_users+n_entities]
        self.ckg_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        # self.cf_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat_cf).to(self.device)

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
        user_emb, item_emb, relation_emb = self.encoder([self.edge_index, self.edge_type], self.res)
        user_int_emb, item_int_emb = self.decoder(user_emb, item_emb, relation_emb)
        return user_int_emb, item_int_emb

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
        # regularizer = (torch.norm(users) ** 2
        #                + torch.norm(pos_items) ** 2
        #                + torch.norm(neg_items) ** 2) / 2
        # emb_loss = self.decay * regularizer / batch_size
        return mf_loss
        # return mf_loss + emb_loss

    # TODO: 改成分母是全局的LOSS
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

    def generate(self):
        return self._calculate_embedding()

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())
