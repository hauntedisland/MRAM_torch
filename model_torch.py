import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


# LightGCN: 更新user, item embedding→self.all_emb
class GraphConv(nn.Module):
    def __init__(self, all_emb, relation_emb, adj_mat, conv_layers, n_users, n_items, n_relations):
        super(GraphConv, self).__init__()
        self.ckg_emb = all_emb  # [entity, channel]
        self.relation_emb = relation_emb
        self.adj_mat = adj_mat  # [entity+n_relation-1, entity+n_relation-1]
        self.convs = conv_layers
        self.n_users = n_users
        self.n_items = n_items
        self.n_relation = n_relations - 1

    def forward(self):
        # concat node emb and relation emb
        all_emb = torch.cat((self.ckg_emb, self.relation_emb), dim=0)
        embs = [all_emb]
        temp_emb = all_emb
        for i in range(self.convs):
            temp_emb = torch.sparse.mm(self.adj_mat, temp_emb)
            embs.append(temp_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        user_emb = light_out[:self.n_users, :]
        item_emb = light_out[self.n_users:self.n_users + self.n_items, :]
        relation_emb = light_out[self.n_users + self.n_items:self.n_users + self.n_items + self.n_relation, :]
        return user_emb, item_emb, relation_emb


class Disentangle(nn.Module):
    def __init__(self, channel, n_users, n_items, n_intent, n_relation):
        super(Disentangle, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_intent = n_intent
        self.n_relation = n_relation - 1
        self.emb_size = channel
        # 将relation解耦和intent解耦矩阵设置为可学习的
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(self.n_intent, self.n_relation))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

    def forward(self, user_emb, item_emb, relation_emb):
        # [n_intent, n_relation] * [n_relation, dim] = [n_intent, dim]
        # TODO: 目前给所有user的weight都是一样的，没有personalized
        disen_weight = torch.mm(nn.Softmax(dim=-1)(self.weight), relation_emb).unsqueeze(0).expand(
            self.n_users, -1, -1)
        user_emb1 = user_emb.unsqueeze(1).expand(-1, self.n_intent, -1)

        # concat user, item embedding to n_intent*dim
        user_int_emb = user_emb1 * disen_weight
        user_int_emb = user_int_emb.reshape(self.n_users, self.n_intent*self.emb_size)
        item_int_emb = torch.cat([item_emb for _ in range(self.n_intent)], dim=1)
        # mean:
        # user_int_emb = torch.mean(user_int_emb, dim=1)
        # item_int_emb = item_emb
        assert user_int_emb.shape == (self.n_users, self.emb_size*self.n_intent)
        assert item_int_emb.shape == (self.n_items, self.emb_size*self.n_intent)
        return user_int_emb, item_int_emb


class MRAM(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(MRAM, self).__init__()
        self.decay = args_config.l2
        self.ssm = args_config.ssm

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
        self.graph = graph

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.relation_emb = nn.Parameter(self.relation_emb)
        # self.intent_emb = nn.Parameter(self.intent_emb)
        # CKG encoder
        self.encoder = GraphConv(self.all_embed, self.relation_emb, self.ckg_mat, self.encode_layer, self.n_users,
                                 self.n_items, self.n_relations)
        self.decoder = Disentangle(self.emb_size, self.n_users, self.n_items, self.n_intent, self.n_relations)
        # self.decoder = Disentangle(self.cf_mat, self.emb_size, self.decode_layer, self.n_users, self.n_items,
        #                            self.n_intent, self.n_relations)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        # self.all_embed = initializer(torch.empty(self.n_users + self.n_items, self.emb_size))
        self.relation_emb = initializer(torch.empty(self.n_relations - 1, self.emb_size))
        # self.intent_emb = initializer(torch.empty(self.n_intent, self.emb_size))  # intent embedding

        # [n_users+n_entities, n_users+n_entities]
        self.ckg_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        # self.cf_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat_cf).to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _calculate_embedding(self):
        user_emb, item_emb, relation_emb = self.encoder()
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
