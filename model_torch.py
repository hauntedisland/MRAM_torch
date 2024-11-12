import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean



# LightGCN: 更新user, item embedding→self.all_emb
class GraphConv(nn.Module):
    def __init__(self, all_emb, adj_mat, conv_layers, n_users, n_items):
        super(GraphConv, self).__init__()
        self.all_emb = all_emb
        self.adj_mat = adj_mat
        self.convs = conv_layers
        self.n_users = n_users
        self.n_items = n_items

    def forward(self):
        embs = [self.all_emb]
        temp_emb = self.all_emb
        for i in range(self.convs):
            # 检查形状？
            temp_emb = torch.sparse.mm(self.adj_mat, temp_emb)
            embs.append(temp_emb)
        print("finish encoding...")
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        user_emb, item_emb = torch.split(light_out, [self.n_users, self.n_items])
        return user_emb, item_emb


class Disentangle(nn.Module):
    def __init__(self, channel, n_users, n_intent, n_relation):
        super(Disentangle, self).__init__()
        self.n_users = n_users
        self.n_intent = n_intent
        self.n_relation = n_relation
        self.emb_size = channel
        # print(self.emb_size,self.n_users,self.n_intent,self.n_relation)
    """latent_emb: relation emb
        weight: relation weight
    """

    def forward(self, user_emb, disen_weight_att, relation_emb):
        # [n_intent, n_relation] * [n_relation, dim] = [n_intent, dim]
        # 扩展到[n_user, n_intent, dim]. editable: 给所有user的weight都是一样的
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att), relation_emb).unsqueeze(0).expand(self.n_users, -1, -1)
        user_emb1 = user_emb.unsqueeze(1).expand(-1, self.n_intent, -1)
        # user_emb1 = user_emb.unsqueeze(2)
        # user_int = torch.matmul(user_emb1, disen_weight.transpose(1, 2))
        user_int = user_emb1 * disen_weight
        # 对relation嵌入也做映射
        # [relation, n_intent, dim]
        relation_emb1 = relation_emb.unsqueeze(1).expand(-1, self.n_intent, -1)
        r_int_emb = torch.matmul(relation_emb1, disen_weight)
        # return user_int, r_int_emb
        return user_int

class MRAM(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(MRAM, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']

        self.n_intent = args_config.n_intent
        self.emb_size = args_config.embed_size
        self.n_layer = args_config.n_layer  # encoder layer
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = adj_mat
        self.graph = graph

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.intent_emb = nn.Parameter(self.intent_emb)

        self.encoder = GraphConv(self.all_embed, self.adj_mat, self.n_layer, self.n_users, self.n_items)
        self.decoder = Disentangle(self.emb_size, self.n_users, self.n_intent, self.n_relations)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.intent_emb = initializer(torch.empty(self.n_intent, self.emb_size))  # intent embedding

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb, item_emb = self.encoder()

        # disentangle的矩阵
        user_int_emb, r_int_emb = self.decoder(user_emb, self.intent_emb)
        # u_e = user_int_emb[user]
        # pos_e, neg_e = item_emb[pos_item], item_emb[neg_item]
        losses = []
        # 对每个intent维度的u_e分别计算loss
        for idx in range(self.n_intent):
            u_e = user_int_emb[:, idx, :][user]
            pos_e = item_emb[pos_item]
            neg_e = item_emb[neg_item]
            losses.append(self.create_bpr_loss(u_e, pos_e, neg_e))
        # 重构loss写在哪儿
        return sum(losses) / len(losses)

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        return mf_loss





