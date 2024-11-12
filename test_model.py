# test: parameters shape, output dim.
# loss scope, exception

import unittest
import torch
import torch.nn as nn
from model_torch import GraphConv, Disentangle, MRAM


# class TestGraphConv(unittest.TestCase):
#     def test_conv(self):
#         channel = 16
#         n_user, n_item = 3, 3
#         n_node = n_item + n_user
#         conv_layers = 2
#
#         all_emb = torch.randn(n_user + n_item, channel)
#         print(all_emb)
#         row_id = torch.tensor([0, 0, 1, 2])
#         col_id = torch.tensor([0, 2, 0, 1])
#
#         values = torch.tensor([1., 1., 1., 1.])
#         # n_user * n_item
#         adj_mat = torch.sparse_coo_tensor(torch.stack([row_id, col_id]), values, (n_node, n_node))
#         graph_conv = GraphConv(all_emb, adj_mat, conv_layers, n_user, n_item)
#         user_e, item_e = graph_conv()
#         self.assertEqual(user_e.shape, (n_user, channel))
#         self.assertEqual(item_e.shape, (n_item, channel))


# class TestDisentangle(unittest.TestCase):
#     def test_forward(self):
#         channel = 16
#         n_user = 3
#         n_intent = 2
#         n_relation = 4
#         user_emb = torch.randn(n_user, channel) # (3, 16)
#         relation_emb = torch.randn(n_relation, channel)     # (4, 16)
#         disen_weight_att = torch.randn(n_intent, n_relation) # (2, 4)
#         #
#         disentangle_model = Disentangle(channel, n_user, n_intent, n_relation)
#         # user_int, r_int_emb = disentangle_model.forward(user_emb, relation_emb, disen_weight_att)
#         user_int = disentangle_model.forward(user_emb, relation_emb, disen_weight_att)
#         self.assertEqual(user_int.shape, (n_user, n_intent, channel))
#         # self.assertEqual(r_int_emb.shape, (n_relation, n_intent, channel))


class TestMRAM(unittest.TestCase):
    def setUp(self):
        # 创建一些示例输入数据，用于测试函数
        self.batch_size = 10
        self.users = torch.randn(self.batch_size, 16)  # 假设嵌入维度为64
        self.pos_items = torch.randn(self.batch_size, 16)
        self.neg_items = torch.randn(self.batch_size, 16)

    def test_create_bpr_loss(self):
        # 创建MRAM模型的实例（这里只需要调用create_bpr_loss函数，所以可以不传入完整的模型参数）
        model = type('MRAM', (), {'create_bpr_loss': lambda self, u, p, n: MRAM.create_bpr_loss(self, u, p, n)})()

        # 调用要测试的函数
        loss = model.create_bpr_loss(self.users, self.pos_items, self.neg_items)
        print(loss)
        # 验证输出是否为标量（即损失值应该是一个单一的数值）
        self.assertEqual(loss.ndim, 0, "Loss should be a scalar.")

        # 验证损失值是否在合理的数值范围内（这里只是简单示例，可根据实际情况调整范围）
        self.assertTrue(not torch.isnan(loss), "Loss should not be NaN.")
        self.assertTrue(not torch.isinf(loss), "Loss should not be infinite.")


if __name__ == "__main__":
    # unittest.main()
    # print("unit test done")
    # data_config = {
    #     'n_users': 10,
    #     'n_items': 10,
    #     'n_relations': 4,
    #     'n_entities': 15,   # kg entity:5, + item:10
    #     'n_nodes': 25   # n_entities + n_users
    # }
    # args_config = type('ArgsConfig', (), {
    #     'n_intent': 2,
    #     'embed_size': 8,
    #     'n_layer': 2,
    #     'cuda': False,
    #     'gpu_id': 0
    # })()
    #
    # # 创建一些示例测试数据
    # batch_size = 5
    # test_batch = {
    #     'users': torch.randint(0, data_config['n_users'], (batch_size,)),
    #     'pos_items': torch.randint(0, data_config['n_items'], (batch_size,)),
    #     'neg_items': torch.randint(0, data_config['n_items'], (batch_size,))
    # }
    #
    # model = MRAM(data_config, args_config, None, None)  # 这里假设图和邻接矩阵先传入None，实际应用中需传入正确的值
    # if torch.cuda.is_available() and args_config.cuda:
    #     model.to(model.device)
    # output = model.forward(test_batch)  # loss
    # assert output.shape == (), "Output shape should be a scalar (average loss), but got {}".format(output.shape)
    # assert not torch.isnan(output), "Output loss is NaN"
    # assert not torch.isinf(output), "Output loss is infinite"
    # print("test done")




