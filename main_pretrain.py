import logging
import os.path
import random
import sys
import argparse

import torch
import numpy as np

from time import time

from matplotlib import pyplot as plt
from prettytable import PrettyTable

from util.parser import parse_args
import util.trans_data_loader as kg_loader
from util.trans_data_loader import load_data as load_kg_data
from util.data_loader import load_data as load_ui_data
from model_torch import MRAM
from modules.CKG import CKGGCN
from modules.LightGCN import LightGCN
from util.evaluate import test
from util.helper import early_stopping


n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)  # 根据batch从原始ui交互数据里选当前batch
    feed_dict['users'] = entity_pairs[:, 0].long()
    feed_dict['pos_items'] = entity_pairs[:, 1].long()
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict

def get_kg_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)  # 根据batch从原始ui交互数据里选当前batch
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """args"""
    # global args, device
    parser = argparse.ArgumentParser(description="pretrain")

    # ===== data ===== #
    parser.add_argument("--dataset", nargs="?", default="movie", help="Choose a dataset:[book,last-fm,amazon-book,alibaba,music,movie,kgcl_book]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")
    parser.add_argument("--pretrain_path", default="pretrain/")

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--kg_epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--kg_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--kg_dim', type=int, default=64, help='KG embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--ssm', type=float, default=0.01, help='SSM loss weight')
    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    # parser.add_argument('--layer_num_kg', default=1, type=int)      # RGAT
    # parser.add_argument('--res_lambda', type=float, default=0.5)    # RGAT 残差链接
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--encode_layer", type=int, default=2, help="layer for GCN/LightGCN")
    parser.add_argument("--kg_encode_layer", type=int, default=1, help="layer for RGCN")
    parser.add_argument("--decode_layer", type=int, default=2, help="layer for disentangle module")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20]', help='Output sizes of every layer') # change
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    # ===== relation context ===== #
    parser.add_argument("--n_intent", type=int, default=4, help="number of users' intent")
    parser.add_argument("--topk", type=int, default=3, help="select top-k similarities for subgraph(EGLN)")

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    args = parser.parse_args()

    print("args.lr",args.lr)
    print("args.l2",args.l2)
    print("args.batch_size",args.batch_size)
    print("args.encode_layer",args.encode_layer)

    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    # train_cf, test_cf, user_dict, n_params, ckg_graph, [adj_mat, ui_mat, iu_mat], adj_mean_mat = load_ui_data(args)
    train_cf, test_cf, user_dict, kg_dict, kg_triplet, n_params, graph, adj_mean_mat = load_kg_data(args)

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    print("--data loaded--")
    print(f"用户数量: {n_users}")
    print(f"物品数量: {n_items}")
    print(f"实体数量: {n_entities}")
    print(f"节点数量: {n_nodes}")
    print(f"关系数量: {n_relations}")
    print(f"训练集大小: {len(train_cf)}")
    print(f"测试集大小: {len(test_cf)}")

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """kg data"""
    kg_pairs = torch.LongTensor(kg_triplet)

    """define model"""
    # ui_model = LightGCN(n_params, args, adj_mean_mat).to(device)
    model = CKGGCN(n_params, args, graph, adj_mean_mat).to(device)

    """define optimizer"""
    # optimizer = torch.optim.Adam(ui_model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print("... start training ...")
    train_losses = []
    test_losses = []
    for epoch in range(args.epoch):
        """training CF"""
        model.train()
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        loss, cor_loss, s = 0, 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            """model training"""
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'])
            batch_loss = model(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size
        train_e_t = time()
        
        """training KG"""
        # kg_model.train()
        # trans_s_t = time()
        # kg_loss = 0
        # n_kg_batch = len(kg_triplet) // args.kg_batch_size + 1
        # for iter in range(1, n_kg_batch + 1):
        #     kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = kg_loader.generate_kg_batch(
        #         kg_dict, args.kg_batch_size, n_entities)
        #     kg_batch_head = kg_batch_head.to(device)
        #     kg_batch_relation = kg_batch_relation.to(device)
        #     kg_batch_pos_tail = kg_batch_pos_tail.to(device)
        #     kg_batch_neg_tail = kg_batch_neg_tail.to(device)
        
        #     kg_batch_loss = kg_model.calculate_loss_transE(kg_batch_head, kg_batch_relation, kg_batch_pos_tail,
        #                                                 kg_batch_neg_tail)
        
        #     if np.isnan(kg_batch_loss.cpu().detach().numpy()):
        #         logging.info(
        #             'ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter,
        #                                                                                          n_kg_batch))
        #         sys.exit()
        
        #     kg_batch_loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     kg_loss += kg_batch_loss
        
        # trans_e_t = time()
        # if epoch % 3 == 2 or epoch == 0:
        #     average_kg_loss = kg_loss / n_kg_batch
        #     kg_res = PrettyTable()
        #     kg_res.field_names = ["Epoch", "training time", "Loss"]
        #     kg_res.add_row([epoch, trans_e_t - trans_s_t, average_kg_loss.item()])
        #     print(kg_res)
            
        """metric test"""
        if epoch % 3 == 2 or epoch == 0:
            test_s_t = time()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "testing time", "Loss", "recall", "ndcg", "precision",
                                     "hit_ratio"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'],
                 ret['precision'], ret['hit_ratio']]
            )
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=5)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:
                if not os.path.exists(args.out_dir):
                    os.makedirs(args.out_dir, exist_ok=True)
                    print(f"Created directory: {args.out_dir}")

                user_emb_path = os.path.join(args.out_dir, f'CKGGCN_{args.dataset}_user_emb.npy')
                item_emb_path = os.path.join(args.out_dir, f'CKGGCN_{args.dataset}_item_emb.npy')
                r_emb_path = os.path.join(args.out_dir, f'CKGGCN_{args.dataset}_relation_emb.npy')

                np.save(user_emb_path, model.user_embed.weight.detach().cpu().numpy())
                np.save(item_emb_path, model.item_embed.weight.detach().cpu().numpy())
                np.save(r_emb_path, model.relation_embed.weight.detach().cpu().numpy())
                # torch.save(ui_model.state_dict(), args.out_dir + 'LightGCN_' + args.dataset + '.ckpt')
                # user_emb = ui_model.user_embedding.weight.detach().cpu().numpy()
                # item_emb = ui_model.item_embedding.weight.detach().cpu().numpy()
                
                # np.save(os.path.join(args.out_dir, 'LightGCN_' + args.dataset + 'user_emb.npy'), user_emb)
                # np.save(os.path.join(args.out_dir, 'LightGCN_' + args.dataset + 'item_emb.npy'), item_emb)
                # print(f"Embeddings saved at epoch {epoch}")

        else:
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))