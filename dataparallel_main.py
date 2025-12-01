import logging
import os.path
import random
import sys

import torch
import numpy as np

from time import time

from matplotlib import pyplot as plt
from prettytable import PrettyTable

from util.parser import parse_args
from util.data_loader import load_data
from model_torch import MRAM
from util.evaluate import test
from util.helper import early_stopping

import MRAM_torch.util.kg_data_loader as kg_loader

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
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set)).to(device)
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

    """read args"""
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, ckg_dict, kg_triplet, n_params, graph, adj_mats, adj_mean_mat = load_data(args)

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

    """load pretrain data"""
    if args.pretrain:
        if not os.path.exists(args.pretrain_path):
            os.makedirs(args.pretrain_path)
        output_file = args.pretrain_path + '/kgin_' + args.dataset + '.pretrain'
        pretrained_embeddings = torch.load(output_file, map_location=device)
    else:
        pretrained_embeddings = None

    """define model"""
    model = MRAM(n_params, args, graph, adj_mats, adj_mean_mat, pretrained_embeddings).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    print(model)
    """define optimizer"""
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
            batch = get_feed_dict(train_cf_pairs, s, s + args.batch_size, user_dict['train_user_set'])
            batch_loss, batch_cor = model(batch)
            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            cor_loss += batch_cor
            s += args.batch_size
        train_e_t = time()

        """batch test"""
        model.eval()
        val_loss, val_cor_loss, val_s = 0, 0, 0
        index = np.arange(len(test_cf))
        np.random.shuffle(index)
        test_cf_pairs = test_cf_pairs[index]
        with torch.no_grad():
            while val_s + args.batch_size <= len(test_cf_pairs):
                val_batch = get_feed_dict(test_cf_pairs, val_s, val_s + args.batch_size, user_dict['test_user_set'])
                val_batch_loss, val_batch_cor = model(val_batch)
                val_loss += val_batch_loss.item()
                val_cor_loss += val_batch_cor.item()
                val_s += args.batch_size
        avg_train_loss = loss / (len(train_cf) / args.batch_size)
        avg_val_loss = val_loss / (len(test_cf_pairs) / args.batch_size)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_val_loss)

        """metric test"""
        if epoch % 3 == 2 or epoch == 0:
            test_s_t = time()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "testing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row([epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']])
            print(train_res)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=5)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')
                else:
                    torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

        else:
            print('using time %.4f, training loss at epoch %d: %.4f, cor: %.6f' % (train_e_t - train_s_t, epoch, loss.item(), cor_loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))