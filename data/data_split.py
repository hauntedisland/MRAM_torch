import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

def get_dict_from_data(data):
    interaction_dict = {}
    for i in data:
        if i[2] == 1:
            if i[0] in interaction_dict.keys():
                interaction_dict[i[0]].append(i[1])
            else:
                interaction_dict[i[0]] = [i[1]]
    for i in interaction_dict.keys():
        interaction_dict[i].sort()

    return [(k, interaction_dict[k]) for k in sorted(interaction_dict.keys())]


def train_test_split(rating_file, test_ratio):
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)
    n_ratings = rating_np.shape[0]
    test_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
    left = set(range(n_ratings)) - set(test_indices)

    train_dict = get_dict_from_data(rating_np[list(left)])
    test_dict = get_dict_from_data(rating_np[test_indices])

    write_dict_to_file(train_dict, 'music/train.txt')
    write_dict_to_file(test_dict, 'music/test.txt')


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)

# def write_dict_to_file(interaction_dict, file):     # for train, test file
#     with open(file, mode='w') as f:
#         for i in interaction_dict:
#             f.write(str(i[0]) + " ")
#             for j in i[1]:
#                 if j is i[1][-1]:
#                     f.write(str(j))
#                 else:
#                     f.write(str(j) + " ")
#             f.write("\n")

def write_dict_to_file(interaction_dict, file):
    """
    :param interaction_dict: key(rid=0): [u_id, i_id]
    :param file: change dict to triplet form, save path
    """
    with open(file, mode='w') as f:
        interaction_dict_ui = interaction_dict[0]  # only ui relation. list
        for pair in interaction_dict_ui:  # iterate (uid, iid)
            uid, iid = pair
            f.write(str(uid) + " " + str(0) + " " + str(iid) + '\n')


def ui2triplets(train_data, file):
    rd = defaultdict(list)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])
    write_dict_to_file(rd, file)


def regroup(in_file, out_file, n_users):
    """
    redirect triplets file. let ID be user-item-entity. all included
    """
    df = pd.read_csv(in_file, sep=" ", names=["hid", "rid", "tid"], dtype={"hid": int})
    df["hid"] = df["hid"] + n_users
    # save
    df.to_csv(out_file, sep=" ", index=False, header=False)

def reformat(in_file, out_file):
    df = pd.read_csv(in_file, sep="\t", names=["hid", "rid", "tid"])
    df.to_csv(out_file, sep=' ', index=False, header=False)

# 合并两个文档
def unify_triplet(file1, file2, out_file):
    df1 = pd.read_csv(file1, sep=" ", names=["hid", "rid", "tid"])
    df2 = pd.read_csv(file2, sep=" ", names=["hid", "rid", "tid"])
    # edit rid of KG
    df1["rid"] += 1
    result = pd.concat([df1, df2], ignore_index=True)
    result.to_csv(out_file, sep=" ", index=False, header=False)


def show_id(file):
    df = pd.read_csv(file, sep=" ", names=["hid", "rid", "tid"])
    print(df.min())
    print(df.max())
    n_entities = max(max(df["hid"]), max(df["tid"])) + 1
    print(n_entities)


def read_triplets(file):
    n_nodes = 0
    triplets = np.loadtxt(file, dtype=np.int32)
    n_nodes = max(n_nodes, max(max(triplets[:, 0]), max(triplets[:, 2])) + 1)
    print(f'数据集节点个数: {n_nodes}')


if __name__ == '__main__':
    # directory = '/home/jianmeng/llc/MRAM_torch/data/lastfm_wxkg/'
    directory = '/home/MRAM_torch/data/yelp2018_kg/'

    # 1. split test, train from ratings
    # train_test_split(directory + 'ratings_final', test_ratio=0.2)

    # KG
    # show_id(directory + 'kg.txt')

    # 2. ui train to triplets
    cf = read_cf(directory + 'train.txt')
    ui2triplets(cf, directory + 'ui_tri.txt')

    # 3. kg regroup for ckg triplets
    """kg.txt format: start from iid"""
    # reformat(directory + 'kg_final.txt', directory + 'kg.txt')    # unneccsary
    n_users = 45919 # edit with different datasets
    regroup(directory + 'kg.txt', directory + 'kg_tri.txt', n_users)

    # 4. unify kg_tri and ui_tri
    unify_triplet(file1=directory+'kg_tri.txt', file2=directory+'ui_tri.txt', out_file=directory+'triplets.txt')

