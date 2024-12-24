import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

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

def write_dict_to_file(interaction_dict, file):
    """
    :param interaction_dict: key(rid=0): [u_id, i_id]
    :param file: change dict to triplet form, save path
    """
    with open(file, mode='w') as f:
        interaction_dict_ui = interaction_dict[0]   # only ui relation. list
        for pair in interaction_dict_ui:  # iterate (uid, iid)
            uid, iid = pair
            f.write(str(uid) + " " + str(0) + " " + str(iid) + '\n')


def ui2triplets(train_data, file):
    rd = defaultdict(list)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])
    write_dict_to_file(rd, file)

def regroup(in_file, out_file):
    """
    redirect triplets file. let ID be user-item-entity. all included
    """
    n_users = 6036
    df = pd.read_csv(in_file, sep=" ", names=["hid", "rid", "tid"])
    # print(df.columns)
    df["hid"] = df["hid"] + n_users
    df["tid"] = df["tid"] + n_users
    # save
    df.to_csv(out_file, sep=" ", index=False, header=False)


if __name__ == '__main__':
    directory = '/home/jianmeng/llc/MRAM_torch/data/movie/'
    # in_file = directory + 'train.txt'
    # out_file = directory + 'train_tri.txt'
    # KG
    in_file = directory + 'kg.txt'
    out_file = directory + 'kg_tri.txt'
    regroup(in_file, out_file)
    # train_cf = read_cf(directory + 'train.txt')
    # ui2triplets(train_cf, file=out_file)

