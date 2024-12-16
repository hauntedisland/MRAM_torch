import torch
import random
import numpy as np


def generate_kg_batch(kg_dict, batch_size, highest_neg_idx):
    exist_heads = kg_dict.keys()
    if batch_size <= len(exist_heads):
        batch_head = random.sample(exist_heads, batch_size)
    else:   # batch_size > exist_heads
        batch_head = [random.choice(list(exist_heads)) for _ in range(batch_size)]

    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for h in batch_head:
        relation, pos_tail = sample_pos_triples_for_h(kg_dict, h, 1)
        batch_relation += relation
        batch_pos_tail += pos_tail

        neg_tail = sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
        batch_neg_tail += neg_tail

    batch_head = torch.LongTensor(batch_head)
    batch_relation = torch.LongTensor(batch_relation)
    batch_pos_tail = torch.LongTensor(batch_pos_tail)
    batch_neg_tail = torch.LongTensor(batch_neg_tail)
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


def sample_pos_triples_for_h(kg_dict, head, n_sample_pos_triples):
    pos_triples = kg_dict[head]
    n_pos_triples = len(pos_triples)

    sample_relations, sample_pos_tails = [], []
    while True:
        if len(sample_relations) == n_sample_pos_triples:
            break

        pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
        tail = pos_triples[pos_triple_idx][0]
        relation = pos_triples[pos_triple_idx][1]

        if relation not in sample_relations and tail not in sample_pos_tails:
            sample_relations.append(relation)
            sample_pos_tails.append(tail)
    return sample_relations, sample_pos_tails


def sample_neg_triples_for_h(kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
    pos_triples = kg_dict[head]

    sample_neg_tails = []
    while True:
        if len(sample_neg_tails) == n_sample_neg_triples:
            break

        tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
        if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
            sample_neg_tails.append(tail)
    return sample_neg_tails
