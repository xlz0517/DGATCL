import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.task_dir = args.data_path

        self.entity2id = self._load_vocab('entities.txt')
        self.relation2id = self._load_vocab('relations.txt')

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)

        self.filters = defaultdict(set)
        self.fact_triples = self.read_triples('facts.txt')
        self.train_triples = self.read_triples('train.txt')
        self.valid_triples = self.read_triples('valid.txt')
        self.test_triples = self.read_triples('test.txt')

        self.all_triples = np.concatenate([self.fact_triples, self.train_triples], axis=0)
        self.tmp_all_triples = np.concatenate([self.fact_triples, self.train_triples, self.valid_triples, self.test_triples], axis=0)

        self.fact_data = self.double_triple(self.fact_triples)
        self.train_data = np.array(self.double_triple(self.train_triples))
        self.valid_data = self.double_triple(self.valid_triples)
        self.test_data = self.double_triple(self.test_triples)

        self.shuffle_train()
        self.load_graph(self.fact_data)
        self.load_test_graph(self.double_triple(self.fact_triples) + self.double_triple(self.train_triples))
        self.valid_queries, self.valid_answers = self.load_query(self.valid_data)
        self.test_queries, self.test_answers = self.load_query(self.test_data)

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_queries)
        self.n_test = len(self.test_queries)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        print(f'n_train: {self.n_train}, n_valid: {self.n_valid}, n_test: {self.n_test}')

    def _load_vocab(self, filename):
        vocab = {}
        with open(os.path.join(self.task_dir, filename)) as f:
            for idx, line in enumerate(f):
                vocab[line.strip()] = idx
        return vocab

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                head, relation, tail = line.strip().split()
                head, relation, tail = self.entity2id[head], self.relation2id[relation], self.entity2id[tail]
                triples.append([head, relation, tail])
                self.filters[(head, relation)].add(tail)
                self.filters[(tail, relation + self.n_rel)].add(head)
        return triples

    def double_triple(self, triples):
        return triples + [[tail, relation + self.n_rel, head] for head, relation, tail in triples]

    def load_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent), 1),
                              2 * self.n_rel * np.ones((self.n_ent, 1)),
                              np.expand_dims(np.arange(self.n_ent), 1)], axis=1)
        self.KG = np.concatenate([np.array(triples), idd], axis=0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones(self.n_fact), 
                                 (np.arange(self.n_fact), self.KG[:, 0])), 
                                shape=(self.n_fact, self.n_ent))

    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent), 1),
                              2 * self.n_rel * np.ones((self.n_ent, 1)),
                              np.expand_dims(np.arange(self.n_ent), 1)], axis=1)
        self.tKG = np.concatenate([np.array(triples), idd], axis=0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones(self.tn_fact),
                                  (np.arange(self.tn_fact), self.tKG[:, 0])),
                                 shape=(self.tn_fact, self.n_ent))

    def load_query(self, triples):
        trip_hr = defaultdict(list)
        for head, relation, tail in triples:
            trip_hr[(head, relation)].append(tail)

        queries = list(trip_hr.keys())
        answers = [np.array(trip_hr[key]) for key in queries]
        return queries, answers

    def get_neighbors(self, nodes, batchsize, mode='train'):
        KG, M_sub = (self.KG, self.M_sub) if mode == 'train' else (self.tKG, self.tM_sub)

        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(self.n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)

        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]], axis=1)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], dim=1)

        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        old_nodes_new_idx = tail_index[mask].sort()[0]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data == 'train':
            return np.array(self.train_data)[batch_idx]
        if data == 'valid':
            query, answer = np.array(self.valid_queries), np.array(self.valid_answers)
        if data == 'test':
            query, answer = np.array(self.test_queries), np.array(self.test_answers)

        subjects = query[batch_idx, 0]
        relations = query[batch_idx, 1]
        objects = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objects[i][answer[batch_idx[i]]] = 1
        return subjects, relations, objects

    def negative_sampling(self, triple):
        entity_ids = list(self.entity2id.values())
        positive_tail = triple[:, 2]
        mask = np.ones(self.n_ent, dtype=bool)
        mask[positive_tail] = 0

        negative_tail = np.random.choice(np.array(entity_ids)[mask], 
                                         size=positive_tail.size * self.args.neg_sample_ratio, 
                                         replace=False).astype(np.int32)

        return negative_tail[:positive_tail.size]

    def shuffle_train(self):
        rand_idx = np.random.permutation(len(self.all_triples))
        shuffled_triples = self.all_triples[rand_idx]

        split_idx = int(len(shuffled_triples) * self.args.fact_ratio)
        self.fact_data = np.array(self.double_triple(shuffled_triples[:split_idx].tolist()))
        self.train_data = np.array(self.double_triple(shuffled_triples[split_idx:].tolist()))

        if self.args.remove_1hop_edges:
            tmp_index = np.ones((self.n_ent, self.n_ent))
            tmp_index[self.train_data[:, 0], self.train_data[:, 2]] = 0
            save_facts = tmp_index[self.fact_data[:, 0], self.fact_data[:, 2]].astype(bool)
            self.fact_data = self.fact_data[save_facts]

        self.n_train = len(self.train_data)
        self.load_graph(self.fact_data)
