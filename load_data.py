import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.task_dir = task_dir = args.data_path

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip()
                self.entity2id[entity] = n_ent
                n_ent += 1

        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            n_rel = 0
            for line in f:
                relation = line.strip()
                self.relation2id[relation] = n_rel
                n_rel += 1

        self.n_ent = n_ent
        self.n_rel = n_rel

        
        self.filters = defaultdict(lambda:set())
        self.fact_triple  = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        self.test_triple  = self.read_triples('test.txt')
        self.all_triple = np.concatenate([np.array(self.fact_triple), np.array(self.train_triple)], axis=0)
        self.tmp_all_triple = np.concatenate([np.array(self.fact_triple), np.array(self.train_triple), np.array(self.valid_triple), np.array(self.test_triple)], axis=0)
        
        
        self.fact_data  = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data  = self.double_triple(self.test_triple)

        self.shuffle_train()
        self.load_graph(self.fact_data)
        self.load_test_graph(self.double_triple(self.fact_triple)+self.double_triple(self.train_triple))
        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q,  self.test_a  = self.load_query(self.test_data)
        
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h,r,t])
                self.filters[(h,r)].add(t)
                self.filters[(t,r+self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 
        return triples + new_triples

    def load_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.KG = np.concatenate([np.array(triples), idd], 0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_ent))

    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:,0])), shape=(self.tn_fact, self.n_ent))

    def load_query(self, triples):
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, batchsize, mode='train'):
        if mode == 'train':
            KG    = self.KG
            M_sub = self.M_sub
        else:
            KG    = self.tKG
            M_sub = self.tM_sub

        node_1hot     = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_ent, nodes.shape[0]))
        edge_1hot     = M_sub.dot(node_1hot)
        edges         = np.nonzero(edge_1hot)
        
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        

        mask = sampled_edges[:,2] == (self.n_rel*2)
        old_nodes_new_idx = tail_index[mask].sort()[0]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':
            return np.array(self.train_data)[batch_idx]
        if data=='valid':
            query, answer = np.array(self.valid_q), np.array(self.valid_a)
        if data=='test':
            query, answer = np.array(self.test_q), np.array(self.test_a)
        
        subs = []
        rels = []
        objs = []
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def negative_sampling(self, triple):
        
        entity_ids = list(self.entity2id.values())
        pos_tail = triple[:,2]
        mask = np.ones([self.n_ent], dtype=np.bool)
        mask[pos_tail] = 0
        
        neg_tail = np.int32(
            np.random.choice(
                np.array(entity_ids)[mask],
                size=pos_tail.size * self.args.neg_sample_ratio,
                replace=False
            )
        ).reshape([-1])

        return neg_tail[:pos_tail.size]

    def shuffle_train(self):
        all_triple = self.all_triple
        n_all = len(all_triple)
        rand_idx = np.random.permutation(n_all)
        all_triple = all_triple[rand_idx]

        bar = int(n_all * self.args.fact_ratio)
        self.fact_data = np.array(self.double_triple(all_triple[:bar].tolist()))
        self.train_data = np.array(self.double_triple(all_triple[bar:].tolist()))

        if self.args.remove_1hop_edges:
            print('==> removing 1-hop links...')
            tmp_index = np.ones((self.n_ent, self.n_ent))
            tmp_index[self.train_data[:, 0], self.train_data[:, 2]] = 0
            save_facts = tmp_index[self.fact_data[:, 0], self.fact_data[:, 2]].astype(bool)
            self.fact_data = self.fact_data[save_facts]
            print('==> done')

        self.n_train = len(self.train_data)
        self.load_graph(self.fact_data)
