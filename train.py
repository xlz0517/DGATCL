import torch
import numpy as np
import os
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from gnn import GNN, ContrastiveLoss
from load_data import DataLoader
from tqdm import tqdm
from scipy.stats import rankdata

class Base(object):
    def __init__(self, args, loader):
        self.model = GNNModel(args, loader).cuda()
        self.loader = loader
        self.n_ent, self.n_rel = loader.n_ent, loader.n_rel
        self.n_batch, self.n_tbatch = args.n_batch, args.n_tbatch
        self.n_train, self.n_valid, self.n_test = loader.n_train, loader.n_valid, loader.n_test
        self.n_layer = args.n_layer
        self.args = args
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.bceloss = torch.nn.BCELoss()
        self.CL = ContrastiveLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)
        self.rank_loss = torch.nn.CrossEntropyLoss()

        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate) if args.scheduler == 'exp' else None
        self.modelName = f'{args.n_layer}-layers'
        self.modelName += ''.join([f'-{args.n_node_topk[i] if isinstance(args.n_node_topk, list) else args.n_node_topk}' for i in range(args.n_layer)])

    def _update(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)

    def loadModel(self, filePath, layers=-1):
        print(f'Load weight from {filePath}')
        assert os.path.exists(filePath)
        checkpoint = torch.load(filePath, map_location=torch.device(f'cuda:{self.args.gpu}'))
        if layers != -1:
            extra_layers = self.model.gnn_layers[layers:]
            self.model.gnn_layers = self.model.gnn_layers[:layers]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.gnn_layers += extra_layers
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train_batch(self):
        epoch_loss = 0
        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)
        self.model.train()
        start_time = time.time()

        for i in tqdm(range(n_batch)):
            start, end = i * batch_size, min(self.loader.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores, hidden, nodes = self.model(triple[:, 0], triple[:, 1])
            pos_scores = scores[[torch.arange(len(scores)).cuda(), torch.LongTensor(triple[:, 2]).cuda()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            bce_loss = torch.sum(-pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1)))

            new_triple = torch.tensor(triple, device=torch.device(f'cuda:{self.args.gpu}'))
            features = torch.cat((new_triple[:, 0].unsqueeze(1), new_triple[:, 2].unsqueeze(1)), dim=1).unsqueeze(-1).float().to(torch.device(f'cuda:{self.args.gpu}'))
            neighbors, sampled_edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), batch_size, mode='train')
            negative_tail_emb = hidden[old_nodes_new_idx[:batch_size]]
            negative_features = torch.stack([hidden, negative_tail_emb], dim=1)
            CL_loss = self.CL(features, negative_features, labels=torch.LongTensor(triple[:, 2]).cuda())

            total_loss = 0.2 * bce_loss + 0.8 * CL_loss
            total_loss.backward()
            self.optimizer.step()

            for p in self.model.parameters():
                p.data.copy_(torch.nan_to_num(p.data))

            epoch_loss += bce_loss.item()

        self.t_time += time.time() - start_time
        if self.args.scheduler == 'exp':
            self.scheduler.step()

        self.loader.shuffle_train()

    def eval(self, verbose=True, eval_val=True, eval_test=False, recordDistance=False):
        batch_size = self.n_tbatch
        n_valid_data = self.n_valid
        n_test_data = self.n_test
        ranking = []

        def eval_data(n_data, eval_type):
            n_batch = n_data // batch_size + (n_data % batch_size > 0)
            result = []
            iterator = tqdm(range(n_batch)) if verbose else range(n_batch)
            for i in iterator:
                start, end = i * batch_size, min(n_data, (i + 1) * batch_size)
                batch_idx = np.arange(start, end)
                subs, rels, objs = self.loader.get_batch(batch_idx, data=eval_type)
                scores, _, _ = self.model(subs, rels, mode=eval_type)
                scores = scores.data.cpu().numpy()
                filters = [np.zeros((self.n_ent,)) for _ in range(len(subs))]
                for i, (sub, rel) in enumerate(zip(subs, rels)):
                    filt = self.loader.filters[(sub, rel)]
                    filters[i][np.array(filt)] = 1
                ranks = rank(scores, objs, np.array(filters))
                result += ranks
            return np.array(result)

        if eval_val:
            v_ranking = eval_data(n_valid_data, 'valid')
            v_mrr, v_mr, v_h1, v_h3, v_h10 = calculate(v_ranking)
        else:
            v_mrr = v_mr = v_h1 = v_h3 = v_h10 = 0

        if eval_test:
            t_ranking = eval_data(n_test_data, 'test')
            t_mrr, t_mr, t_h1, t_h3, t_h10 = calculate(t_ranking)
        else:
            t_mrr = t_mr = t_h1 = t_h3 = t_h10 = -1

        i_time = time.time() - self.t_time
        out_str = f'[VALID] MRR: {v_mrr:.4f} MR: {v_mr:.4f} H@1: {v_h1:.4f} H@3: {v_h3:.4f} H@10: {v_h10:.4f} ' \
                  f'[TEST] MRR: {t_mrr:.4f} MR: {t_mr:.4f} H@1: {t_h1:.4f} H@3: {t_h3:.4f} H@10: {t_h10:.4f} ' \
                  f'[TIME] train: {self.t_time:.4f} inference: {i_time:.4f}'
        
        result_dict = {'v_mrr': v_mrr, 'v_mr': v_mr, 'v_h1': v_h1, 'v_h3': v_h3, 'v_h10': v_h10,
                       't_mrr': t_mrr, 't_h1': t_h1, 't_h3': t_h3, 't_h10': t_h10}
        return result_dict, out_str
        
def rank(score_matrix, label_mask, filter_mask):
    norm_scores = score_matrix - np.min(score_matrix, axis=1, keepdims=True) + 1e-8
    full_ranks = rankdata(-norm_scores, method='average', axis=1)
    filtered_ranks = rankdata(-norm_scores * filter_mask, method='min', axis=1)
    raw_ranks = (full_ranks - filtered_ranks + 1) * label_mask
    return list(raw_ranks[raw_ranks != 0])

def calculate(rank_list):
    rank_array = np.asarray(rank_list)
    reciprocal_rank = 1. / rank_array
    mrr = reciprocal_rank.mean()
    mr = rank_array.mean()
    hits_at = lambda k: np.mean(rank_array <= k)
    return mrr, mr, hits_at(1), hits_at(3), hits_at(10)
