import torch
import numpy as np
import time
import os

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from gnnmodel import GNNModel, ContrastiveLoss
from utils import *
from tqdm import tqdm
from load_data import DataLoader


class Base(object):
    def __init__(self, args, loader):
        self.model = GNNModel(args, loader)
        self.model.cuda()
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch
        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test = loader.n_test
        self.n_layer = args.n_layer
        self.args = args
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)

        self.bceloss = torch.nn.BCELoss()
        self.loader=DataLoader
        self.CL = ContrastiveLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)
        self.rank_loss = torch.nn.CrossEntropyLoss()

        if self.args.scheduler == 'exp':
            self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        else:
            raise NotImplementedError(f'==> [Error] {self.scheduler} scheduler is not supported yet.')

        self.t_time = 0
        self.lastSaveGNNPath = None
        self.modelName = f'{args.n_layer}-layers'
        for i in range(args.n_layer):
            i_n_node_topk = args.n_node_topk if 'int' in str(type(args.n_node_topk)) else args.n_node_topk[i]
            self.modelName += f'-{i_n_node_topk}'
        print(f'==> model name: {self.modelName}')

    def _update(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)

    def saveModelToFiles(self, best_metric, deleteLastFile=True):
        savePath = f'{self.loader.task_dir}/saveModel/{self.modelName}-{best_metric}.pt'
        print(f'Save checkpoint to : {savePath}')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mrr': best_metric,
        }, savePath)

        if deleteLastFile and self.lastSaveGNNPath != None:
            print(f'Remove last checkpoint: {self.lastSaveGNNPath}')
            os.remove(self.lastSaveGNNPath)

        self.lastSaveGNNPath = savePath

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

    def train_batch(self, ):
        epoch_loss = 0
        i = 0
        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)
        t_time = time.time()
        self.model.train()

        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = min(self.loader.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores, hidden, nodes = self.model(triple[:, 0], triple[:, 1])
            pos_scores = scores[[torch.arange(len(scores)).cuda(), torch.LongTensor(triple[:, 2]).cuda()]]

            # 二进制交叉熵损失
            max_n = torch.max(scores, 1, keepdim=True)[0]
            bce_loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1)))


            new_triple = torch.tensor(triple, device=torch.device(f'cuda:{self.args.gpu}'))

            features = torch.cat((new_triple[:, 0].unsqueeze(1), new_triple[:, 2].unsqueeze(1)), dim=1).unsqueeze(
                -1).float().to(torch.device(f'cuda:{self.args.gpu}'))
            neighbors, sampled_edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(),
                                                                                    batch_size, mode='train')

            batch_size = head_emb.size(0)
            negative_tail_emb = hidden[old_nodes_new_idx[:batch_size]]
            negative_features = torch.stack([head_emb, negative_tail_emb], dim=1)  # [B, 2, dim]
            # 对比损失
            CL_loss = self.CL(features, negative_features, labels=torch.LongTensor(triple[:, 2]).cuda())  # triple[:, 2]

            # 联合训练对比损失和二进制交叉熵损失
            total_loss = 0.8 * bce_loss + 0.2 * CL_loss

            total_loss.backward()
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += bce_loss.item()

        self.t_time += time.time() - t_time

        if self.args.scheduler == 'exp':
            self.scheduler.step()

        self.loader.shuffle_train()

        return

    def evaluate(self, verbose=True, eval_val=True, eval_test=False, recordDistance=False):
        batch_size = self.n_tbatch
        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        i_time = time.time()

        # - - - - - - val set - - - - - - 
        if not eval_val:
            v_mrr, v_mr, v_h1, v_h3, v_h10 = 0, 0, 0, 0, 0
        else:
            iterator = tqdm(range(n_batch)) if verbose else range(n_batch)
            for i in iterator:
                start = i * batch_size
                end = min(n_data, (i + 1) * batch_size)
                batch_idx = np.arange(start, end)
                subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
                scores, hidden, nodes = self.model(subs, rels, mode='valid')
                scores = scores.data.cpu().numpy()

                filters = []
                for i in range(len(subs)):
                    filt = self.loader.filters[(subs[i], rels[i])]
                    filt_1hot = np.zeros((self.n_ent,))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)

                    # scores / objs / filters: [batch_size, n_ent]
                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks

            ranking = np.array(ranking)
            v_mrr, v_mr, v_h1, v_h3, v_h10 = cal_performance(ranking)

        # - - - - - - test set - - - - - - 
        if not eval_test:
            t_mrr, t_mr, t_h1, t_h3, t_h10 = -1, -1, -1, -1
        else:
            n_data = self.n_test
            n_batch = n_data // batch_size + (n_data % batch_size > 0)
            ranking = []
            self.model.eval()
            iterator = tqdm(range(n_batch)) if verbose else range(n_batch)

            for i in iterator:
                start = i * batch_size
                end = min(n_data, (i + 1) * batch_size)
                batch_idx = np.arange(start, end)
                subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
                scores, hidden, nodes = self.model(subs, rels, mode='test')
                scores = scores.data.cpu().numpy()

                filters = []
                for i in range(len(subs)):
                    filt = self.loader.filters[(subs[i], rels[i])]
                    filt_1hot = np.zeros((self.n_ent,))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)

                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks

            ranking = np.array(ranking)
            t_mrr, t_mr, t_h1, t_h3, t_h10 = cal_performance(ranking)

        i_time = time.time() - i_time
        out_str = '[VALID] MRR:%.4f mr:%.4f H@1:%.4f H@3:%.4f H@10:%.4f\t [TEST] MRR:%.4f mr:%.4f H@1:%.4f H@3:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n' % (
        v_mrr, v_mr, v_h1, v_h3, v_h10, t_mrr, t_mr, t_h1, t_h3, t_h10, self.t_time, i_time)

        result_dict = {}
        result_dict['v_mrr'] = v_mrr
        result_dict['v_mr'] = v_mr
        result_dict['v_h1'] = v_h1
        result_dict['v_h3'] = v_h3
        result_dict['v_h10'] = v_h10
        result_dict['t_mrr'] = t_mrr
        result_dict['t_h1'] = t_h1
        result_dict['t_h3'] = t_h3
        result_dict['t_h10'] = t_h10

        return result_dict, out_str
