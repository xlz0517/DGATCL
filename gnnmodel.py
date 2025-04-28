import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch_scatter import scatter, scatter_softmax
from collections import defaultdict


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, n_ent, n_node_topk=-1, n_edge_topk=-1, tau=1.0,
                 act=lambda x: x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.n_ent = n_ent
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.n_node_topk = n_node_topk
        self.n_edge_topk = n_edge_topk
        self.tau = tau
        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.W_samp = nn.Linear(in_dim, 1, bias=False)
        self.W_node_attn = nn.Linear(in_dim, 1, bias=False)
        self.attn_fc = nn.Linear(2 * dim, 1)  
        self.W_node = nn.Linear(dim, dim)     
        self.leaky_relu = nn.LeakyReLU(0.2)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if self.training and self.tau > 0:
            self.softmax = lambda x: F.gumbel_softmax(x, tau=self.tau, hard=False)
        else:
            self.softmax = lambda x: F.softmax(x, dim=1)
        for module in self.children():
            module.train(mode)
        return self

    def forward(self, q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, batchsize):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        hs = hidden[sub]
        hr = self.rela_embed(rel)
        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]
        n_node = nodes.shape[0]
        message = hs - hr
        
        alpha = torch.sigmoid(self.w_alpha(
                nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))  # [N_edge_of_all_batch, 1]
        # aggregate message and then propagate
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='max')#
        hidden_new = self.act(self.W_h(message_agg))  # [n_node, dim]
        hidden_new = hidden_new.clone()
        
        node_attention_scores = self.leaky_relu(self.attn_fc(torch.cat([hidden_new[sub], hidden_new[obj]], dim=-1))).squeeze(-1)
        node_attention_weights = scatter_softmax(node_attention_scores, sub)
        node_attention_weights = self.softmax(node_attention_scores)
        hidden_new = scatter(
            node_attention_weights.unsqueeze(-1) * self.W_node(hidden_new[obj]),
            sub,
            dim=0,
            dim_size=n_node,
            reduce='sum'
        )
        hidden_new = self.act(hidden_new)
        
        if self.n_node_topk <= 0:
            return hidden_new
        # print('-------------')
        tmp_diff_node_idx = torch.ones(n_node)
        tmp_diff_node_idx[old_nodes_new_idx] = 0
        bool_diff_node_idx = tmp_diff_node_idx.bool()
        diff_node = nodes[bool_diff_node_idx]
        # project logit to fixed-size tensor via indexing
        diff_node_logit = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1)  # [all_batch_new_nodes]

        
        node_scores = torch.ones((batchsize, self.n_ent)).cuda() * float('-inf')
        node_scores[diff_node[:, 0], diff_node[:, 1]] = diff_node_logit

        # 采样
        node_scores = self.softmax(node_scores)  # [batchsize, n_ent]
        # print(node_scores)
        topk_index = torch.topk(node_scores, self.n_node_topk, dim=1).indices.reshape(-1)
        # print(topk_index)
        #print(1 / 0)
        topk_batchidx = torch.arange(batchsize).repeat(self.n_node_topk, 1).T.reshape(-1)
        batch_topk_nodes = torch.zeros((batchsize, self.n_ent)).cuda()
        batch_topk_nodes[topk_batchidx, topk_index] = 1

        
        bool_sampled_diff_nodes_idx = batch_topk_nodes[diff_node[:, 0], diff_node[:, 1]].bool()
        bool_same_node_idx = ~bool_diff_node_idx.cuda()
        bool_same_node_idx[bool_diff_node_idx] = bool_sampled_diff_nodes_idx

        
        diff_node_prob_hard = batch_topk_nodes[diff_node[:, 0], diff_node[:, 1]]
        diff_node_prob = node_scores[diff_node[:, 0], diff_node[:, 1]]
        hidden_new[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)

        
        new_nodes = nodes[bool_same_node_idx]
        hidden_new = hidden_new[bool_same_node_idx]

        return hidden_new, new_nodes, bool_same_node_idx


class GNNModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_ent = params.n_ent
        self.n_rel = params.n_rel
        self.n_node_topk = params.n_node_topk
        self.n_edge_topk = params.n_edge_topk
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            i_n_node_topk = self.n_node_topk if 'int' in str(type(self.n_node_topk)) else self.n_node_topk[i]
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, self.n_ent, \
                                            n_node_topk=i_n_node_topk, n_edge_topk=self.n_edge_topk, tau=params.tau,
                                            act=act))

        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def updateTopkNums(self, topk_list):
        assert len(topk_list) == self.n_layer
        for idx in range(self.n_layer):
            self.gnn_layers[idx].n_node_topk = topk_list[idx]

    def fixSamplingWeight(self):
        def freeze(m):
            m.requires_grad = False

        for i in range(self.n_layer):
            self.gnn_layers[i].W_samp.apply(freeze)

    def forward(self, subs, rels, mode='train'):
        n = len(subs)  # n == B (Batchsize)
        q_sub = torch.LongTensor(subs).cuda()  # [B]
        q_rel = torch.LongTensor(rels).cuda()  # [B]
        # print(rels)
        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()  # [1, B, dim]
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)],
                          1)  # [B, 2] with (batch_idx, node_idx)
        hidden = torch.zeros(n, self.hidden_dim).cuda()  # [B, dim]

        # 多层GNN之间的消息传递
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), n, mode=mode)
            n_node = nodes.size(0)

            hidden, nodes, sampled_nodes_idx = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx,
                                                                  n)

            h0 = torch.zeros(1, n_node, hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            h0 = h0[0, sampled_nodes_idx, :].unsqueeze(0)
            hidden = self.dropout(hidden)
            # GRU聚合GNN层与层之间的实体嵌入表示
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)


        scores = self.W_final(hidden).squeeze(-1)
        # print(scores)

        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()

        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores

        return scores_all, hidden, nodes

class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, neg_features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size).float().to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        B = features.shape[0]
        K = neg_features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # concat all contrast features at dim 0
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 正样本相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        # 负样本
        h_neg = neg_features[:, 0, :]
        t_neg = neg_features[:, 1, :]
        h_neg = h_neg.view(B, K, -1)   # [B, K, dim]
        t_neg = t_neg.view(B, K, -1)   # [B, K, dim]

        anchor = features[:, 0, :]  # [B, dim]
        anchor_expand = anchor.unsqueeze(1).expand(-1, K, -1)  # [B, K, dim]

        sim_neg = F.cosine_similarity(anchor_expand, t_neg, dim=-1)  # [B, K]
        sim_neg = sim_neg / self.temperature

        # 整合正负样本
        sim_pos = F.cosine_similarity(anchor, features[:, 1, :], dim=-1).unsqueeze(1)
        sim_pos = sim_pos / self.temperature
        logits_all = torch.cat([sim_pos, sim_neg], dim=1)
        log_prob_all = logits_all - torch.logsumexp(logits_all, dim=1, keepdim=True)
        pos_mask = torch.zeros_like(log_prob_all).to(device)
        pos_mask[:, 0] = 1

        mean_log_prob_pos = (pos_mask * log_prob_all).sum(1)  # [B]
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
