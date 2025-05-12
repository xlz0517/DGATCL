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
        self.attn_fc = nn.Linear(2 * in_dim, 1)
        self.W_node = nn.Linear(in_dim, in_dim)     
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

    def forward(self, query_subs, query_rels, node_features, edge_index, node_index, old_node_map, batch_size):
        edge_sub = edge_index[:, 4]
        edge_rel = edge_index[:, 2]
        edge_obj = edge_index[:, 5]
    
        src_entity_feat = node_features[edge_sub]
        relation_feat = self.rela_embed(edge_rel)
        query_rel_expanded = self.rela_embed(query_rels)[edge_index[:, 0]]
    
        num_nodes = node_index.shape[0]
        msg_content = src_entity_feat - relation_feat

        attn_raw_edge = self.w_alpha(
            nn.ReLU()(self.Ws_attn(src_entity_feat) + self.Wr_attn(relation_feat) + self.Wqr_attn(query_rel_expanded))
        )
        attn_weights_edge = torch.sigmoid(attn_raw_edge)
        msg_weighted_edge = attn_weights_edge * msg_content
        msg_aggregated_edge = scatter(msg_weighted_edge, index=edge_obj, dim=0, dim_size=num_nodes, reduce='max')
        node_transformed_edge = self.act(self.W_h(msg_aggregated_edge)).clone()
        
        node_pair = torch.cat([node_transformed_edge[edge_sub], node_transformed_edge[edge_obj]], dim=-1)
        attn_score_node = self.leaky_relu(self.attn_fc(node_pair)).squeeze(-1)
        edge_weights_node = scatter_softmax(attn_score_node, edge_sub)
        edge_msgs_node = self.W_node(node_transformed_edge[edge_obj])
        node_msg_updated_node = scatter(edge_weights_node.unsqueeze(-1) * edge_msgs_node,
                                        edge_sub, dim=0, dim_size=num_nodes, reduce='sum')
        node_transformed_node = self.act(node_msg_updated_node)
        combined = torch.stack([node_transformed_edge, node_transformed_node], dim=0)
        updated_node_features = combined.sum(dim=0)
        if self.n_node_topk <= 0:
            return updated_node_features
        
        diff_flags = torch.ones(num_nodes)
        diff_flags[old_node_map] = 0
        diff_mask = diff_flags.bool()
        diff_node_index = node_index[diff_mask]
    
        sampled_logits = self.W_samp(updated_node_features[diff_mask]).squeeze(-1)
    
        all_node_scores = torch.ones((batch_size, self.n_ent)).cuda() * float('-inf')
        all_node_scores[diff_node_index[:, 0], diff_node_index[:, 1]] = sampled_logits
        all_node_scores = self.softmax(all_node_scores)
    
        topk_ids = torch.topk(all_node_scores, self.n_node_topk, dim=1).indices.reshape(-1)
        topk_batch_ids = torch.arange(batch_size).repeat(self.n_node_topk, 1).T.reshape(-1)
    
        topk_mask = torch.zeros((batch_size, self.n_ent)).cuda()
        topk_mask[topk_batch_ids, topk_ids] = 1

        diff_prob_selected = topk_mask[diff_node_index[:, 0], diff_node_index[:, 1]].bool()
        same_mask = ~diff_mask.cuda()
        same_mask[diff_mask] = diff_prob_selected
    
        topk_confidence = topk_mask[diff_node_index[:, 0], diff_node_index[:, 1]]
        soft_confidence = all_node_scores[diff_node_index[:, 0], diff_node_index[:, 1]]

        updated_node_features[diff_mask] *= (
            topk_confidence - soft_confidence.detach() + soft_confidence
        ).unsqueeze(-1)

        new_node_index = node_index[same_mask]
        final_node_features = updated_node_features[same_mask]

        return final_node_features, new_node_index, same_mask


    

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

    def forward(self, input_sub_ids, input_rel_ids, mode='train'):
        batch_size = len(input_sub_ids)
        query_sub_ids = torch.LongTensor(input_sub_ids).cuda()  # [B]
        query_rel_ids = torch.LongTensor(input_rel_ids).cuda()  # [B]
        device = query_sub_ids.device
    
        init_hidden_state = torch.zeros((1, batch_size, self.hidden_dim), device=device)
        batch_entity_pairs = torch.cat([
            torch.arange(batch_size).unsqueeze(1).cuda(),
            query_sub_ids.unsqueeze(1)
        ], dim=1)  # [B, 2] 形如 (batch_idx, entity_id)
    
        entity_features = torch.zeros(batch_size, self.hidden_dim, device=device)

        for layer_idx in range(self.n_layer):
            sampled_node_pairs, edge_index_matrix, index_old_to_new = self.loader.get_neighbors(
                batch_entity_pairs.data.cpu().numpy(),
                batch_size,
                mode=mode
            )
            total_sampled_nodes = sampled_node_pairs.size(0)
    
            entity_features, sampled_node_pairs, selected_node_indices = self.gnn_layers[layer_idx](
                query_sub_ids, query_rel_ids, entity_features,
                edge_index_matrix, sampled_node_pairs,
                index_old_to_new, batch_size
            )
    
            updated_hidden = torch.zeros(1, total_sampled_nodes, entity_features.size(1), device=device)
            updated_hidden.index_copy_(1, index_old_to_new, init_hidden_state)
    
            init_hidden_state = updated_hidden[0, selected_node_indices, :].unsqueeze(0)
            entity_features = self.dropout(entity_features)
    
            entity_features, init_hidden_state = self.gate(entity_features.unsqueeze(0), init_hidden_state)
            entity_features = entity_features.squeeze(0)
    
        final_scores = self.W_final(entity_features).squeeze(-1)
    
        all_entity_scores = torch.zeros((batch_size, self.loader.n_ent)).cuda()
        all_entity_scores[[sampled_node_pairs[:, 0], sampled_node_pairs[:, 1]]] = final_scores
    
        return all_entity_scores, entity_features, sampled_node_pairs


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
