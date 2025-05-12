import numpy as np
from scipy.stats import rankdata
import os

def check_path(directory):
    os.makedirs(directory, exist_ok=True)

def cal_ranks(score_matrix, label_mask, filter_mask):
    norm_scores = score_matrix - np.min(score_matrix, axis=1, keepdims=True) + 1e-8
    full_ranks = rankdata(-norm_scores, method='average', axis=1)
    filtered_ranks = rankdata(-norm_scores * filter_mask, method='min', axis=1)
    raw_ranks = (full_ranks - filtered_ranks + 1) * label_mask
    return list(raw_ranks[raw_ranks != 0])

def cal_performance(rank_list):
    rank_array = np.asarray(rank_list)
    reciprocal_rank = 1. / rank_array
    mrr = reciprocal_rank.mean()
    mr = rank_array.mean()
    hits_at = lambda k: np.mean(rank_array <= k)
    return mrr, mr, hits_at(1), hits_at(3), hits_at(10)

def unique_without_sort(sequence):
    _, first_indices = np.unique(sequence, return_index=True)
    return list(np.array(sequence)[np.sort(first_indices)])
