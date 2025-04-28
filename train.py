import random
import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base import Base
from utils import *

parser = argparse.ArgumentParser(description="Parser for DGATCL")
parser.add_argument('--data_path', type=str, default='data/fb15k-237/')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--topk', type=int, default=2000)
parser.add_argument('--layers', type=int, default=5)
parser.add_argument('--sampling', type=str, default='incremental')
parser.add_argument('--weight', type=str, default=None)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--loss_in_each_layer', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--HPO', action='store_true')
parser.add_argument('--eval_with_node_usage', action='store_true')
parser.add_argument('--scheduler', type=str, default='exp')
parser.add_argument('--remove_1hop_edges', action='store_true')
parser.add_argument('--fact_ratio', type=float, default=0.96)
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--eval_interval', type=int, default=1)
parser.add_argument("--neg_sample_ratio", type=int, default=2, help="description for experiment")
parser.add_argument('--k_h', type=int, default=10)
parser.add_argument('--k_w', type=int, default=20)
parser.add_argument('--ent_drop_pred', type=float, default=0.3)
parser.add_argument('--conv_drop', type=float, default=0.1)
parser.add_argument('--fc_drop', type=float, default=0.4)
parser.add_argument('--ker_sz', type=int, default=7)
parser.add_argument('--out_channel', type=int, default=250)


args = parser.parse_args()

if __name__ == '__main__':
    opts = args
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    
    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    
    torch.cuda.set_device(opts.gpu)
    print('==> gpu:', opts.gpu)
    loader = DataLoader(opts)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
        
    if dataset == 'umls':
        opts.lr = 0.0012
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.01
        opts.act = 'tanh'
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.n_batch = opts.n_tbatch = 10

    elif dataset == 'wn':
        opts.lr = 0.0030
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64 # 64
        opts.attn_dim = 5
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = opts.n_tbatch = 50 # 50

    elif dataset == 'WN18RR':
        opts.lr = 0.0030
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64 # 64
        opts.attn_dim = 5
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = opts.n_tbatch = 50 # 50
        
    elif dataset == 'fb15k-237':
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_node_topk = [opts.topk] * opts.layers
        opts.n_edge_topk = -1
        opts.n_layer = opts.layers
        opts.dropout = 0.0391
        opts.act = 'idd'
        opts.n_batch = opts.n_tbatch = 20 # 10
    
    # check all output paths
    checkPath('./results/')
    checkPath(f'./results/{dataset}/')
    checkPath(f'{loader.task_dir}/saveModel/')

    model = Base(opts, loader)
    opts.perf_file = f'results/{dataset}/{model.modelName}_perf.txt'
    print(f'==> perf_file: {opts.perf_file}')
    
    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)  

    if args.weight != None:
        model.loadModel(args.weight)
        model._update()
        model.model.updateTopkNums(opts.n_node_topk)

    if opts.train:
        # training mode
        best_v_mrr = 0
        for epoch in range(opts.epoch):
            model.train_batch()
            # eval on val/test set
            if (epoch+1) % args.eval_interval == 0:
                result_dict, out_str = model.evaluate(eval_val=True, eval_test=True)
                v_mrr, t_mrr = result_dict['v_mrr'], result_dict['t_mrr']
                out_str = '(epoch:' + str(epoch)+ ') ' + out_str
                print()
                with open(opts.perf_file, 'a+') as f:
                    f.write(out_str)
                if v_mrr > best_v_mrr:
                    best_v_mrr = v_mrr
                    best_str = out_str
                    print(best_str)
                    # BestMetricStr = f'ValMRR_{str(v_mrr)[:5]}_TestMRR_{str(t_mrr)[:5]}'
                    # model.saveModelToFiles(BestMetricStr, deleteLastFile=False)
        
        # show the final result
        print(best_str)

        # 将best_str写到文件的第一行
        with open(opts.perf_file, 'r') as f:
            temp_lines = f.readlines()  # 读取所有行到列表中
        # 在列表的开头插入best_str，并添加一个换行符（如果需要）
        temp_lines.insert(0, best_str + '\n')
        # 将修改后的内容写回文件，覆盖原文件
        with open(opts.perf_file, 'w') as f:
            f.writelines(temp_lines)

    if opts.eval:
        # evaluate on test set with loaded weight to save time
        result_dict, out_str = model.evaluate(eval_val=False, eval_test=True, verbose=True)
        print(result_dict, '\n', out_str)
        
