import torch
import numpy as np
import random
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"


def parse_args():
    parser = argparse.ArgumentParser(description="Graphormer")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    # gcn
    parser.add_argument('--use_gcn', action='store_true')
    parser.add_argument('--gcn_layers', type=int, default=1)
    parser.add_argument('--gcn_left', type=float, default=1.0)
    parser.add_argument('--gcn_right', type=float, default=0.0)
    parser.add_argument('--gcn_mean', action='store_true')
    # rankformer
    parser.add_argument('--use_rankformer', action='store_true')
    parser.add_argument('--rankformer_layers', type=int, default=1)
    parser.add_argument('--rankformer_tau', type=float, default=0.5)
    parser.add_argument('--rankformer_alpha', type=float, default=2)
    parser.add_argument('--rankformer_clamp_value', type=float, default=0)
    # cl-Loss
    parser.add_argument('--use_cl', action='store_true')
    parser.add_argument('--cl_layer', type=int, default=1)
    parser.add_argument('--cl_lambda', type=float, default=0.05)
    parser.add_argument('--cl_eps', type=float, default=0.1)
    parser.add_argument('--cl_tau', type=float, default=0.15)
    # Train
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--reg_lambda', type=float, default=1e-4)
    parser.add_argument('--loss_batch_size', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--show_loss_interval', type=int, default=1)
    # Test
    parser.add_argument('--topks', type=str, default='[20]')
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--valid_interval', type=int, default=20)
    parser.add_argument('--stopping_step', type=int, default=10)
    # Data
    parser.add_argument('--data', type=str, default="Ali-Display")
    # Experiment Setting
    parser.add_argument('--del_neg', action='store_true')
    parser.add_argument('--del_benchmark', action='store_true')
    parser.add_argument('--del_omega_norm', action='store_true')
    parser.add_argument('--save_emb', action='store_true')
    parser.add_argument('--load_emb', action='store_true')
    return parser.parse_args()


args = parse_args()
args.topks = eval(args.topks)
args.device = torch.device(f'cuda:{args.device:d}')

if args.seed != -1:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f'seed: {args.seed:d}')

print('Using', args.device)

print('Model Setting')
print(f'    hidden dim: {args.hidden_dim:d}')
if args.use_gcn:
    print(f'    Using {args.gcn_layers:d} layers GCN.')
    print(f'      gcn left = {args.gcn_left:f}')
    print(f'      gcn right = {args.gcn_right:f}')
    if args.gcn_mean:
        print(f'      Z = Mean( Z(0~{args.gcn_layers:d}) )')
    else:
        print(f'      Z = Z({args.gcn_layers:d})')
if args.use_rankformer:
    print(f'    Using {args.rankformer_layers:d} layers Rankformer:')
    print(f'      rankformer alpha = {args.rankformer_alpha:f}')
    print(f'      rankformer tau = {args.rankformer_tau:f}')
    print(f'      rankformer clamp value = {args.rankformer_clamp_value:f}')
if args.use_cl:
    print(f'    Using CL Loss:')
    print(f'      cl layer: {args.cl_layer:d}')
    print(f'      cl lambda: {args.cl_lambda:f}')
    print(f'      cl eps: {args.cl_eps:f}')
    print(f'      cl tau: {args.cl_tau:f}')

print('Train Setting')
print(f'    learning rate: {args.learning_rate:f}')
print(f'    reg_lambda: {args.reg_lambda:f}')
print(f'    loss batch size: {args.loss_batch_size:d}')
print(f'    max epochs: {args.max_epochs:d}')

print('Test Setting')
print(f'    topks: ', args.topks)
print(f'    test batch size: {args.test_batch_size:d}')
print(f'    valid interval: {args.valid_interval:d}')
print(f'    stopping step: {args.stopping_step:d}')

print('Data Setting')
args.data_dir = "data/"
args.train_file = os.path.join(args.data_dir, args.data, f'train.txt')
args.valid_file = os.path.join(args.data_dir, args.data, f'valid.txt')
args.test_file = os.path.join(args.data_dir, args.data, f'test.txt')
print(f'    train: {args.train_file:s}')
print(f'    valid: {args.valid_file:s}')
print(f'    test: {args.test_file:s}')

print('Experiment Setting')
print(f'    |                   Ablation Study Setting                 |')
print(f'    | Negative pairs | Benchmark | Offset | Normalize of Omega |')
print(f'    |        {"N" if args.del_neg else "Y"}       |     {"N" if args.del_benchmark else "Y"}     |   {"N" if args.rankformer_alpha==0 else "Y"}    |          {"N" if args.del_omega_norm else "Y"}         |')
if args.rankformer_alpha < 2:
    print('    Setting args.rankformer_alpha < 2 may violate some assumptions of the code. If the experimental results do not converge, please try increasing args.rankformer_clamp_value.')
if args.save_emb or args.load_emb:
    args.user_emb_path = 'saved/{args.data:s}_user.pt'
    args.item_emb_path = 'saved/{args.data:s}_item.pt'
    if args.save_emb:
        print(f'    Initial features of users and items will be loaded from {args.user_emb_path} and {args.item_emb_path}.')
    if args.load_emb:
        print(f'    The trained user and item features will be saved in {args.user_emb_path} and {args.item_emb_path}.')

print('---------------------------')
