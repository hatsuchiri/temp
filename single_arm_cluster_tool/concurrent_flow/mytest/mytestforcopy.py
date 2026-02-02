import copy
import logging
import argparse
from turtle import done
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
import os

from envs.sdcfEnv import sdcfEnv as Env, State
from model.model_concat import CONCATNet as CONCATModel
from envs.algorithms.cbs import ConcurrentBackwardSequence

# Global configurations
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
SEED = 1000

def set_seed(seed=SEED):
    """Fix random seed for reproducibility."""
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a model with a specific number of lot types')
    parser.add_argument('--foup_size', type=int, default=100, help='Size of the foup')
    parser.add_argument('--group1_stage', type=int, default=1, help='Stages for type 1')
    parser.add_argument('--group1_min_prs_time', type=int, default=10, help='Minimum processing time for type 1')
    parser.add_argument('--group1_max_prs_time', type=int, default=300, help='Maximum processing time for type 1')
    parser.add_argument('--group2_stage', type=int, default=1, help='Stages for type 2')
    parser.add_argument('--group2_min_prs_time', type=int, default=10, help='Minimum processing time for type 2')
    parser.add_argument('--group2_max_prs_time', type=int, default=300, help='Maximum processing time for type 2')
    parser.add_argument('--prod_quantity', type=int, default=10, help='Production quantity (Unit: FOUP)')
    parser.add_argument('--done_quantity', type=int, default=100, help='Done Production quantity (Unit: Wafer)')
    parser.add_argument('--num_lot_type', type=int, default=2, help='Total number of lot types')
    
    parser.add_argument('--model_type', type=str, default='concat', help='Model type = {is, rms, concat}')
    parser.add_argument('--input_action', type=str, default='wafer', help='loadlock input action type = {wafer, type}')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='Discount factor')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving model')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging')
    parser.add_argument('--baselineN', type=int, default=50, help='Sample number for each baseline')

    return parser.parse_args()

def get_stage_list():
    """Define stage configurations."""
    return [
        [1,1],
        [1,1,1],
    ]

def setup_trainer_params(args):
    """Initialize trainer parameters."""
    return {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'model_save': {
            'enable': True,
            'path': f'./saved_models/',
        },
        'batch_size': args.batch_size,
        'num_episodes': args.num_episodes,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'save_interval': args.save_interval,
        'log_interval': args.log_interval,
        'baselineN': args.baselineN,
    }
    
def setup_env_params(args, stage_list):
    """Initialize environment parameters."""
    return {
        'foup_size': args.foup_size,
        'group1_stage': stage_list[args.group1_stage],
        'group1_min_prs_time': args.group1_min_prs_time,
        'group1_max_prs_time': args.group1_max_prs_time,
        'group2_stage': stage_list[args.group2_stage],
        'group2_min_prs_time': args.group2_min_prs_time,
        'group2_max_prs_time': args.group2_max_prs_time,
        'prod_quantity': args.prod_quantity,
        'done_quantity': args.done_quantity,
        'num_lot_type': args.num_lot_type,
    }
    
def setup_model_params(args, env_params):
    """Initialize model parameters."""
    return {
        'type': args.model_type,
        'input_action': args.input_action,
        'purge': False,
        'embedding_dim': 256,
        'sqrt_embedding_dim': 256**(1/2),
        'encoder_layer_num': 3,
        'qkv_dim': 16,
        'sqrt_qkv_dim': 16**(1/2),
        'head_num': 16,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'ms_hidden_dim': 16,
        'ms_layer1_init': (1/2)**(1/2),
        'ms_layer2_init': (1/16)**(1/2),
        'eval_type': 'softmax',
        'normalize': 'instance'
    }

def main():
    set_seed()
    args = parse_arguments()
    stage_list = get_stage_list()
    
    env_params = setup_env_params(args, stage_list)
    model_params = setup_model_params(args, env_params)
    trainer_params = setup_trainer_params(args)

    env1 = Env(**env_params)
    env2 = copy.deepcopy(env1)
    ## 如果只是env2 = env1，那么env2的变化会影响env1，实际上只是起了个别名
    print(env1)
    print(env2)

if __name__ == "__main__":
    main()



