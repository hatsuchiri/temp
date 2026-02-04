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
import time
import shutil
import sys

from torch.optim import optimizer

from envs.sdcfEnv import sdcfEnv as Env, State
from model.model_concat import CONCATNet as CONCATModel
from envs.algorithms.cbs import ConcurrentBackwardSequence

# Global configurations
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 1
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
    parser.add_argument('--foup_size', type=int, default=50, help='Size of the foup')
    parser.add_argument('--group1_stage', type=int, default=1, help='Stages for type 1')
    parser.add_argument('--group1_min_prs_time', type=int, default=10, help='Minimum processing time for type 1')
    parser.add_argument('--group1_max_prs_time', type=int, default=300, help='Maximum processing time for type 1')
    parser.add_argument('--group2_stage', type=int, default=1, help='Stages for type 2')
    parser.add_argument('--group2_min_prs_time', type=int, default=10, help='Minimum processing time for type 2')
    parser.add_argument('--group2_max_prs_time', type=int, default=300, help='Maximum processing time for type 2')
    parser.add_argument('--prod_quantity', type=int, default=10, help='Production quantity (Unit: FOUP)')
    parser.add_argument('--done_quantity', type=int, default=20, help='Done Production quantity (Unit: Wafer)')
    parser.add_argument('--num_lot_type', type=int, default=2, help='Total number of lot types')
    
    parser.add_argument('--model_type', type=str, default='concat', help='Model type = {is, rms, concat}')
    parser.add_argument('--input_action', type=str, default='wafer', help='loadlock input action type = {wafer, type}')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='Discount factor')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving model')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging')
    parser.add_argument('--baselineN', type=int, default=4, help='Sample number for each baseline')

    return parser.parse_args()

def get_stage_list():
    """Define stage configurations."""
    return [
        [1,1,1],
        [1,1,1],
    ]

def setup_trainer_params(args):
    """Initialize trainer parameters."""
    file_name = os.path.basename(__file__).split('.')[0]
    t = time.localtime()
    return {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'model_save': {
            'enable': True,
            'path': f'./saved_models/result_{file_name}_{time.strftime("%Y%m%d_%H_%M", t)}',
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


class Trainer:
    def __init__(self, env_params, model_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.trainer_params = trainer_params
        self.device = torch.device('cuda' if trainer_params['use_cuda'] else 'cpu')
        self.model_params['device'] = self.device
        
        # Initialize model and optimizer
        self.model = CONCATModel(**self.env_params, **self.model_params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=trainer_params['learning_rate'])
        
        # Create directory for saving models
        if trainer_params['model_save']['enable']:
            os.makedirs(trainer_params['model_save']['path'], exist_ok=True)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_makespans = []

        self.baselineN = trainer_params['baselineN']
        self.done_quantity = env_params['done_quantity']
    

    
    def train(self):
        """Execute the training process."""
        print("Starting training...")

        def _stack_states(states: list):
            """Stack multiple states into a single batch state."""
            return State(**{field: torch.stack([getattr(state, field) for state in states])
                            for field in State.__dataclass_fields__})

        envs = []
        states = []
           
        for b in range(self.trainer_params['batch_size']):
            env = Env(**self.env_params)
            state = env.reset()
            envs.append(env)
            states.append(state)

        for episode in range(self.trainer_params['num_episodes']):
            # Initialize batch
            for b in range(self.trainer_params['batch_size']):
                state = envs[b].reset()
                states[b] = copy.deepcopy(state)            
            # Convert to batch state
            batch_state = _stack_states(states)
            batch_state.batch_idx = torch.arange(batch_state.batch_size())
            batch_state.to(self.device)
            batch_state_trajectory = []
            batch_action_trajectory = []
            batch_prob_trajectory = []

            
            # Initialize model encoding
            self.model.train()
            self.model.to(self.device)
            self.model.encoding(batch_state)
            makespans = [1e10 for _ in range(batch_state.batch_size())]
            batchGt = [[] for _ in range(batch_state.batch_size())]
            Tlength_of_each_Instance = [0 for _ in range(batch_state.batch_size())]
            done_until_of_each_Instance = [[] for _ in range(batch_state.batch_size())]
            T = 1
            while not batch_state.done.all():                
                # Get action and log probability
                # 终于发现违和的地方在哪了，model可以直接batch处理，env不行要分开处理
                ## batch处理          
                next_states = []
                # Step environments
                batch_action, batch_prob = self.model(batch_state)
                ##batch_action = batch_action.to(self.device) 
                ##state的to方法没有返回值，是sdcEnv里自定义的，这么写返回None
                batch_action.to(self.device) 
                for b,a in enumerate(batch_action):
                    next_state = envs[b].step(a.item())
                    next_states.append(next_state)
                    if envs[b].done and makespans[b] == 1e10:
                        makespans[b] = copy.deepcopy(envs[b].clock)
                        done_until_of_each_Instance[b].append(True)
                        Tlength_of_each_Instance[b] = T
                    else:
                        done_until_of_each_Instance[b].append(False)
                #回去试试这句_stack_states try except
                try:
                    batch_next_state = _stack_states(next_states)
                except:
                    print('next_states==',next_states)
                batch_next_state.batch_idx = torch.arange(batch_next_state.batch_size())
                batch_next_state.to(self.device)  

                batch_state_trajectory.append(batch_state)
                batch_action_trajectory.append(batch_action)
                batch_prob_trajectory.append(batch_prob)

                batch_state = copy.deepcopy(batch_next_state)

                T += 1
            print('makespans==',makespans) 
            print('avg makespan==',sum(makespans)/batch_state.batch_size())               
            ## 本来想用T来计算每个Instance的长度，以免对T时刻已经done的Instance做梯度下降
            ## 但是后来发现好像没必要，已经done的action prob等于1，log1 == 0
            ## 想想bI这么写会不会被算在反向传播里，需不需要requires_grad = False
            
                      
                
            
            print('start to calculate loss')
            loss_trajectory = []
            for makespan,prob in zip(makespans,batch_prob_trajectory):
                loss = -(-makespan) * torch.log(prob)/self.done_quantity
                loss_trajectory.append(loss)   


            sum_loss = sum(loss_trajectory)
            sum_loss = sum_loss.to(self.device)
            self.optimizer.zero_grad()
            sum_loss.backward(torch.ones_like(sum_loss))
            self.optimizer.step()
            print(f"Episode {episode}, Loss: {sum_loss}")   


        # Save final model
        self.save_model(self.trainer_params['num_episodes'])
        print("Training completed!")
    
    def save_model(self, episode):
        """Save the model checkpoint."""
        if self.trainer_params['model_save']['enable']:
            checkpoint_path = f"{self.trainer_params['model_save']['path']}/checkpoint_ep{episode}.pt"
            torch.save({
                'episode': episode,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'episode_makespans': self.episode_makespans,
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

            current_file = os.path.abspath(__file__)
            file_copy_path = f"{self.trainer_params['model_save']['path']}"
            shutil.copy(current_file, file_copy_path)


def main():
    set_seed()
    args = parse_arguments()
    stage_list = get_stage_list()
    
    env_params = setup_env_params(args, stage_list)
    model_params = setup_model_params(args, env_params)
    trainer_params = setup_trainer_params(args)
    
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      trainer_params=trainer_params)
    
    trainer.train()


if __name__ == "__main__":
    main()
