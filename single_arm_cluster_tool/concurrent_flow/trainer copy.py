import copy
import logging
import argparse
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
DEBUG_MODE = True
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
        'use_cuda': False,
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
        'eval_type': 'argmax',
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
    
    def _stack_states(self, states: list):
        """Stack multiple states into a single batch state."""
        return State(**{field: torch.stack([getattr(state, field) for state in states])
                        for field in State.__dataclass_fields__})
    
    def train(self):
        """Execute the training process."""
        print("Starting training...")
        
        for episode in range(self.trainer_params['num_episodes']):
            # Initialize batch
            envs = []
            states = []
            rewards = []
            log_probs = []
            done_flags = []
            
            # Reset environments
            for _ in range(self.trainer_params['batch_size']):
                env = Env(**self.env_params)
                state = env.reset()
                envs.append(env)
                states.append(state)
            
            # Convert to batch state
            batch_state = self._stack_states(states)
            batch_state.batch_idx = torch.arange(batch_state.batch_size())
            batch_state.to(self.device)
            
            # Initialize model encoding
            self.model.encoding(batch_state)
            
            # Episode loop
            while not batch_state.done.all():
                # Get action and log probability
                action, prob = self.model(batch_state)
                
                # Step environments
                next_states = []
                step_rewards = []
                step_log_probs = []
                step_dones = []
                
                for b, (env, a) in enumerate(zip(envs, action)):
                    if not env.done:
                        next_state = env.step(a.item())
                        #reward = -env.clock  # Negative makespan as reward
                        done = env.done
                        
                        next_states.append(next_state)
                        step_rewards.append(reward)
                        step_log_probs.append(torch.log(prob[b]))
                        step_dones.append(done)
                    else:
                        next_states.append(states[b])  # Keep old state for done environments
                        step_rewards.append(0)
                        step_log_probs.append(torch.tensor(0.0))
                        step_dones.append(True)
                
                # Update states
                states = next_states
                batch_state = self._stack_states(states)
                batch_state.batch_idx = torch.arange(batch_state.batch_size())
                batch_state.to(self.device)
                
                # Store rewards and log probabilities
                rewards.extend(step_rewards)
                log_probs.extend(step_log_probs)
                done_flags.extend(step_dones)
            
            # Calculate returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.trainer_params['gamma'] * G
                returns.insert(0, G)
            
            # # Normalize returns
            # returns = torch.tensor(returns, device=self.device)
            # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            returns = torch.tensor(returns, device=self.device)
            
            # Calculate loss
            loss = 0
            for log_prob, R in zip(log_probs, returns):
                loss -= log_prob * R
            loss = loss.mean()
            
            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate episode makespan
            episode_makespan = sum(env.clock for env in envs) / len(envs)
            self.episode_makespans.append(episode_makespan)
            
            # Logging
            if episode % self.trainer_params['log_interval'] == 0:
                avg_makespan = sum(self.episode_makespans[-self.trainer_params['log_interval']:]) / self.trainer_params['log_interval']
                print(f"Episode {episode}/{self.trainer_params['num_episodes']}, Avg Makespan: {avg_makespan:.2f}")
            
            # Save model
            if episode % self.trainer_params['save_interval'] == 0:
                self.save_model(episode)
        
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
