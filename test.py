import argparse, math, os, sys
import numpy as np
import robosuite as suite
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.REINFORCE import REINFORCE
from models.DDPG import DDPG

parser = argparse.ArgumentParser(description='PyTorch robot arm testing script')
parser = argparse.ArgumentParser(description='PyTorch robot arm training script')
parser.add_argument('--env_name', type=str, default='Lift')
parser.add_argument('--robot', type=str, default='Panda')
parser.add_argument('--algo', type=str, default='DDPG')
parser.add_argument('--model_path', type=str, default='Lift')
parser.add_argument('--render', type=bool, default=True)
parser.add_argument('--enable_her',type=bool,default=False)
parser.add_argument('--dense_rewards',type=bool,default=True)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--max_mem_size', type=int, default=50000)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_actor', type=float, default=0.0001)
parser.add_argument('--lr_critic', type=float, default=0.001)
parser.add_argument('--epsilon', type=float, default=10000)
parser.add_argument('--warmup', type=int, default=100)
parser.add_argument('--theta', type=int, default=0.15)
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--horizon', type=int, default=200,
                    help='max episode length (default: 200)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Setting up the robot enviornment
env = suite.make(
    env_name=args.env_name,
    robots=args.robot,
    has_renderer=args.render,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,                    
    horizon = args.horizon, 
    reward_shaping=False                
)
obs = env.reset()
state_dim = obs['robot0_robot-state'].shape[0]+obs['object-state'].shape[0]
state = np.append(obs['robot0_robot-state'],obs['object-state'])


# Setting algorithm according to args
if args.algo=='REINFORCE':
	agent = REINFORCE(state_dim,env.action_dim, args.gamma, args.lr, args.num_episodes, args.horizon, args.hidden_size)
	agent.model = torch.load(args.model_path) 
elif args.algo=='DDPG':
    agent = DDPG(state_dim, env.action_dim, env, args)
    agent.actor = torch.load(args.model_path) 
else:
	sys.exit('Incorrect algorithms specification. Please check the algorithm argument provided.')

# Visualize a single run
done=False
while done==False: 
    if args.algo=='REINFORCE':
        action, log_prob = agent.select_action(state)
    else:
        action = agent.select_action(state)           
    obs, reward, done, info = env.step(action)
    state = np.append(obs['robot0_robot-state'],obs['object-state'])
    env.render()

# Add code to print out model stats + plots