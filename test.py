import argparse, math, os, sys
import numpy as np
import robosuite as suite
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.REINFORCE import REINFORCE

parser = argparse.ArgumentParser(description='PyTorch robot arm testing script')
parser.add_argument('--env_name', type=str, default='Lift')
parser.add_argument('--robot', type=str, default='Panda')
parser.add_argument('--model_path', type=str, default='Lift')
parser.add_argument('--render', type=bool, default=True)
parser.add_argument('--save_results', type=str, default='Panda')
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
    reward_shaping=True                 
)
obs = env.reset()
state_dim = obs['robot0_robot-state'].shape[0]+obs['object-state'].shape[0]

# Setting algorithm according to args
if args.algo=='REINFORCE':
	agent = REINFORCE(state_dim,env.action_dim, args.gamma, args.lr, args.num_episodes, args.horizon, args.hidden_size)
	agent.model = torch.load(args.model_path) 
elif args.algo=='DDPG':
    agent = DDPG(state_dim, env.action_dim, args)
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
    env.render()