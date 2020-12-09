# This script needs to be updated to use click instead 
# of argparse.

import argparse, math, os, sys
from copy import deepcopy
import numpy as np
import robosuite as suite
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ray
import roboray.models as M
from roboray.train import train_reinforce
from roboray.train import train_ddpg


def main():
	parser = argparse.ArgumentParser(description='PyTorch robot arm training script')
	parser.add_argument('--env_name', type=str, default='Lift')
	parser.add_argument('--robot', type=str, default='Panda')
	parser.add_argument('--algo', type=str, default='DDPG')
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
	parser.add_argument('--cube_x_distro',type=float, nargs='+', default=[0, 0],
	                    help='x distribution for cube location')
	parser.add_argument('--cube_y_distro',type=float,  nargs='+', default=[0, 0],
	                    help='y distribution for cube location')
	parser.add_argument('--enable_arm_randomization', type=bool, default=False,
	                    help='enable arm randomization (uniform) for each of the 7 joints')
	parser.add_argument('--gamma', type=float, default=0.99,
	                    help='discount factor for reward (default: 0.99)')
	parser.add_argument('--num_epochs',type=int, default=500,
	                    help='number of epochs to train on' )
	parser.add_argument('--num_episodes', type=int, default=1,
	                    help='number of episodes per epoch')
	parser.add_argument('--horizon', type=int, default=20,
	                    help='max episode length (default: 200)')
	parser.add_argument('--wandb_api', type=str, default=None, 
	                    help='wandb api key')
	parser.add_argument('--wandb_project', type=str, default='cs221-project',
	                    help='wandb project name')
	parser.add_argument('--wandb_name', type=str, default='test-run',
	                    help='name of run')
	parser.add_argument('--wandb_entity', type=str, default='peterdavidfagan', metavar='N',
	                    help='name of user running experiment')

	args = parser.parse_args()
	os.environ['WANDB_API_KEY'] = args.wandb_api
	os.environ['WANDB_PROJECT'] = args.wandb_project
	os.environ['WANDB_NAME'] =  args.wandb_name
	os.environ['ENTITY'] = args.wandb_entity

	if args.wandb_api is not None:
	    wandb.init(config=vars(args))
	#ray.init(address="auto")


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Setting up the robot enviornment
	env = suite.make(
	    env_name=args.env_name,
	    robots=args.robot,
	    has_renderer=False,
	    has_offscreen_renderer=False,
	    use_camera_obs=False,
	    use_object_obs=True,                    
	    horizon = args.horizon, 
	    reward_shaping=args.dense_rewards,
	    )

	obs = env.reset()
	state_dim = obs['robot0_robot-state'].shape[0]+obs['object-state'].shape[0]

	# Setting algorithm according to args
	if args.algo == 'REINFORCE':
	    print('Starting to train REINFORCE...')
	    train_reinforce()
	elif args.algo == 'DDPG':
	    print('Starting to train DDPG')
	    train_ddpg()        
	else:
	    sys.exit('Incorrect algorithms specification. Please check the algorithm argument provided.')

