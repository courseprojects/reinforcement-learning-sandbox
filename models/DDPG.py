# Implementation of DDPG algorithm

import robosuite as suite
import numpy as np
import math

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from DDPG_utils import ReplayBuffer, OUNoise

class DDPGActor(nn.Module):
	'''This class represents our actor model'''
	def __init__(self, state_dim, action_dim, hidden_size):
        super(DDPGActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
    	x = F.Relu(self.l1(x))
    	x = F.Relu(self.l2(x))
    	x = self.l3(x)

    	return x


class DDPGCritic(nn.Module):
	'''This class represents our critic model'''
	def __init__(self, state_dim, action_dim, hidden_size):
        super(DDPGCritic, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)


    def forward(self, x):
    	x = F.Relu(self.l1(x))
    	x = F.Relu(self.l2(x))
    	x = self.l3(x)

    	return x 


class DDPG:
	'''This class represents our implementation of DDPG'''
	def __init__(self, ):
        self.actor = DDPGActor()
        self.actor = self.actor.to(device)
        self.actor_target = DDPGActor()
        self.actor_target = self.actor_target.to(device)
        self.actor_optim  = optim.Adam(self.actor.parameters(), lr = self.lr)

        self.critic = DDPGCritic()
        self.critic = self.critic.to(device)
        self.critic_target = DDPGCritic()
        self.critic_target = self.critic_target.to(device)
        self.critic_optim  = optim.Adam(self.critic.parameters(), lr = self.lr)

        self.max_mem_size
        self.memory = ReplayBuffer(self.max_mem_size, self.state_dim, self.action_dim)

        self.tau = tau
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma 

    def random_process(self):
    	''''''


    def select_action(self, state):


    def update_parameters(self, ):

        
  