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
        self.actor_model = DDPGActor()
        self.actor_model = self.actor_model.to(device)
        self.critic_model = DDPGCritic()
        self.critic_model = self.critic_model.to(device)
        # Need to organize target models
        
        # Need to define replay buffer
        self.memory = 

        # Hyper parameters
        self.batch_size = 
        self.lr = 
        self.gamma = 

    def select_action(self, state):


    def update_parameters(self, ):

        
  