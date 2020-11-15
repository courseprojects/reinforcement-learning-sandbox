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


class OUActionNoise(object):
	'''ornstein uhlenbeck process, source: "https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/DDPG/pytorch/lunar-lander/ddpg_torch.py" '''	
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)


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

        
        # Need to define replay buffer
        self.memory = 

        # Hyper parameters
        self.tau = 
        self.batch_size = 
        self.lr = 
        self.gamma = 

    def random_process(self):
    	''''''


    def select_action(self, state):


    def update_parameters(self, ):

        
  