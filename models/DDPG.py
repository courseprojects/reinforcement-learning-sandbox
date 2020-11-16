# Implementation of DDPG algorithm with inspiration from
# "https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py"

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

from utils import *

class DDPGActor(nn.Module):
	'''This class represents our actor model'''
	def __init__(self, state_dim, action_dim, hidden_size):
        super(DDPGActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
    	x = F.Relu(self.l1(x))
    	x = F.Relu(self.l2(torch.cat(x,)))
    	x = self.l3(x)

    	return x


class DDPGCritic(nn.Module):
	'''This class represents our critic model'''
	def __init__(self, state_dim, action_dim, hidden_size):
        super(DDPGCritic, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)


    def forward(self, xs):
    	x,a = xs
    	x = F.Relu(self.l1(torch.cat([x,a],1)))
    	x = F.Relu(self.l2(x))
    	x = self.l3(x)

    	return x 


class DDPG:
	'''This class represents our implementation of DDPG'''
	def __init__(self, args, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = DDPGActor(state_dim, action_dim, args.hidden_size)
        self.actor = self.actor.to(device)
        self.actor_target = DDPGActor(state_dim, action_dim, args.hidden_size)
        self.actor_target = self.actor_target.to(device)
        self.actor_optim  = optim.Adam(self.actor.parameters(), lr = self.lr)

        self.critic = DDPGCritic(state_dim, action_dim, args.hidden_size)
        self.critic = self.critic.to(device)
        self.critic_target = DDPGCritic(state_dim, action_dim, args.hidden_size)
        self.critic_target = self.critic_target.to(device)
        self.critic_optim  = optim.Adam(self.critic.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)

        self.max_mem_size
        self.memory = ReplayBuffer(args.max_mem_size, state_dim, action_dim)

        self.tau = args.tau
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gamma = args.gamma 
        self.depsilon = 1.0 / args.epsilon

        self.s_t = None
        self.a_t = None
        self.is_training = True


    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1


    def select_action(self, state, decay_epsilon=True):
    	 action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def update_parameters(self):
    	# Sample batch from replay buffer
    	state_batch, action_batch, reward_batch, \
    	next_state_batch, done_batch = self.memory.sample(self.batch_size)

    	# Calculate next q-values
    	q_next = self.critic_target(next_state_batch, \
    				 self.actor_target(to_tensor(next_state_batch)))

		target_q_batch = to_tensor(reward_batch) + \
			self.discount*to_tensor(done_batch.astype(np.float))*q_next

    	# Critic update
    	self.critic.zero_grad()

    	q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])
    	value_loss = self.criterion(q_batch, target_q_batch)
    	value_loss.backward()
    	self.critic_optim.step()

    	# Actor update 
    	self.actor.zero_grad()

    	policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        
  