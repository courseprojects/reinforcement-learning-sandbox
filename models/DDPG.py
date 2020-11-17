# Implementation of DDPG algorithm with inspiration from "https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py"

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

from models.utils import * # Improve by adding path var


class DDPGActor(nn.Module):
    '''This class represents our actor model'''

    def __init__(self, state_dim, action_dim, hidden_size):
        super(DDPGActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
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
        x, a = xs
        x = F.relu(self.l1(torch.cat([x, a], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


class DDPG:
    '''This class represents our implementation of DDPG'''

    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = DDPGActor(state_dim, action_dim, args.hidden_size)
        self.actor = self.actor.to(device)
        self.actor_target = DDPGActor(state_dim, action_dim, args.hidden_size)
        self.actor_target = self.actor_target.to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.lr)

        self.critic = DDPGCritic(state_dim, action_dim, args.hidden_size)
        self.critic = self.critic.to(device)
        self.critic_target = DDPGCritic(
            state_dim, action_dim, args.hidden_size)
        self.critic_target = self.critic_target.to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.max_mem_size = args.max_mem_size
        self.memory = ReplayBuffer(args.max_mem_size, state_dim, action_dim)

        self.random_process = OrnsteinUhlenbeckProcess(args.theta)

        self.tau = args.tau
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = 1.0
        self.depsilon = 1.0 / args.epsilon

        self.s_t = None
        self.a_t = None
        self.is_training = True


    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.store_transition(self.s_t, self.a_t, r_t, s_t1, done)
            self.s_t = s_t1


    def select_action(self, state, decay_epsilon=True):
        action = self.actor(to_tensor(state).to(device)).detach().to('cpu').numpy()
        action += self.is_training*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.action_dim)
        self.a_t = action
        return action

    def update_parameters(self):
        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, \
        next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        state_batch = to_tensor(state_batch).to(device)
        action_batch = to_tensor(action_batch).to(device)
        reward_batch = to_tensor(reward_batch).to(device)
        next_state_batch = to_tensor(next_state_batch).to(device)
        done_batch = to_tensor(done_batch).to(device)

        # Calculate next q-values
        with torch.no_grad():
            q_next = self.critic_target([next_state_batch, \
                         self.actor_target(next_state_batch)])

            target_q_batch = reward_batch + \
                self.gamma*q_next

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([state_batch, action_batch])
        value_loss = self.criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update 
        self.actor.zero_grad()

        policy_loss = -self.critic([
            state_batch,
            self.actor(state_batch)
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        
  
