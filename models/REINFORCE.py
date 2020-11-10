# Implementation of REINFORCE algorithm for multiple simulataneous continuous actions. 
# This implementation is inspired by https://github.com/chingyaoc/pytorch-REINFORCE
# For more on REINFORCE please consult Reinforcement Learning Richard S. Sutton &
# Andrew G.Barto chapter 13

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

pi = Variable(torch.FloatTensor([math.pi])).to(device)

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi).sqrt()
    return a*b
    

class REINFORCEPolicy(nn.Module):
    '''
    This class represent our policy parameterization.
    '''
    def __init__(self, state_dim,action_dim,hidden_layer):
        super(REINFORCEPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layer = hidden_layer
        
        self.l1 = nn.Linear(self.state_dim, self.hidden_layer)
        self.l2 = nn.Linear(self.hidden_layer, self.hidden_layer//2)
        self.l3 = nn.Linear(self.hidden_layer//2, self.action_dim)
        self.l3_ = nn.Linear(self.hidden_layer//2, self.action_dim)
        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.5)

    def forward(self,x):
        out = F.relu(self.d1(self.l1(x)))
        out = F.relu(self.d2(self.l2(out)))
        mu = self.l3(out)
        sigma_sq = self.l3_(out)
        return mu, sigma_sq
    
    
class REINFORCE:
    '''
    This class encapsulates functionality required to run the REINFORCE algorithm.
    '''
    def __init__(self, state_dim,action_dim, gamma, lr, episodes, horizon, hidden_layer):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layer = hidden_layer
        self.lr = lr
        self.model = REINFORCEPolicy(state_dim, action_dim,hidden_layer)
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.model.train()
        
        self.gamma = gamma
        self.episodes = episodes
        self.horizon = horizon
        
        
    def select_action(self, state):
        actions = []
        log_probs = []
        mu , sigma_sq = self.model(Variable(state).to(device)) 
        for i in range(self.action_dim):
            mu_ = mu[i]
            sigma_sq_ = sigma_sq[i]
            sigma_sq_ = F.softplus(sigma_sq_) # ensures that the estimate is always positive

            eps = torch.randn(mu_.size())
            action = (mu_ + sigma_sq_.sqrt()*Variable(eps).to(device)).data
            prob = normal(action, mu_, sigma_sq_)
            log_prob = prob.log()
            actions.append(action)
            log_probs.append(log_prob)
        
        return actions, log_probs
    

    def episode_update_parameters(self, rewards, log_probs):
        R = torch.zeros(1, 1)
        loss = torch.zeros(self.action_dim)
        for i in reversed(range(self.horizon)):
            R = self.gamma * R + rewards[0][i]
            for j in range(self.action_dim):
                loss[j] = loss[j] - (log_probs[0][i][j]*(Variable(R.data.squeeze()).expand_as(log_probs[0][i][j])).to(device)).sum()
        loss = loss.sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
    def epoch_update_parameters(self, rewards, log_probs):
        R = torch.zeros(self.episodes)
        loss = torch.zeros(self.episodes,self.action_dim)
        for episode in range(self.episodes):
            for i in reversed(range(self.horizon)):
                R[episode] = self.gamma * R[episode] + rewards[episode][i]
                for j in range(self.action_dim):
                    loss[episode][j] = loss[episode][j] - (log_probs[episode][i][j]*(Variable(R[episode].data.squeeze()).expand_as(log_probs[episode][i][j])).to(device)).sum()
        
        loss = loss.sum(dim=0)/self.episodes
        loss = loss.sum()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()