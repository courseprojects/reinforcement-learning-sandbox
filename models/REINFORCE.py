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

pi = Variable(torch.FloatTensor([math.pi])) #.cuda()

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi).sqrt()
    return a*b
    

class REINFORCEPolicy(nn.Module):
    '''
    This class represent our policy parameterization.
    '''
    def __init__(self, state_dim,action_dim):
        super(REINFORCEPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.l1 = nn.Linear(self.state_dim, 128, bias = False)
        self.l2 = nn.Linear(128, self.action_dim*2, bias = False)

    def forward(self,x):
        model = nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
         )
        return model(x)
    
    
class REINFORCE:
    '''
    This class encapsulates functionality required to run the REINFORCE algorithm.
    '''
    def __init__(self, state_dim,action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = REINFORCEPolicy(state_dim, action_dim)
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)
        self.model.train()
        
    def select_action(self, state):
        actions = []
        log_probs = []
        outputs = self.model(Variable(state).to(device)) 
        for i in range(self.action_dim):
            mu = outputs[i]
            sigma_sq = outputs[i+1]
            sigma_sq = F.softplus(sigma_sq) # ensures that the estimate is always positive

            eps = torch.randn(mu.size())
            action = (mu + sigma_sq.sqrt()*Variable(eps).to(device)).data
            prob = normal(action, mu, sigma_sq)
            log_prob = prob.log()
            actions.append(action)
            log_probs.append(log_prob)
        
        return actions, log_probs
        
        
    def update_parameters(self, rewards, log_probs, gamma):
        R = torch.zeros(1, 1)
        loss = torch.zeros(self.action_dim)
        for i in reversed(range(len(rewards))):
            for j in range(self.action_dim):
                R = gamma * R + rewards[i]
                loss[j] = loss[j] - (log_probs[i][j]*(Variable(R.data.squeeze()).expand_as(log_probs[i][j])).to(device)).sum()
        loss = loss.sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()