import os
import random
from collections import deque
import numpy as np
import copy

import torch
from torch.autograd import Variable

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def store_transition(self, state, action, reward, state_, done):
        experience = (state, action, np.array([reward]), state_, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class OUActionNoise(object):
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

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def to_tensor(ndarray):
    return torch.FloatTensor(ndarray)

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

'''
Implementing Hindsight Experience Replay: https://arxiv.org/abs/1707.01495
to help with training with sparse rewards. The implementation referenced this:
https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/her/replay_buffer.py

:param replay_buffer: a replay buffer for storing and sampling past experiences
:param enable_her: use hindsight experience replay if set
:param reward_func: function to generate new reward for new goals
:param env: robosuite environemnt
'''
class ReplayBufferWithHindsight:
    def __init__(self, replay_buffer, enable_her = False,
    reward_func = None, env = None):
        super(ReplayBufferWithHindsight, self).__init__()

        self.replay_buffer = replay_buffer
        self.enable_her = enable_her
        self.reward_func = reward_func
        self.env = env
        self.tmp_transition_buffer = []

        print("Initialized Replay Buffer with [buffer size]: {}, [enable_her]: {},"
        " [reward function]: {}, [robosuite env]: {}".\
            format(len(replay_buffer), self.enable_her, self.reward_func, self.env))

    
    '''
    Function for storing transitions. If using plain buffer, we store transitions directly If using HER, we cache the 
    experiences in a temporary buffer until end the episode, and then do sampling and commit transitions with "new goals" into memory.

    :param state: robot arm state
    :param action: control action for robot arm
    :param reward: reward for current action
    :param state_: next robot arm state
    :param done: whether this episode is done
    '''
    def store_transition(self, state, action, reward, state_, done):
        experience = (state, action, reward, state_, done)
        if self.enable_her:
            self.tmp_transition_buffer.append(experience)
            if done:
                self._commit_tmp_transitions()
                self.tmp_transition_buffer = []
        else:
            self.replay_buffer.store_transition(*experience)

    '''
    Sampling from HER buffer is just the same as sampling from the underlying buffer
    :param batch_size: batch size to sample
    '''
    def sample(self, batch_size):
        return self.replay_buffer.sample(batch_size)

    '''
    Store the transition stored in tmp_transition_buffer into underlying replay_buffer
    Here's where we same the tmp_transition_buffer and create new goals and calculate new reward 
    '''
    def _commit_tmp_transitions(self):
        for experience_idx, experience in enumerate(self.tmp_transition_buffer):
            # store original experience
            self.replay_buffer.store_transition(*experience)
            # cannot sample from future if we are the last experience in the tmp buffer
            if (experience_idx == len(self.tmp_transition_buffer) - 1):
                break
            
            sampled_state = self._sample_state_for_goal(self.tmp_transition_buffer, experience_idx)
            state, action, reward, state_, done = copy.deepcopy(experience)

            # update new goal
            # assume state is composed of cancatenation of (robot_arm_obs, object_state), where object_state is actually goal here
            # also, assume length of the state is 42, with [0, 31] being for robot_arm_obs and [32, 41] for object_state
            # index [21, 22, 23] is 3D location for gripper, index [32,33,34] is 3D location for cube. We replace old cube location
            # with sampled gripper location as the new goal

            if len(sampled_state) != 42:
                raise Exception("length of state is expected to be 42, but found {}".format(len(sampled_state)))
            goal = copy.deepcopy(sampled_state[21:24])
            # [32, 34, 35] is the old cube location
            state[32:35] = goal     
            state_[32:35] = goal
            # calculate new reward
            reward = self.reward_func(goal, state, state_, self.env)
            done = False
            # store experience with new goal
            self.replay_buffer.store_transition(state, action, reward, state_, done)

    '''
    Sample a state from which to get new goal
    :param experience_buffer: all transitions in the current episode
    :param experience_idx: index of current experience inside experience_buffer
    '''
    def _sample_state_for_goal(self, experience_buffer, experience_idx):
        new_sample_idx = np.random.choice(np.arange(experience_idx + 1, len(experience_buffer)))
        sampled_experience = experience_buffer[new_sample_idx]
        state, action, reward, state_, done = sampled_experience
        return state

'''
Generate new reward based on logic outlined in reward() function in Lift environment
https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/lift.py#L202

reward() function in robosuite doesn't take into account for current state and next state. so
we have to reverse engineer it from source code using robosuite env variable and internal states

Below is internal state for robot and object (cube)

robot0_robot-state [ 0.01608289  0.22106382  0.00264442 -0.46841418  0.03275821  0.04317115
  0.72038251  0.99987066  0.97525934  0.9999965  -0.883509    0.99946331
 -0.99906769  0.69357699  0.03416938  0.16799118  0.02474135 -0.06182279
 -0.34365342  1.35041627  0.28726022 -0.09950834  0.00604963  1.00750317
  0.99386743 -0.01622762  0.10927269 -0.00486571  0.03010999 -0.03024107
  0.16840519 -0.1868636 ]
object-state [ 0.02238403  0.02294394  0.81805657  0.          0.          0.99922173
  0.03944523 -0.12189237 -0.01689431  0.1894466 ]
CHECKING False 1.0
REWARD 0.0
GRIPPER STATE: env.sim.data.site_xpos[env.robots[0].eef_site_id] [-0.09950834  0.00604963  1.00750317]
CUBE STATE: env.sim.data.body_xpos[env.cube_body_id] [0.02238403 0.02294394 0.81805657]

Gripper position in robot0_robot-state: idx (zero-based): 21, 22, 23
Cube position in object-state: idx (zero-based): 0, 1, 2

:param goal: 3D coordinate of goal.
:param cur_state: robot current state (robot_arm_obs, object_state)
:param next_state: next robot current state (robot_arm_obs, object_state)
:param env: robosuite environment
'''
def Lift_reward_func1(goal, cur_state, next_state, env):
    # return true if 3D points are closer than threshold (0.005)
    def closeEnough(pos1, pos2):
        if np.linalg.norm(np.subtract(pos1, pos2)) < 0.005:
            return True
        return False

    reward = float(0)

    # if we are at goal state, give all rewards
    if env._check_success():
        reward = 2.25
    else:
        # reward proximity of gripper to cube. current state and next state each get 0.5 reward for proximity
        if closeEnough((cur_state[21], cur_state[22], cur_state[23]), goal):
            reward += 0.5
        if closeEnough((next_state[21], next_state[22], next_state[23]), goal):
            reward += 0.5

        # todo: other rewards like grasp
    # Scale reward if requested
    if env.reward_scale is not None:
        reward *= env.reward_scale / 2.25
    return reward
